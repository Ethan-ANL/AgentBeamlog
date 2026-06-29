#!/usr/bin/env python3
"""beamlog_frames - optional area-detector frame capture for beamlog.

This module isolates the only *optional* runtime dependency in the project:
pvapy/pvaccess (and numpy, which pvapy pulls in). The core (beamlog.py, the GUI)
stays stdlib-only; nothing here is imported unless frame capture is actually
requested -- and pvaccess itself is imported lazily, inside capture(), so a box
without it still runs every other part of beamlog.

What it does: grab the *current* image from an EPICS areaDetector PVA "Image"
channel (e.g. "13SIM1:Pva1:Image") the moment a SPEC command completes, cache it
as a raw .npy (full dtype/bit-depth, the corpus artifact) plus an 8-bit
thumbnail PNG (for the GUI). The frame stays *pending* until a human keeps it.

Caveat (documented in the README): reading a log file means the captured frame
reflects detector state ~1 poll after the command finished -- "what the detector
showed around when this command completed", not a frame intrinsically bound to
the command.

A "synthetic" source (numpy gradient + noise, no pvapy) lets the whole pipeline
-- capture -> save -> DB row -> GUI thumbnail -> keep/discard -> sweep -- be
exercised without a beamline, the way replay_spec_log.py exercises `tail`.
"""

from __future__ import annotations

import os
import struct
import zlib

SYNTHETIC = "synthetic"  # magic frame_pv value: generate frames locally, no pvapy


class FrameError(Exception):
    """Capture failed (detector down, timeout, unsupported codec/color, no
    pvapy, ...). The caller treats any FrameError as 'no frame this time' and
    records it in the row's `error` column -- it never interrupts ingestion."""


# --------------------------------------------------------------------------- #
# numpy: required for any capture (raw arrays). Imported lazily so the module
# loads even where numpy is absent; only capture()/save use it.
# --------------------------------------------------------------------------- #

def _numpy():
    try:
        import numpy as np  # noqa: PLC0415 - lazy by design
        return np
    except Exception as e:  # noqa: BLE001
        raise FrameError(f"numpy not available: {e}")


def pvapy_available() -> bool:
    """True if pvaccess can be imported -- used by `bl resolve` as a sanity check."""
    try:
        import pvaccess  # noqa: F401, PLC0415
        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# capture
# --------------------------------------------------------------------------- #

# union member name (NTNDArray `value`) -> numpy dtype string
_UNION_DTYPE = {
    "booleanValue": "bool",
    "byteValue": "int8", "ubyteValue": "uint8",
    "shortValue": "int16", "ushortValue": "uint16",
    "intValue": "int32", "uintValue": "uint32",
    "longValue": "int64", "ulongValue": "uint64",
    "floatValue": "float32", "doubleValue": "float64",
}

_synthetic_counter = 0          # so successive synthetic frames differ
_channel_cache: dict = {}       # pv name -> live pva.Channel (reused across captures)


def capture(source: str, *, timeout: float = 1.0, shape=None, dtype="uint16") -> dict:
    """Fetch the current frame from `source`.

    `source` is either the configured PV name or the literal "synthetic".
    Returns {"array", "width", "height", "dtype", "unique_id", "pv"}.
    Raises FrameError on any failure (the caller records it and moves on).
    """
    if source == SYNTHETIC:
        return _capture_synthetic(shape=shape, dtype=dtype)
    return _capture_pva(source, timeout=timeout)


def _capture_synthetic(*, shape=None, dtype="uint16") -> dict:
    """A locally generated gradient+noise image -- no pvapy, no beamline."""
    global _synthetic_counter
    np = _numpy()
    h, w = shape if shape else (256, 256)
    _synthetic_counter += 1
    n = _synthetic_counter
    yy, xx = np.mgrid[0:h, 0:w].astype("float64")
    # a moving gaussian blob on a gradient, plus a little noise -- visibly
    # different each capture so the GUI/keep flow is easy to eyeball.
    cx = (w / 2) * (1 + 0.5 * np.sin(n / 3.0))
    cy = (h / 2) * (1 + 0.5 * np.cos(n / 5.0))
    blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (min(h, w) / 6) ** 2)))
    grad = (xx / max(w - 1, 1) + yy / max(h - 1, 1)) / 2
    rng = np.random.default_rng(n)
    img = 0.55 * blob + 0.35 * grad + 0.10 * rng.random((h, w))
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.integer):
        hi = float(np.iinfo(dt).max if dt != np.dtype("uint16") else 65535)
        arr = (img * hi).astype(dt)
    else:
        arr = img.astype(dt)
    return {
        "array": arr, "width": w, "height": h,
        "dtype": str(arr.dtype), "unique_id": n, "pv": SYNTHETIC,
    }


def _capture_pva(pv_name: str, *, timeout: float = 1.0) -> dict:
    """Grab one NTNDArray from a PVA Image channel and decode it to a numpy array.

    Mono (2-D) and RGB1 color are handled; compressed codecs and RGB2/RGB3 are
    guarded-and-skipped (raised as FrameError) rather than mis-decoded.
    """
    np = _numpy()
    try:
        import pvaccess as pva  # noqa: PLC0415 - the optional dependency
    except Exception as e:  # noqa: BLE001
        raise FrameError(f"pvaccess not installed: {e}")

    chan = _channel_cache.get(pv_name)
    try:
        if chan is None:
            chan = pva.Channel(pv_name)          # provider defaults to pva.PVA
            try:
                chan.setTimeout(timeout)          # newer builds; older use a default
            except Exception:                     # noqa: BLE001
                pass
            _channel_cache[pv_name] = chan
        try:
            chan.setUseNumPyArrays(True)          # numeric arrays come back as numpy
        except Exception:                         # noqa: BLE001
            pass
        pv = chan.get("field()")
    except Exception as e:                         # noqa: BLE001 - timeout/disconnect
        _channel_cache.pop(pv_name, None)          # force reconnect next time
        raise FrameError(f"PV get failed for {pv_name}: {e}")

    # compression: the value buffer is opaque without the matching C library.
    try:
        codec = (pv["codec"]["name"] or "").strip()
    except Exception:                              # noqa: BLE001 - field may be absent
        codec = ""
    if codec:
        raise FrameError(f"compressed frame (codec={codec!r}) not supported")

    try:
        dims = list(pv["dimension"])
    except Exception as e:                         # noqa: BLE001
        raise FrameError(f"no dimension field: {e}")
    if len(dims) not in (2, 3):
        raise FrameError(f"unsupported dimensionality ({len(dims)} dims)")

    # the active member of the `value` union -> flat pixel buffer + dtype
    try:
        value = dict(pv["value"])
    except Exception as e:                         # noqa: BLE001
        raise FrameError(f"no value field: {e}")
    member = next((k for k, v in value.items() if v is not None), None)
    if member is None:
        raise FrameError("empty value union")
    flat = np.asarray(value[member])
    np_dtype = _UNION_DTYPE.get(member)
    if np_dtype and flat.dtype != np.dtype(np_dtype):
        flat = flat.astype(np_dtype, copy=False)

    # areaDetector is fast-axis-first: dimension[0]=X (cols), dimension[1]=Y (rows).
    w = int(dims[0]["size"])
    h = int(dims[1]["size"])
    if len(dims) == 2:
        try:
            arr = flat.reshape(h, w)               # numpy is (rows, cols)
        except Exception as e:                     # noqa: BLE001
            raise FrameError(f"reshape {flat.size} -> ({h},{w}) failed: {e}")
    else:
        # color: only RGB1 (interleaved by pixel, dimension [3, W, H]) is handled.
        if int(dims[0]["size"]) == 3:
            w, h = int(dims[1]["size"]), int(dims[2]["size"])
            try:
                arr = flat.reshape(h, w, 3)
            except Exception as e:                 # noqa: BLE001
                raise FrameError(f"reshape RGB1 failed: {e}")
        else:
            raise FrameError("unsupported color mode (only RGB1 handled)")

    try:
        unique_id = int(pv["uniqueId"])
    except Exception:                              # noqa: BLE001
        unique_id = None

    return {
        "array": arr, "width": w, "height": h,
        "dtype": str(arr.dtype), "unique_id": unique_id, "pv": pv_name,
    }


# --------------------------------------------------------------------------- #
# storage: raw .npy (corpus) + 8-bit thumbnail PNG (GUI). Both written
# atomically (temp name -> os.replace) so a crash never leaves a row pointing at
# a half-written file.
# --------------------------------------------------------------------------- #

def _paths(frames_dir: str, action_id: int):
    return (
        os.path.join(frames_dir, f"frame_{action_id}.npy"),
        os.path.join(frames_dir, f"frame_{action_id}.png"),
    )


def save_pending(frames_dir: str, action_id: int, cap: dict):
    """Write the raw array (.npy) and an 8-bit thumbnail (.png). Returns
    (npy_path, png_path)."""
    np = _numpy()
    os.makedirs(frames_dir, exist_ok=True)
    npy_path, png_path = _paths(frames_dir, action_id)
    arr = cap["array"]

    tmp_npy = npy_path + ".tmp"
    np.save(tmp_npy, arr)                 # np.save appends .npy if missing...
    if os.path.exists(tmp_npy + ".npy"):  # ...so normalise the temp name
        tmp_npy += ".npy"
    os.replace(tmp_npy, npy_path)

    png = _to_png8(arr)
    tmp_png = png_path + ".tmp"
    with open(tmp_png, "wb") as f:
        f.write(png)
    os.replace(tmp_png, png_path)
    return npy_path, png_path


def discard(*paths):
    """Delete cached files, ignoring any that are already gone."""
    for p in paths:
        if not p:
            continue
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _to_png8(arr) -> bytes:
    """Encode a 2-D (mono) or 3-D (HxWx3) array to an 8-bit PNG, autoscaled by
    min/max. Uses PIL if importable, else a minimal stdlib zlib PNG writer."""
    np = _numpy()
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[2] == 3:
        gray = False
    elif a.ndim == 2:
        gray = True
    else:
        a = a.reshape(a.shape[0], -1)
        gray = True

    a = a.astype("float64")
    lo, hi = float(a.min()), float(a.max())
    if hi > lo:
        a = (a - lo) / (hi - lo)
    else:
        a = a * 0.0
    a8 = (a * 255.0 + 0.5).astype("uint8")

    try:  # PIL is a nicety, not a requirement
        from PIL import Image  # noqa: PLC0415
        mode = "L" if gray else "RGB"
        import io  # noqa: PLC0415
        buf = io.BytesIO()
        Image.fromarray(a8, mode).save(buf, format="PNG")
        return buf.getvalue()
    except Exception:  # noqa: BLE001 - fall back to the stdlib writer
        return _stdlib_png(a8, gray)


def _stdlib_png(a8, gray: bool) -> bytes:
    """Minimal PNG encoder (stdlib zlib+struct) for 8-bit gray or RGB arrays."""
    h = a8.shape[0]
    w = a8.shape[1]
    color_type = 0 if gray else 2          # 0=grayscale, 2=truecolor
    channels = 1 if gray else 3

    # raw scanlines, each prefixed with filter byte 0 (None)
    rows = bytearray()
    data = a8.tobytes()
    stride = w * channels
    for y in range(h):
        rows.append(0)
        rows.extend(data[y * stride:(y + 1) * stride])

    def chunk(tag: bytes, payload: bytes) -> bytes:
        out = struct.pack(">I", len(payload)) + tag + payload
        return out + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    idat = zlib.compress(bytes(rows), 6)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


# --------------------------------------------------------------------------- #
# sweeper helpers: keep undecided ("pending") frames from filling a disk.
# The DB-side decisions (which rows are pending/over-TTL) live in beamlog.py;
# these just measure/delete files.
# --------------------------------------------------------------------------- #

def dir_size_bytes(frames_dir: str) -> int:
    total = 0
    try:
        for name in os.listdir(frames_dir):
            try:
                total += os.path.getsize(os.path.join(frames_dir, name))
            except OSError:
                pass
    except OSError:
        pass
    return total

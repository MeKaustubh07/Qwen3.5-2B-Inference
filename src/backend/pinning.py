import ctypes
import functools

import torch

from quant import QuantTensor

_SHARED = 0                                               # MTLStorageModeShared


@functools.cache
def _runtime():
    """The Objective-C runtime and libc entry points (macOS only; loaded on first use, so importing this module
    works everywhere, e.g. in the Linux container, where nothing is on MPS and nothing gets locked)."""
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    objc.sel_registerName.restype, objc.sel_registerName.argtypes = ctypes.c_void_p, [ctypes.c_char_p]
    send_ptr = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
    send_u64 = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p)(("objc_msgSend", objc))
    libc = ctypes.CDLL(None, use_errno=True)
    for f in (libc.mlock, libc.munlock):
        f.argtypes, f.restype = [ctypes.c_void_p, ctypes.c_size_t], ctypes.c_int
    sel = {name: objc.sel_registerName(name.encode()) for name in ("contents", "length", "storageMode")}
    return send_ptr, send_u64, libc, sel


def model_tensors(model) -> list[torch.Tensor]:
    """Every tensor the backend has prepared for the model (its lazily filled weight cache; quantized weights as
    their data, scales and mins). Complete once every layer has run, e.g. after the scheduler's warm-up."""
    out = []
    for v in model._cache.values():
        if isinstance(v, QuantTensor):
            out += [v.data, v.scales] + ([v.mins] if v.mins is not None else [])
        else:
            out.append(v)
    return out


def _ranges(tensors: list[torch.Tensor]) -> list[tuple[int, int]]:
    """(CPU address, length) of each distinct MTLBuffer behind the MPS tensors; CPU tensors are skipped."""
    out, seen = [], set()
    for t in tensors:
        buf = t.untyped_storage().data_ptr()
        if t.device.type != "mps" or buf in seen:
            continue
        seen.add(buf)
        send_ptr, send_u64, _, sel = _runtime()
        if send_u64(buf, sel["storageMode"]) != _SHARED:
            raise ValueError("MPS buffer without shared storage: no CPU address to lock")
        out.append((send_ptr(buf, sel["contents"]), send_u64(buf, sel["length"])))
    return out


def lock_in_memory(tensors: list[torch.Tensor]) -> int:
    """mlock the pages of the MPS tensors' buffers; returns the bytes locked. Raises OSError if the OS refuses
    (e.g. over vm.user_wire_limit); pages locked before the failure stay locked."""
    total = 0
    for ptr, n in _ranges(tensors):
        if _runtime()[2].mlock(ptr, n) != 0:
            err = ctypes.get_errno()
            raise OSError(err, f"mlock of {n} bytes failed after {total} bytes locked")
        total += n
    return total


def unlock(tensors: list[torch.Tensor]) -> None:
    for ptr, n in _ranges(tensors):
        _runtime()[2].munlock(ptr, n)

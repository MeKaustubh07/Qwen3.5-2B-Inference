import ctypes
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
LIB_PATH = ROOT / "build" / "libmetal_runtime.dylib"
KERNELS = ROOT / "src" / "kernels"


class NativeMetal:
    def __init__(self):
        if not LIB_PATH.exists():
            raise FileNotFoundError(f"{LIB_PATH} missing; run scripts/build_native.sh")
        lib = ctypes.CDLL(str(LIB_PATH))
        lib.mr_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        lib.mr_device_name.restype = ctypes.c_char_p
        lib.mr_upload.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        lib.mr_alloc.argtypes = [ctypes.c_uint64]
        lib.mr_contents.argtypes = [ctypes.c_int]
        lib.mr_contents.restype = ctypes.c_void_p
        lib.mr_matvec_bf16.argtypes = [ctypes.c_int] * 3 + [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int]
        lib.mr_matvec_bf16.restype = ctypes.c_double
        lib.mr_matvec_bf16_rotate.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                              ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int]
        lib.mr_matvec_bf16_rotate.restype = ctypes.c_double
        err = ctypes.create_string_buffer(512)
        src = (KERNELS / "matvec.metal").read_bytes()           # the SAME kernel source the torch path uses
        if lib.mr_init(src, b"matvec_bf16", err, 512) != 0:
            raise RuntimeError(f"native Metal init failed: {err.value.decode()}")
        self.lib = lib
        self.device_name = lib.mr_device_name().decode()

    def upload(self, t: torch.Tensor) -> int:
        t = t.contiguous().cpu()
        return self.lib.mr_upload(t.data_ptr(), t.numel() * t.element_size())

    def alloc(self, nbytes: int) -> int:
        return self.lib.mr_alloc(nbytes)

    def view(self, handle: int, n: int, dtype=torch.float32) -> torch.Tensor:
        """A torch tensor aliasing the shared GPU buffer (unified memory: no copy)."""
        ptr = self.lib.mr_contents(handle)
        itemsize = torch.tensor([], dtype=dtype).element_size()
        raw = (ctypes.c_char * (n * itemsize)).from_address(ptr)
        return torch.frombuffer(raw, dtype=dtype, count=n)

    def matvec(self, w: int, x: int, y: int, N: int, K: int, iters: int = 1) -> float:
        return self.lib.mr_matvec_bf16(w, x, y, N, K, iters)

    def matvec_rotate(self, ws: list[int], x: int, y: int, N: int, K: int, iters: int) -> float:
        """GPU seconds per dispatch while cycling through distinct weight buffers (cold, DRAM-bound reads)."""
        arr = (ctypes.c_int * len(ws))(*ws)
        return self.lib.mr_matvec_bf16_rotate(arr, len(ws), x, y, N, K, iters)

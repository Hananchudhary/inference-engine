import ctypes
import numpy as np
import torch

lib = ctypes.CDLL("../../src/kernals/softmax.so")

lib.softmax.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ctypes.c_uint32
]

lib.softmax.restype = None

num_test = 1000
length = 100

passed = 0
for _ in range(num_test):
    a = np.random.rand(length)
    c = np.empty_like(a)

    lib.softmax(a, c, len(a))
    x = torch.tensor(a)
    res = torch.softmax(x, dim = 0)
    torch.set_printoptions(precision=17)
    np.set_printoptions(precision=17)
    try:
        assert (np.allclose(c, res, rtol=1e-9, atol=1e-12))
        passed +=1
    except AssertionError:
        pass
print(f"Passed: {passed}/{num_test}")
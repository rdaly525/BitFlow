from BitFlow.utils import LUTGenerator
import numpy as np

DOMAIN = np.pi


def myfunc(x):
    x = np.clip(x, -DOMAIN, +DOMAIN)
    return np.sin(x)


def test_LUTGenerator():
    lut = LUTGenerator(myfunc, [-np.pi, np.pi])
    print(lut)
    print(len(lut.lut))
    print(lut[5])

    return

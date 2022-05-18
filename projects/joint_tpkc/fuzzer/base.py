import datetime
import argparse
import platform
import datetime
import random
import time

import numpy as np
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
from itertools import product

import functions as FUNC

import log as LOG

if "常量":
    INT_INF = 2 ** 31 - 1
    LONG_INF = 2 ** 63 - 1
    FLOAT_INF = 1.7E37
    FLOAT_INF_64 = 1.7e308
    EPS = 1E-30
    NAN = float('nan')
    INF = float('inf')
    _INIT_DIR = "D:\learn\code\oss-fuzz\database"

if "基本类型":

    def INTEGERS(min_value=-INT_INF, max_value=INT_INF):
        return st.integers(min_value=min_value, max_value=max_value)


    def FLOATS(width=32, allow_nan=False, allow_inf=True, min_float=None, max_float=None):
        return st.floats(width=width, allow_nan=allow_nan, allow_infinity=allow_inf, min_value=min_float,
                         max_value=max_float)


    def NUMBERS(min_value=-INT_INF, max_value=INT_INF, width=32, allow_nan=False, allow_inf=True, only_int=False,
                only_float=False, min_float=None, max_float=None):
        floats = FLOATS(width=width, allow_nan=allow_nan, allow_inf=allow_inf, min_float=min_float,
                        max_float=max_float)
        ints = INTEGERS(min_value, max_value)

        if only_float:
            return floats
        elif only_int:
            return ints
        return st.one_of(ints, floats)


    def ARRAY_1D(min_value=-INT_INF, max_value=INT_INF, width=32, allow_nan=False, allow_inf=True, only_int=False,
                 only_float=False, min_size=0, max_size=100000, min_float=None, max_float=None):
        return st.lists(
            NUMBERS(
                min_value, max_value,
                width, allow_nan, allow_inf,
                only_int, only_float,
                min_float, max_float
            ),
            min_size=min_size, max_size=max_size
        )


    def SHAPE(min_dims=1, max_dims=5, min_side=1, max_side=10):
        return hnp.array_shapes(min_dims=min_dims, max_dims=max_dims, min_side=min_side, max_side=max_side)


    def ARRAY_ND(dtype=np.float32, shape=SHAPE(), elements=FLOATS()):
        return hnp.arrays(dtype=dtype, shape=shape, elements=elements)


    @st.composite
    def COMPLEX(draw, width=64, allow_nan=True, allow_inf=True):
        real = draw(FLOATS(width=width, allow_nan=allow_nan, allow_inf=allow_inf))
        imag = draw(FLOATS(width=width, allow_nan=allow_nan, allow_inf=allow_inf))
        return np.complex(real=real, imag=imag)


    def is_same(a, b):
        if np.isnan(a) or np.isnan(b):
            return np.isnan(a) and np.isnan(b)
        if np.isinf(a) or np.isinf(b):
            return a == b
        return a - b < EPS or b - a < EPS


    def erase_inf(_a, EPS=EPS, INF=INF):
        a = _a
        a = np.where(a > FLOAT_INF, a, INF)
        a = np.where(a < -FLOAT_INF, a, -INF)
        return a


    def erase_compinf(_a):
        a = _a
        real = np.real(a)
        imag = np.imag(a)
        real = np.where(real > FLOAT_INF_64, real, INF)
        real = np.where(real < -FLOAT_INF_64, real, -INF)
        imag = np.where(imag > FLOAT_INF_64, imag, INF)
        imag = np.where(imag < -FLOAT_INF_64, imag, -INF)
        return real + imag * (1j)


    def array_same(a, b, EPS=EPS, INF=INF, ifcomplex=False):
        if a.shape != b.shape:
            return False
        if not ifcomplex:
            a = erase_inf(a)
            b = erase_inf(b)
        else:
            a = erase_compinf(a)
            b = erase_compinf(b)

        r1 = (np.isnan(a) & np.isnan(b))
        r2 = (np.isinf(a) & np.isinf(b) & (a == b))
        r3 = (np.isfinite(b) & np.isfinite(b)) & np.isclose(a, b, EPS)
        return np.all(r1 | r2 | r3)

    def all_same(ary:list,EPS=EPS,INF=INF,ifcomplex=False):
        for i,it in enumerate(ary[:-1]):
            for j,jt in enumerate(ary[i+1:]):
                if not array_same(it,jt,EPS,INF,ifcomplex):
                    return False
        return  True

    def toFloat(x):
        return np.float32(x)

if "帮助函数":
    PLATFORM = FUNC.PLATFORM

    self = FUNC.self

    shape_space = FUNC.shape_space

    index_except = FUNC.index_except

    index_join = FUNC.index_join

    get = FUNC.get

    trans = FUNC.trans

    new_name = FUNC.new_name

    now_time = FUNC.now_time

if "日志输出":
    _cout = LOG.Log(headline="STD >>> ", console=True)


    def logHead():
        _cout.logHead()


    def log(*argc, **argv):
        _cout.log(*argc, **argv)


    def logEnd():
        _cout.logEnd()

if "运行参数":
    PARSER = argparse.ArgumentParser("pytorch测试专用参数")
    PARSER.add_argument("--test_model", help="定义测试模式", required=False, default="none", type=str)
    ARGS, _ = PARSER.parse_known_args()

if "暂存测试用例":
    pass

if __name__ == "__main__":
    if 0:
        inf = INF
        nan = NAN
        a = np.array([[[nan], [inf]], [[nan], [inf]]])
        b = np.array([[[nan], [nan]], [[inf], [inf]]])
        print(array_same(a, b))
    if 0:
        import torch
        import tensorflow

        for i in range(100):
            x = 10.0 ** i
            print(f"x={x}", torch.tensor(x), tensorflow.constant(x))

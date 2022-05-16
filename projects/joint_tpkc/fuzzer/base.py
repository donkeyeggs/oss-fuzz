import itertools
import math
import argparse
import platform

import numpy as np
import hypothesis as hyp
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
from itertools import product
from hypothesis import given, settings, example

if "基本类型":
    INT_INF = 2 ** 31 - 1
    LONG_INF = 2 ** 63 - 1
    FLOAT_INF = 1.7E37
    EPS = 1E-30
    NAN = float('nan')
    INF = float('inf')


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


    def is_same(a, b):
        if np.isnan(a) or np.isnan(b):
            return np.isnan(a) and np.isnan(b)
        if np.isinf(a) or np.isinf(b):
            return a == b
        return a - b < EPS or b - a < EPS


    def erase_inf(_a, EPS=EPS, INF=INF):
        a = _a
        a = np.where((a > FLOAT_INF), a, INF)
        a = np.where((a < -FLOAT_INF), a, -INF)
        return a


    def array_same(a, b, EPS=EPS, INF=INF):
        if a.shape != b.shape :
            return False
        a = erase_inf(a)
        b = erase_inf(b)
        r1 = (np.isnan(a) & np.isnan(b))
        r2 = (np.isinf(a) & np.isinf(b) & (a == b))
        r3 = (np.isfinite(b) & np.isfinite(b)) & np.isclose(a, b, EPS)
        return np.all(r1 | r2 | r3)


    def toFloat(x):
        return np.float32(x)
if "帮助函数":
    def PLATFORM():
        return platform.system().lower()


    def self(x):
        return x


    def shape_space(shape):
        return product(*[range(i) for i in shape])


    def index_except(index, dim):
        return index[:dim] + index[dim + 1:]


    def index_join(index, dim, i):
        return index[:dim] + [int(i)] + index[dim:]


    def get(array, index):
        v = array
        for i in index:
            v = v[i]
        return v


    def trans(tensor, cond=self):
        ret = []
        for index in shape_space(tensor.shape):
            item = get(tensor, index)
            ret.append((list(index), cond(item)))
        return ret

if "日志输出":
    def logHead():
        print("\033[1;34;32m == Logging == \033[0m")


    def log(*argc, end='\n'):
        print("\033[1;34;32mDEBUG >>>\033[0m", *argc, end=end)


    def logEnd():
        print()

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
    if 1:
        import torch
        import tensorflow

        for i in range(100):
            x = 10.0 ** i
            print(f"x={x}", torch.tensor(x), tensorflow.constant(x))

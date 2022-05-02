import argparse
from ctypes.wintypes import INT
import hypothesis as hyp
import hypothesis.strategies as st
from hypothesis import given,settings,example
from hypothesis.extra._array_helpers import array_shapes
from sqlalchemy import FLOAT

if "基础类型": 
    INT_INF = 2**31-1
    LONG_INF = 2**63-1
    EPS = 1E-6

    INTEGERS = st.integers(min_value=-INT_INF,max_value=INT_INF)
    FLOATS = st.floats(width=32)
    NUMBERS = st.one_of(INTEGERS,FLOATS)
    TENSOR = st.lists(NUMBERS)
    TENSORSHAPE = array_shapes(min_dims=1,max_dims=5,min_side=1,max_side=11)
    def equals(a,b):
        return a-b<EPS or b-a<EPS



PARSER = argparse.ArgumentParser("pytorch测试专用参数")
PARSER.add_argument("--test_model",help="定义测试模式",required=False,default="none",type=str)

ARGS = PARSER.parse_args()
def logHead():
    print("\033[1;34;32m == Logging == \033[0m")
def log(*argc,end='\n'):
    print("\033[1;34;32mDEBUG >>>\033[0m",*argc,end=end)
def logEnd():
    print()

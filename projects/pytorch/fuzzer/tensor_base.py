import itertools
import math
import argparse
import platform

import numpy as np
import hypothesis as hyp
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
from itertools import product 
from hypothesis import given,settings,example
#from sqlalchemy import FLOAT


if "基础类型": 
    INT_INF = 2**31-1
    LONG_INF = 2**63-1
    EPS = 1E-6
    NAN = float('nan')
    INF = float('inf')

    INTEGERS = st.integers(min_value=-INT_INF,max_value=INT_INF)
    FLOATS = st.floats(width=32,allow_nan=False)
    NUMBERS = st.one_of(INTEGERS,FLOATS)
    TENSOR = st.lists(NUMBERS)
    TENSORSHAPE = hnp.array_shapes(min_dims=1,max_dims=5,min_side=1,max_side=10)
    MULITTENSOR = hnp.arrays(dtype=np.float32,shape=TENSORSHAPE,elements=FLOATS)


    def equals(a,b):
        if np.isnan(a) or np.isnan(b):
            return np.isnan(a) and np.isnan(b)
        if np.isinf(a) or np.isinf(b):
            return a==b
        #print(a,b,a-b,b-a)
        return a-b<EPS or b-a<EPS

if "帮助函数":
    PLATFORM = platform.system().lower()

    def self(x):
        return x
    def shape_space(shape):
        return product(*[range(i) for i in shape])
    def index_except(index,dim):
        return index[:dim]+index[dim+1:]
    def index_join(index,dim,i):
        return index[:dim]+[int(i)]+index[dim:]
    def get(a,index):
        v = a
        for i in index:
            v=v[i]
        return v
    '''
    def _trans(tensor,lshape,dim,index,cond=self):
        if dim == lshape:
            return (index,cond(tensor))
        ret = []
        for (i,item) in enumerate(tensor):
            v = _trans(item,lshape,dim+1,index+[i],cond)
            if dim==lshape-1:
                ret.append(v)
            else:
                ret.extend(v)
        return ret
    '''
    def trans(tensor,cond=self):
        ret =[]
        for index in shape_space(tensor.shape):
            item = get(tensor,index)
            ret.append((list(index),cond(item)))
        return ret

    


if "日志输出":
    def logHead():
        print("\033[1;34;32m == Logging == \033[0m")
    def log(*argc,end='\n'):
        print("\033[1;34;32mDEBUG >>>\033[0m",*argc,end=end)
    def logEnd():
        print()

if "运行参数":
    PARSER = argparse.ArgumentParser("pytorch测试专用参数")
    PARSER.add_argument("--test_model",help="定义测试模式",required=False,default="none",type=str)

    ARGS,_ = PARSER.parse_known_args()

if "暂存测试用例":
    pass

if __name__=="__main__":
    import torch
    a = torch.tensor(np.random.random(27).reshape((3,3,3)))
    print(trans(a))

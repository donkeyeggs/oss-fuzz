import sys
import atheris
import traceback
import numpy as np
import tensor_base as TEST
import hypothesis.strategies as st
from hypothesis import given,settings,assume
with atheris.instrument_imports():
    import torch
K_RANGE = st.integers(min_value=0,max_value=9)
DIM_RANGE = st.integers(min_value=1,max_value=5)

def check_indices(input,dim,selects,indices):
    indice = TEST.trans(indices)
    for index,item in indice:
        rindex = index[:dim]+[int(item)]+index[dim:]
        a = TEST.get(input,rindex)
        b = TEST.get(selects,index)
        if not TEST.equals(a,b):
            return False
    return True

def check_values(input,k,dim,selects):
    lshape = len(input.shape)
    ldim = input.shape[dim]
    for index,item in TEST.trans(selects):
        count_min = 0
        count_equal = 0
        for i in range(ldim):
            rindex = TEST.index_join(index,dim,i)
            value = TEST.get(input,rindex)
            if TEST.equals(value,item):
                count_equal+=1
            elif value<item:
                count_min+=1
        if not ((1+count_min)<=(k)<(1+count_min+count_equal)):
            return False
    return True
    
def check_kthvalue(input,k,dim,selects,indices):
    assert check_indices(input,dim,selects,indices)
    assert check_values(input,k,dim,selects)
    return True

@settings(max_examples=10)
@given(_input=TEST.MULITTENSOR,_k=K_RANGE,_dim=DIM_RANGE)
def torch_kthvalue_test(_input,_k,_dim):
    input = torch.tensor(np.array(_input))
    k = _k
    dim = _dim
    shape = input.shape
    assume(0<=dim<len(shape))
    assume(1<=k<=shape[dim])
    ret = torch.kthvalue(input=input,k=k,dim=dim)
    if TEST.ARGS.test_model == "debug":
        TEST.logHead()
        TEST.log(f"input: {input}")
        TEST.log(f"shape: {shape}")
        TEST.log(f"k: {k}")
        TEST.log(f"dim: {dim}")
        TEST.log(f"ret: {ret}")
        TEST.logEnd()

    assert check_kthvalue(input,k,dim,ret[0],ret[1])

fuzz_target = atheris.instrument_func(torch_kthvalue_test.hypothesis.fuzz_one_input)

if __name__ == "__main__":
    atheris.Setup(sys.argv,fuzz_target)
    atheris.Fuzz()
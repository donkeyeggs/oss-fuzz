import sys

import tensor_base as TEST
import platform
import hypothesis.strategies as st
from hypothesis import given,settings
#with atheris.instrument_imports():
import torch

#TODO: 尚未测试float64状态下的精度
FLOATS = st.floats(min_value=-1.0,max_value=1.0,width=32) 
TENSOR = st.lists(FLOATS,min_size=1)

@given(x=TENSOR)
def test_acos(x):
    if TEST.ARGS.test_model == "debug":
        TEST.logHead()
        TEST.log("x:",x)
        TEST.log("tensor x:",torch.tensor(x))
        TEST.log("acos tensor x:",torch.acos(torch.tensor(x)))
        TEST.log("cos acos tensor x:",torch.cos(torch.acos(torch.tensor(x))))
        TEST.log("tensor x == cos acos tensor x:",torch.tensor(x)==torch.cos(torch.acos(torch.tensor(x))))
        TEST.logEnd()
    x = torch.tensor(x)
    assert TEST.equals(torch.cos(torch.acos(x)),x)


if __name__ == '__main__' and platform.system().lower() == "linux":
    import atheris
    fuzz_target = atheris.instrument_func(test_acos.hypothesis.fuzz_one_input)
    atheris.Setup(sys.argv,fuzz_target)
    atheris.Fuzz()
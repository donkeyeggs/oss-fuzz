import sys
import atheris
import tensor_base as TEST
import hypothesis.strategies as st
from hypothesis import given,settings
#with atheris.instrument_imports():
import torch

Numbers = st.one_of(
    st.floats(allow_infinity=False,allow_nan=False,width=32),
    TEST.INTEGERS
)
Tensor = st.lists(Numbers)

@settings(max_examples=10)
@given(_x=Tensor)
def torch_abs_test(_x):
    x = torch.tensor(_x)
    y = torch.abs(x)
    if TEST.ARGS.test_model == "debug":
        TEST.logHead()
        TEST.log("x:",_x)
        TEST.log("tensor x:",x)
        TEST.log("abs tensor x: ",y)
        TEST.log("all-equals:",[j>=0 and (i+j==0 or i==j) for i,j in zip(x,y)])
        TEST.logEnd()
    
    assert all([j>=0 and (i+j==0 or i==j) for i,j in zip(x,y) ])

fuzz_tareget = atheris.instrument_func(torch_abs_test.hypothesis.fuzz_one_input)
if __name__ == '__main__':
    atheris.Setup(sys.argv,fuzz_tareget)
    atheris.Fuzz()
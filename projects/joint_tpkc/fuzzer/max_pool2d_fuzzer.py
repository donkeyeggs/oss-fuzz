import sys
import numpy
import base as TEST
import traceback
import hypothesis.strategies as st
from hypothesis import given, settings, assume, example

import torch
import tensorflow

@st.composite
def input_data(draw):
    max_batch = 5
    max_channel = 5
    max_data_len = 100
    batch = draw(st.integers(min_value=2,max_value=max_batch))
    in_channels = draw(st.integers(min_value=2,max_value=max_channel))
    H_len = draw(st.integers(min_value=2,max_value=max_data_len))
    W_len = draw(st.integers(min_value=2,max_value=max_data_len))

    kH_len = draw(st.integers(min_value=2, max_value=H_len))
    kW_len = draw(st.integers(min_value=2, max_value=W_len))

    input_shape = (batch,in_channels,H_len,W_len)
    input = draw(TEST.ARRAY_ND(shape = input_shape))

    kernel_size= (kH_len,kW_len)

    padding = draw(
        st.integers(min_value=0,max_value=min(kW_len,kH_len)//2)
    )
    stride = draw(
        st.one_of(
            st.integers(min_value=1, max_value=max(H_len,W_len)+10),
            st.tuples(
                st.integers(min_value=1, max_value=H_len+10),
                st.integers(min_value=1, max_value=W_len+10)
            )
        )
    )
    return (input,kernel_size,stride,padding)
    pass

def assert_equals(_a,_b):
    a = numpy.array(_a)
    b = numpy.array(_b)
    if a.shape != b.shape :
        return False
    cmp = TEST.array_same(a,b)
    if not cmp:
        TEST.logHead()
        TEST.log(f"cmp:={cmp}")
        TEST.log(f"\n(shape a)={a.shape}\n(shape b)={b.shape}")
        TEST.log(f"\na={a}\nb={b}")
        TEST.logEnd()
    return cmp

def test_torch(_input,_kernel_size,_stride,_padding):
    input = torch.tensor(_input)
    kernel_size = _kernel_size
    stride = _stride
    padding = _padding
    output = torch.nn.functional.max_pool2d(
        input = input, 
        kernel_size = kernel_size, 
        stride = stride, 
        padding = padding, 
    )
    return output

def test_tensorflow(_input,_kernel_size,_stride,_padding):
    input = tensorflow.constant(_input)
    kernel_size = _kernel_size
    stride = _stride
    padding = _padding
    output = tensorflow.nn.max_pool2d(
        input = input, 
        ksize = kernel_size, 
        strides = stride, 
        padding = [[0,0],[0,0],[padding,padding],[padding,padding]], 
        data_format = "NCHW"
    )
    return output
    
@settings(max_examples=100, deadline=10000)
@given(_input=input_data())
def test_max_pool2d(_input):
    (input,kernel_size,stride,padding) = _input
    torch_output = test_torch(input,kernel_size,stride,padding)
    tensorflow_output = test_tensorflow(input,kernel_size,stride,padding)
    assertation = assert_equals(torch_output, tensorflow_output)
    assert assertation

TEST.log("running on", TEST.PLATFORM())
if __name__ == "__main__" and TEST.PLATFORM() == "windows":
    test_max_pool2d()
    pass

if __name__ == "__main__" and TEST.PLATFORM() == "linux":
    import atheris

    fuzz_target = atheris.instrument_func(test_max_pool2d.hypothesis.fuzz_one_input)
    atheris.Setup(sys.argv, fuzz_target)
    atheris.Fuzz()

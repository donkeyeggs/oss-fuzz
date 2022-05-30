import sys
import numpy
import paddle

import base as TEST
import log as LOG
import traceback
import hypothesis.strategies as st
from hypothesis import given, settings, assume, example

import torch
import tensorflow
cout = LOG.Log('max_pool3d_fuzzer',log_dir=TEST._INIT_DIR)
@st.composite
def input_data(draw):
    max_batch = 5
    max_channel = 5
    max_data_len = 100
    batch = draw(st.integers(min_value=2, max_value=max_batch))
    in_channels = draw(st.integers(min_value=2, max_value=max_channel))
    H_len = draw(st.integers(min_value=2, max_value=max_data_len))
    W_len = draw(st.integers(min_value=2, max_value=max_data_len))
    D_len = draw(st.integers(min_value=2, max_value=max_data_len))

    kH_len = draw(st.integers(min_value=2, max_value=H_len))
    kW_len = draw(st.integers(min_value=2, max_value=W_len))
    kD_len = draw(st.integers(min_value=2, max_value=D_len))

    input_shape = (batch, in_channels, H_len, W_len, D_len)
    input = draw(TEST.ARRAY_ND(shape=input_shape))

    kernel_size = (kH_len, kW_len, kD_len)

    #padding = draw(
    #    st.integers(min_value=0, max_value=min(kW_len, kH_len, kD_len) // 2)
    #)
    padding = 0
    stride = draw(
        st.one_of(
            st.integers(min_value=1, max_value=max(H_len, W_len, D_len) + 10),
            st.tuples(
                st.integers(min_value=1, max_value=H_len + 10),
                st.integers(min_value=1, max_value=W_len + 10),
                st.integers(min_value=1, max_value=D_len + 10),
            )
        )
    )
    return (input, kernel_size, stride, padding)
    pass


def assert_equals(_a, _b, _c):
    a = numpy.array(_a)
    b = numpy.array(_b)
    c = numpy.array(_c)
    # TEST.log("assert_equal ",numpy.shape(a)!=numpy.shape(b))
    ret = TEST.all_same([a, b, c])
    if not ret:
        cout.logHead()
        cout.log(f"(a)={a.shape} (b)={b.shape} (c)={c.shape}")
        cout.log(f"\na={a} \nb={b} \n c={c}")
        cout.log_empty()
        cout.logEnd()
        pass
    return ret

def test_paddle(_input,_kernel_size,_stride,_padding):
    input = paddle.to_tensor(_input)
    kernel_size = _kernel_size
    stride = _stride
    padding = _padding
    output = paddle.nn.functional.max_pool3d(
        x=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        data_format="NCDHW",
    )
    return output

def test_torch(_input, _kernel_size, _stride, _padding):
    input = torch.tensor(_input)
    kernel_size = _kernel_size
    stride = _stride
    padding = _padding
    output = torch.nn.functional.max_pool3d(
        input=input,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )
    return output


def test_tensorflow(_input, _kernel_size, _stride, _padding):
    input = tensorflow.constant(_input)
    kernel_size = _kernel_size
    stride = _stride
    padding = _padding
    output = tensorflow.nn.max_pool3d(
        input=input,
        ksize=kernel_size,
        strides=stride,
        padding='VALID' if padding==0 else 'SAME',
        data_format="NCDHW"
    )
    return output


@settings(max_examples=100, deadline=10000)
@given(_input=input_data())
def test_max_pool3d(_input):
    (input, kernel_size, stride, padding) = _input
    torch_output = test_torch(input, kernel_size, stride, padding)
    tensorflow_output = test_tensorflow(input, kernel_size, stride, padding)
    paddle_output = test_paddle(input, kernel_size, stride, padding)
    assertation = assert_equals(torch_output, tensorflow_output, paddle_output)
    assert assertation


TEST.log("running on", TEST.PLATFORM())
if __name__ == "__main__" and TEST.PLATFORM() == "windows":
    test_max_pool3d()
    pass

if __name__ == "__main__" and TEST.PLATFORM() == "linux":
    import atheris

    fuzz_target = atheris.instrument_func(test_max_pool3d.hypothesis.fuzz_one_input)
    atheris.Setup(sys.argv, fuzz_target)
    atheris.Fuzz()

import sys
import numpy
import paddle

import base as TEST
import traceback
import hypothesis.strategies as st
from hypothesis import given, settings, assume, example
import log as LOG
import warnings
warnings.filterwarnings("ignore")

import torch
import tensorflow

cout = LOG.Log("ifft2d_fuzzer",log_dir=TEST._INIT_DIR)

@st.composite
def input_data(draw):
    input = draw(
        TEST.ARRAY_ND(
            dtype=numpy.complex,
            shape=TEST.SHAPE(min_dims=2,max_dims=2,min_side=1,max_side=80),
            elements=TEST.COMPLEX()
        )
    )
    return input


def test_torch(_input):
    input = torch.tensor(_input)
    output = torch.fft.ifft2(
        input=input
    )
    return output


def test_tensorflow(_input):
    input = tensorflow.constant(_input)
    output = tensorflow.raw_ops.IFFT2D(
        input=input
    )
    return output

def test_paddle(_input):
    input = paddle.to_tensor(_input)
    output = paddle.fft.ifft2(
        input
    )
    return output

def assert_equals(_a, _b, _c):
    a = numpy.array(_a)
    b = numpy.array(_b)
    c = numpy.array(_c)
    # TEST.log("assert_equal ",numpy.shape(a)!=numpy.shape(b))
    ret = TEST.all_same([a, b, c],ifcomplex=True)
    if not ret:
        cout.logHead()
        cout.log(f"(a)={a.shape} (b)={b.shape} (c)={c.shape}")
        cout.log(f"\na={a} \nb={b} \n c={c}")
        cout.log_empty()
        cout.logEnd()
        pass
    return ret


@settings(max_examples=100, deadline=10000)
@given(_input=input_data())
def test_ifft2d(_input):
    input = _input
    torch_output = test_torch(input)
    tensorflow_output = test_tensorflow(input)
    paddle_output = test_paddle(input)
    assertation = assert_equals(torch_output, tensorflow_output, paddle_output)
    assert assertation



torch.set_default_dtype(torch.float64)

if __name__ == "__main__" and TEST.PLATFORM() == "windows":
    TEST.log("running on", TEST.PLATFORM())
    TEST.log("only could in float64(tensorflow)")
    test_ifft2d()
    pass

if __name__ == "__main__" and TEST.PLATFORM() == "linux":
    import atheris

    fuzz_target = atheris.instrument_func(test_ifft2d.hypothesis.fuzz_one_input)
    atheris.Setup(sys.argv, fuzz_target)
    atheris.Fuzz()

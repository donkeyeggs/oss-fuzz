import sys
import numpy
import paddle

import base as TEST
import log as LOG
import traceback
import hypothesis.strategies as st
from hypothesis import given, settings, assume, example
import warnings
warnings.filterwarnings("ignore")

import torch
import tensorflow

cout = LOG.Log('ifft_fuzzer',log_dir=TEST._INIT_DIR)
@st.composite
def input_data(draw):
    input = draw(
        st.lists(
            TEST.COMPLEX(),
            min_size=1
        )
    )
    return input


def test_torch(_input):
    input = torch.tensor(_input)
    output = torch.fft.ifft(
        input=input
    )
    return output

def test_paddle(_input):
    input = paddle.to_tensor(_input)
    output = paddle.fft.ifft(
        input
    )
    return output

def test_tensorflow(_input):
    input = tensorflow.constant(_input)
    output = tensorflow.raw_ops.IFFT(
        input=input
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
def test_ifft(_input):
    input = _input
    torch_output = test_torch(input)
    tensorflow_output = test_tensorflow(input)
    paddle_output = test_paddle(input)
    assertation = assert_equals(torch_output, tensorflow_output, paddle_output)
    assert assertation



torch.set_default_dtype(torch.float64)

if __name__ == "__main__" and TEST.PLATFORM() == "windows":
    cout.log("running on", TEST.PLATFORM())
    cout.log("only could in float64(tensorflow)")
    test_ifft()
    pass

if __name__ == "__main__" and TEST.PLATFORM() == "linux":
    import atheris

    fuzz_target = atheris.instrument_func(test_ifft.hypothesis.fuzz_one_input)
    atheris.Setup(sys.argv, fuzz_target)
    atheris.Fuzz()

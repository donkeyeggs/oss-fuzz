import sys
import numpy
import paddle

import traceback
import hypothesis.strategies as st
from hypothesis import given, settings, assume, example
import log as LOG
import base as TEST
import warnings

# warnings.filterwarnings("ignore")

import torch
import tensorflow

cout = LOG.Log("softmax_fuzzer", log_dir=TEST._INIT_DIR)


def test_torch(_input, _dim):
    input = torch.tensor(_input)
    dim = _dim
    output = torch.nn.functional.softmax(input=input, dim=dim)
    return output


def test_tensorflow(_logits, _axis):
    logits = tensorflow.constant(_logits)
    axis = _axis
    output = tensorflow.nn.softmax(logits=logits, axis=axis)
    return output


def test_paddle(_input, _dim):
    input = paddle.to_tensor(_input)
    dim = _dim
    output = paddle.nn.functional.softmax(x=input, axis=dim)
    return output


def assert_equals(_a, _b, _c):
    a = numpy.array(_a)
    b = numpy.array(_b)
    c = numpy.array(_c)
    ret = TEST.all_same([a, b, c])
    if not ret:
        cout.logHead()
        cout.log(f"(a)={a.shape} (b)={b.shape} (c)={c.shape}")
        cout.log(f"\na={a} \nb={b} \n c={c}")
        cout.log_empty()
        cout.logEnd()
        pass
    return ret


@st.composite
def make_input(draw):
    shape = draw(TEST.SHAPE(min_dims=1, max_dims=5, min_side=1, max_side=100))
    lens = len(shape)
    input = draw(TEST.ARRAY_ND(shape=shape, elements=TEST.FLOATS(min_float=-1, max_float=1, allow_inf=False)))
    dim = draw(st.integers(min_value=0, max_value=lens - 1))
    return (input, dim)


@settings(max_examples=1000, deadline=10000)
@given(_input=make_input())
def test_softmax(_input):
    input, dim = _input
    torch_output = test_torch(input, dim)
    tensorflow_output = test_tensorflow(input, dim)
    paddle_output = test_paddle(input, dim)
    assertation = assert_equals(torch_output, tensorflow_output, paddle_output)
    assert assertation


TEST.log("running on", TEST.PLATFORM())
if __name__ == "__main__" and TEST.PLATFORM() == "windows":
    test_softmax()
    pass

if __name__ == "__main__" and TEST.PLATFORM() == "linux":
    import atheris

    fuzz_target = atheris.instrument_func(test_softmax.hypothesis.fuzz_one_input)
    atheris.Setup(sys.argv, fuzz_target)
    atheris.Fuzz()

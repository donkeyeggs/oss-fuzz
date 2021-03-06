import sys
import numpy
import base as TEST
import log as LOG
import traceback
import hypothesis.strategies as st
from hypothesis import given, settings, assume,example

import torch
import tensorflow
import paddle

@st.composite
def input_data(draw):
    input = draw(TEST.ARRAY_ND(elements=TEST.FLOATS(allow_nan=True,allow_inf=True)))
    return input
def test_torch(_input):
    input = torch.tensor(_input)
    output = torch.nn.functional.relu(input)
    return output
def test_tensorflow(_input):
    input = tensorflow.constant(_input)
    output = tensorflow.nn.relu(input)
    return output
def test_paddle(_input):
    input = paddle.to_tensor(_input)
    output = paddle.nn.functional.relu(input)
    return output

def assert_equals(_a, _b, _c):
    a = numpy.array(_a)
    b = numpy.array(_b)
    c = numpy.array(_c)
    return TEST.all_same([a,b,c])


@settings(max_examples=100,deadline=10000)
@given(_input = input_data())
def test_relu(_input):
    torch_output = test_torch(_input)
    tensorflow_output = test_tensorflow(_input)
    paddle_output = test_paddle(_input)
    assertation = assert_equals(torch_output, tensorflow_output, paddle_output)
    assert assertation

TEST.log("running on", TEST.PLATFORM())
if __name__ == "__main__" and TEST.PLATFORM() == "windows":
    test_relu()
    pass

if __name__ == "__main__" and TEST.PLATFORM() == "linux":
    import atheris

    fuzz_target = atheris.instrument_func(test_relu.hypothesis.fuzz_one_input)
    atheris.Setup(sys.argv, fuzz_target)
    atheris.Fuzz()
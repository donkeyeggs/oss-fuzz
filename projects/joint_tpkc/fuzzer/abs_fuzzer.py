import sys
import numpy
import base as TEST
import traceback
import hypothesis.strategies as st
from hypothesis import given, settings, assume,example

import torch
import tensorflow


def test_torch(_input):
    input = torch.tensor(_input)
    output = torch.abs(input)
    return output


def test_tensorflow(_input):
    input = tensorflow.constant(_input)
    output = tensorflow.math.abs(input)
    return output


def assert_equals(_a, _b):
    a = numpy.array(_a)
    b = numpy.array(_b)
    return numpy.all(a == b)

# (-inf,-1e-37]U[1e37,inf)
input_float = st.one_of(
    TEST.FLOATS(max_float=TEST.toFloat(-1E-37)),
    TEST.FLOATS(min_float=TEST.toFloat(1E-37)),
)

@settings(max_examples=10, deadline=10000)
@given(_input=TEST.ARRAY_ND(elements=input_float))
#@example(_input=[1.0000002153053333e-39])
def test_abs(_input):
    #input = numpy.array(_input)
    #TEST.log(input)
    #assume(numpy.all(input > 1E-5))
    torch_output = test_torch(_input)
    tensorflow_output = test_tensorflow(_input)
    assertation = assert_equals(torch_output, tensorflow_output)
    #TEST.log("assertation", assertation)
    assert assertation


TEST.log("running on", TEST.PLATFORM())
if __name__ == "__main__" and TEST.PLATFORM() == "windows":
    test_abs()
    pass

if __name__ == "__main__" and TEST.PLATFORM() == "linux":
    import atheris

    fuzz_target = atheris.instrument_func(test_abs.hypothesis.fuzz_one_input)
    atheris.Setup(sys.argv, fuzz_target)
    atheris.Fuzz()

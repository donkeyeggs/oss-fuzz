import sys
import numpy
import base as TEST
import traceback
import hypothesis.strategies as st
from hypothesis import given, settings, assume, example

import torch
import tensorflow


def test_torch(_input):
    input = torch.tensor(_input)
    output = torch.acos(input)
    return output


def test_tensorflow(_input):
    input = tensorflow.constant(_input)
    output = tensorflow.math.acos(input)
    return output


def assert_equals(_a, _b):
    a = numpy.array(_a)
    b = numpy.array(_b)
    return TEST.array_same(a,b, EPS=1E-4)


@settings(max_examples=1000, deadline=10000)
@given(_input=TEST.ARRAY_ND(elements=TEST.FLOATS(min_float=-1,max_float=1)))
def test_acos(_input):
    torch_output = test_torch(_input)
    tensorflow_output = test_tensorflow(_input)
    assertation = assert_equals(torch_output, tensorflow_output)
    #TEST.log("torch output =",torch_output)
    #TEST.log("tensorflow output =",tensorflow_output)
    assert assertation


TEST.log("running on", TEST.PLATFORM())
if __name__ == "__main__" and TEST.PLATFORM() == "windows":
    test_acos()
    pass

if __name__ == "__main__" and TEST.PLATFORM() == "linux":
    import atheris

    fuzz_target = atheris.instrument_func(test_acos.hypothesis.fuzz_one_input)
    atheris.Setup(sys.argv, fuzz_target)
    atheris.Fuzz()

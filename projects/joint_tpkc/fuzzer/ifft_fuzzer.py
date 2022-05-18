import sys
import numpy as np
import base as TEST
import traceback
import hypothesis.strategies as st
from hypothesis import given, settings, assume, example
import warnings
warnings.filterwarnings("ignore")

import torch
import tensorflow


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


def test_tensorflow(_input):
    input = tensorflow.constant(_input)
    output = tensorflow.raw_ops.IFFT(
        input=input
    )
    return output


def assert_equals(_a, _b):
    a = np.array(_a)
    b = np.array(_b)
    if a.shape != b.shape:
        return False
    #TEST.log("comped")
    cmp = TEST.array_same(a, b, ifcomplex=True)
    if not cmp:
        TEST.logHead()
        TEST.log(f"a={a} b={b}")

    return cmp



@settings(max_examples=100, deadline=10000)
@given(_input=input_data())
def test_fft(_input):
    input = _input
    torch_output = test_torch(input)
    tensorflow_output = test_tensorflow(input)
    assertation = assert_equals(torch_output, tensorflow_output)
    assert assertation


torch.set_default_dtype(torch.float64)

if __name__ == "__main__" and TEST.PLATFORM() == "windows":
    TEST.log("running on", TEST.PLATFORM())
    TEST.log("only could in float64(tensorflow)")
    test_fft()
    pass

if __name__ == "__main__" and TEST.PLATFORM() == "linux":
    import atheris

    fuzz_target = atheris.instrument_func(test_fft.hypothesis.fuzz_one_input)
    atheris.Setup(sys.argv, fuzz_target)
    atheris.Fuzz()

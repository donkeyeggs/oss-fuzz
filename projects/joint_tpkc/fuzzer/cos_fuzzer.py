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

cout=LOG.Log("cos_fuzzer",log_dir=TEST._INIT_DIR)

def test_torch(_input):
    input = torch.tensor(_input)
    output = torch.cos(input)
    return output


def test_tensorflow(_input):
    input = tensorflow.constant(_input)
    output = tensorflow.math.cos(input)
    return output

def test_paddle(_input):
    input = paddle.to_tensor(_input)
    output = paddle.acos(input)
    return output

def assert_equals(_a, _b, _c):
    a = numpy.array(_a)
    b = numpy.array(_b)
    c = numpy.array(_c)
    # TEST.log("assert_equal ",numpy.shape(a)!=numpy.shape(b))
    ret = TEST.all_same([a, b,c])
    if not ret:
        cout.logHead()
        cout.log(f"(a)={a.shape} (b)={b.shape} (c)={c.shape}")
        cout.log(f"\na={a} \nb={b} \n c={c}")
        cout.log_empty()
        cout.logEnd()
        pass
    return ret


@settings(max_examples=10, deadline=10000)
@given(_input=TEST.ARRAY_ND())
# @example(_input=[1.0000002153053333e-39])
def test_cos(_input):
    torch_output = test_torch(_input)
    tensorflow_output = test_tensorflow(_input)
    paddle_output = test_paddle(_input)
    assertation = assert_equals(torch_output, tensorflow_output, paddle_output)
    assert assertation


TEST.log("running on", TEST.PLATFORM())
if __name__ == "__main__" and TEST.PLATFORM() == "windows":
    test_cos()
    pass

if __name__ == "__main__" and TEST.PLATFORM() == "linux":
    import atheris

    fuzz_target = atheris.instrument_func(test_cos.hypothesis.fuzz_one_input)
    atheris.Setup(sys.argv, fuzz_target)
    atheris.Fuzz()

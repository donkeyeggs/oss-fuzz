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

cout = LOG.Log("acos_fuzzer", log_dir=TEST._INIT_DIR)


def test_torch(_input):
    input = torch.tensor(_input)
    output = torch.acos(input)
    return output


def test_tensorflow(_input):
    input = tensorflow.constant(_input)
    output = tensorflow.math.acos(input)
    return output


def test_paddle(_input):
    input = paddle.to_tensor(_input)
    output = paddle.acos(input)
    return output


def assert_equals(_a, _b, _c):
    a = numpy.array(_a)
    b = numpy.array(_b)
    c = numpy.array(_c)
    return TEST.all_same([a, b, c])


@settings(max_examples=1000, deadline=10000)
@given(_input=TEST.ARRAY_ND(elements=TEST.FLOATS(min_float=-1, max_float=1, allow_inf=False)))
def test_acos(_input):
    if 1 and "Gather DATA":
        import gather as GATHER
        item = GATHER.Gather_Data(cout.PROJECT_NAME, [_input], test_torch, test_tensorflow, test_paddle)
        print(item)
        assert True
        return
    torch_output = test_torch(_input)
    tensorflow_output = test_tensorflow(_input)
    paddle_output = test_paddle(_input)
    assertation = assert_equals(torch_output, tensorflow_output, paddle_output)
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

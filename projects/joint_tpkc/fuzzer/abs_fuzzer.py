import sys
import numpy
import traceback
import hypothesis.strategies as st
from hypothesis import given, settings, assume, example

import base as TEST
import log as LOG

import torch
import tensorflow
import paddle

cout = LOG.Log("abs_fuzzer", log_dir=TEST._INIT_DIR)


def test_torch(_input):
    input = torch.tensor(_input)
    output = torch.abs(input)
    return output


def test_tensorflow(_input):
    input = tensorflow.constant(_input)
    output = tensorflow.math.abs(input)
    return output


def test_paddle(_input):
    input = paddle.to_tensor(_input)
    output = paddle.abs(input)
    return output


def assert_equals(_a, _b, _c):
    a = numpy.array(_a)
    b = numpy.array(_b)
    c = numpy.array(_c)
    cmp = TEST.all_same([a, b, c])
    if not cmp:
        cout.logHead()
        cout.log(f"a={a} b={b} c={c}")
        cout.log_empty()
        cout.logEnd()
    return cmp


# (-inf,-1e-37]U[1e-37,inf)
# input_float = st.one_of(
#    TEST.FLOATS(max_float=TEST.toFloat(-1E-37)),
#    TEST.FLOATS(min_float=TEST.toFloat(1E-37)),
# )
input_float = TEST.FLOATS()


@settings(max_examples=1000, deadline=10000)
@given(_input=TEST.ARRAY_ND(elements=input_float))
def _test_abs(_input):
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
    _test_abs()
    pass

if __name__ == "__main__" and TEST.PLATFORM() == "linux":
    import atheris

    fuzz_target = atheris.instrument_func(_test_abs.hypothesis.fuzz_one_input)
    atheris.Setup(sys.argv, fuzz_target)
    atheris.Fuzz()

import sys

import tensor_base as TEST
import hypothesis.strategies as st
from hypothesis import given,settings
import torch

@given(_input,_weight)
def test_conv1d(_input,_weight):
    pass

if __name__ == "__main__" and TEST.PLATFORM == 'linux':
    import atheris
    fuzz_target = atheris.instrument_func(test_conv1d.hypothesis.fuzz_one_input)
    atheris.Setup(sys.argv, fuzz_target)
    atheris.Fuzz()
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

cout = LOG.Log("conv1d_fuzzer",log_dir=TEST._INIT_DIR)
def test_torch(_input, _weight, _stride, _padding, _dilation):
    input = torch.tensor(_input)
    weight = torch.tensor(_weight)
    stride = _stride
    padding = _padding
    dilation = _dilation
    output = torch.nn.functional.conv1d(
        input=input,
        weight=weight,
        stride=stride,
        padding=padding,
        dilation=dilation
    )
    return output


def test_tensorflow(_input, _filters, _stride, _padding, _dilations):
    input = tensorflow.constant(_input)
    filters = numpy.transpose(tensorflow.constant(_filters), [2, 1, 0])
    stride = _stride
    padding = _padding
    dilations = _dilations
    output = tensorflow.nn.conv1d(
        input=input,
        filters=filters,
        stride=stride,
        padding='VALID' if padding == 0 else 'SAME',
        dilations=dilations,
        data_format="NCW"
    )
    return output


def test_paddle(_input, _filters, _stride, _padding, _dilations):
    input = paddle.to_tensor(_input)
    filters = paddle.to_tensor(_filters)
    stride = _stride
    padding = _padding
    dilations = _dilations
    output = paddle.nn.functional.conv1d(
        x = input,
        weight = filters,
        stride = stride,
        padding = 'VALID' if padding == 0 else 'SAME',
        dilation = dilations,
        data_format="NCL"
    )
    return  output

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


@st.composite
def input_data(draw):
    inf = TEST.toFloat(1e38)

    batch = draw(st.integers(min_value=2, max_value=20))
    in_channels = draw(st.integers(min_value=2, max_value=20))
    out_channels = draw(st.integers(min_value=2, max_value=in_channels))

    H_input_len = draw(st.integers(min_value=2, max_value=100))
    H_weight_len = draw(st.integers(min_value=2, max_value=H_input_len))

    dilation = draw(
        st.integers(
            min_value=1,
            max_value=int((H_input_len - H_weight_len) // (H_weight_len - 1) + 1)
        )
    )
    input_shape = (batch, in_channels, H_input_len)
    input = draw(
        TEST.ARRAY_ND(shape=input_shape, elements=TEST.FLOATS())
    )

    weight_shape = (out_channels, in_channels, H_weight_len)
    weight = draw(
        TEST.ARRAY_ND(shape=weight_shape, elements=TEST.FLOATS())
    )

    stride = draw(
        st.one_of(
            st.integers(min_value=1, max_value=20),
            st.tuples(
                st.integers(min_value=1, max_value=20)
            )
        )
    )

    # padding = draw(
    #    st.one_of(
    #        st.just('valid'),
    #        st.just('same')
    #    )
    # )
    padding = 0
    return (input, weight, stride, padding, dilation)


@settings(max_examples=100, deadline=10000)
@given(
    _input=input_data()
)
def test_conv1d(_input):
    (input, weight, stride, padding, dilation) = _input
    # TEST.log("[input shape]", numpy.shape(input), numpy.shape(weight),
    #         f"stride = {stride} padding = {padding} dilation = {dilation}")
    torch_output = test_torch(input, weight, stride, padding, dilation)
    tensorflow_output = test_tensorflow(input, weight, stride, padding, dilation)
    paddle_output = test_paddle(input, weight, stride, padding, dilation)
    # TEST.log("torch_output := ",torch_output)
    # TEST.log("tensor_output := ",tensorflow_output)
    # TEST.log("[output shape]", torch_output.shape, tensorflow_output.shape)
    assertation = assert_equals(torch_output, tensorflow_output, paddle_output)
    # TEST.log("torch output =",torch_output)
    # TEST.log("tensorflow output =",tensorflow_output)
    # TEST.log("assertation ",assertation)
    assert assertation


TEST.log("running on", TEST.PLATFORM())
if __name__ == "__main__" and TEST.PLATFORM() == "windows":
    test_conv1d()
    pass

if __name__ == "__main__" and TEST.PLATFORM() == "linux":
    import atheris

    fuzz_target = atheris.instrument_func(test_conv1d.hypothesis.fuzz_one_input)
    atheris.Setup(sys.argv, fuzz_target)
    atheris.Fuzz()
'''
Falsifying
example: test_conv1d(
    _input=(array([[[0.0000000e+00, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37],
                    [6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37],
                    [6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37],
                    [6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37],
                    [6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37],
                    [6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37]],

                   [[6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37],
                    [6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37],
                    [6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37],
                    [6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37],
                    [6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37],
                    [6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37, 6.8056474e+37, 6.8056474e+37,
                     6.8056474e+37, 6.8056474e+37]]], dtype=float32),
            array([[[0., -5., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]],

                   [[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                    [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                     1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]]],
                  dtype=float32),
            1,
            0,
            1),
)
with result:
DEBUG >>> (pytorch)=(2, 2, 1) (tensorflow)=(2, 2, 1)
DEBUG >>> 
(pytorch)=[[[inf]
  [inf]]

 [[inf]
  [inf]]] 
(tensorflow)=[[[-inf]
  [ inf]]

 [[-inf]
  [ inf]]]
'''

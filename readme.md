# 测试任务

## 本地测试环境

|项|值|
|----|----|
|引擎|libfuzzer|
|环境|win11/docker|
|C/G|CPU(4)|
|次数|50000次|

## 默认值域

|项|别名|约定|
|----|:----:|----|
|整形|INT|64位有符号整数|
|浮点数|FLOAT|32位浮点数|
|张量|TENSOR|1维张量，单元为FLOAT或者INT|
|张量大小|TENSORSHAPE|1~5维，每纬最长为10|
|多维张量|MULITTENSOR||

# 默认依赖

> 测试正确性依赖以下API

|API|
|----|
|numpy|
|atheris|
|hypthesis|

## 测试任务

| pytorch API                    | tensorflow api         | paddle api                      | 基础测试 | float64模式 | 多维张量 | 测试备注                | 测试依赖 |
|--------------------------------|------------------------|---------------------------------|:----:|:---------:|:---:|---------------------|------|
| torch.abs                      | tensor.math.abs        | paddle.abs                      |  v   |     v     |  v  | 输入中不包含inf与nan       ||
| torch.acos                     | tensor.math.acos       | paddle.acos                     |  v   |     v     |  v  | 输入张量非空              ||
| torch.cos                      | tensor.math.cos        | paddle.cos                      |  v   |     v     |  v  | 与torch.acos使用相同测试单元 ||
| torch.nn.functional.conv1d     | tensor.nn.conv1d       | paddle.nn.functional.conv1d     |  v   |     v     |     |                     |
| torch.nn.functional.conv2d     | tensor.nn.conv2d       | paddle.nn.functional.conv2d     |  v   |     v     |     |
| torch.nn.functional.conv3d     | tensor.nn.conv3d       | paddle.nn.functional.conv3d     |  v   |     v     |     |
| torch.fft.fft2d                | tensor.raw_ops.FFT2D   | paddle.fft.fft2                 |  v   |     v     |     |
| torch.fft.fft                  | tensor.raw_ops.FFT     | paddle.fft.fft                  |  v   |     v     |
| torch.fft.ifft2d               | tensor.raw_ops.IFFT2D  | paddle.fft.ifft2                |  v   |     v     |
| torch.fft.ifft                 | tensor.raw_ops.ifft    | paddle.fft.ifft                 |  v   |     v     |
| torch.nn.functional.max_pool1d | tensor.nn.max_pool1d   | paddle.nn.functional.max_pool1d |  v   |     v     |     |
| torch.nn.functional.max_pool2d | tensor.nn.max_pool2d   | paddle.nn.functional.max_pool2d |  v   |     v     |     |
| torch.nn.functional.max_pool3d | tensor.nn.max_pool3d   | paddle.nn.functional.max_pool3d |  v   |     v     |     |
| torch.nn.functional.relu       | tensor.nn.relu         | paddle.nn.functional.relu       |  v   |     v     |     |
| torch.nn.functional.softmax    | tensor.nn.softmax      | paddle.nn.functional.softmax    |  v   |     v     |     |

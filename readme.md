# pytorch测试任务

## 本地测试环境
|项|值|
|----|----|
|引擎|libfuzzer|
|环境|win11/ubuntu20|
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
|torch.tensor|


## 测试任务
|API|基础测试|float64模式|多维张量|测试备注|测试依赖|
|----|:----:|:----:|:----:|----|----|
|torch\.abs| :white_check_mark: |  ||输入中不包含inf与nan||
|torch\.acos| :white_check_mark: | | |输入张量非空||
|torch\.cos| :white_check_mark: | | | 与torch.acos使用相同测试单元||
|torch.kthvalue|:white_check_mark:||:white_check_mark:|nan被从float中移除||


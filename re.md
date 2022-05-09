# 本周进展汇报

CAFFE和KERAS的测试簇构建完毕

本周主要尝试构建的任务主要是tensorflow和pytorch之间的平行对比测试,将相同的测试数据投喂给不同的api观察结果

oss-fuzz没有提供平行测试的工具，我自己采用的是随机生成种子再分别进行离线测试的方法，每组api只测试了5个种子分别32768组数据，以下是本周测试的项目；

|`Pytorch`|`Tensorflow`|
|:----:|:----:|
| | |
|torch.abs|tf.math.abs|
|torch.acos|tf.math.acos|
|torch.cos|tf.math.cos|
|torch.nn.functional.conv1d|tf.nn.conv1d|
|torch.nn.functional.conv2d|tf.nn.conv2d|
|torch.nn.functional.conv3d|tf.nn.conv3d|
|torch.nn.functional.max_pool2d|torch.nn.max_pool2d|
|torch.nn.functional.relu|torch.nn.relu|


在单独测试中torch.kthvalue报告了一个关于nan的问题，在确认这个问题是错误还是手册规定


# 下周任务
进行tensorflow,pytorch,keras,keras的对比测试

完善项目，以志愿者将项目发布到google；

尝试写论文开头部分初稿

$$out(N_i, C_{out_j})=bias(C_{out_j})+\sum^{C_{in}-1}_{k=0}weight(C{out_j},k)\bigotimes input(N_i,k)$$
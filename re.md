# 本周进展汇报

重构了一下代码的结构，现在代码在pytorch和tensorflow方面同时使用同一组数据进行对比测试，能够迅速构建许多测试项目；
caffe和keras仍然使用单独测试；
目前发现的问题主要是边角问题，主要内容都是精度相关或者api提醒缺失的问题；
此外在对比测试的过程中发现各个库的参数设计较为不一致，给对比测试带来了一定的麻烦；

目前决定在论文中将对运行速度、参数设计、库细节进行对比；
尝试写了开头部分的论文，但是思路比较混乱;

搁置了发布到google的任务，因为不太确定对比其它google上的整合项目完成到了怎样的程度；

# 接下来的任务

下周尝试修改，将keras和caffe的高级api部分进行对比测试；
下周完成论文细纲，画一些需要的图表；
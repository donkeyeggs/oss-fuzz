from pickle import FALSE
import numpy as np
import torch
import tensor_base as TEST
def conv(input,weight,bias,stride=1,padding=0,dilation=1,groups=1):
    Ni,Cin,L = input.shape
    Nw,Cout,Lout = weight.shape

    pass
def case1():

    filters = torch.autograd.Variable(torch.randn(2,2,2))
    inputs = torch.autograd.Variable(torch.randn(2,2,2))
    print(inputs)
    print(filters)
    print(conv(inputs,filters))
    print(torch.nn.functional.conv1d(inputs, filters))

case1()
from pickle import FALSE
import numpy as np
import torch
import tensor_base as TEST
def case1():
    shape = (3,3,3)
    dim = 0
    k = 1
    a = np.random.random(size=shape)
    b = torch.tensor(a)
    c = torch.kthvalue(b,k=1,dim=0)
    print(b)
    print(c)
    #print(TEST.trans(b))
    def check_indices(input,dim,selects,indices):
        indice = TEST.trans(indices)
        for index,item in indice:
            rindex = index[:dim]+[int(item)]+index[dim:]
            #print("rindex=",rindex)
            #print("index=",index)
            a = TEST.get(input,rindex)
            b = TEST.get(selects,index)
            #print("a=",a,"b=",b)
            if TEST.equals(a,b):
                return False
        return True
    def check_values(input,k,dim,selects):
        lshape = len(input.shape)
        ldim = input.shape[dim]

        for index,item in TEST.trans(selects):
            count_min = 0
            count_equal = 0
            for i in range(ldim):
                rindex = TEST.index_join(index,dim,i)
                value = TEST.get(input,rindex)
                if TEST.equals(value,item):
                    count_equal+=1
                elif value<item:
                    count_min+=1
            #print("index=",index,"item=",item)
            #print(f"{1+count_min}<={k}<{1+count_min+count_equal}")
            if not ((1+count_min)<=(k)<(1+count_min+count_equal)):
                return False
        return True
    
    def check_kthvalue(input,k,dim,selects,indices):
        assert check_indices(input,dim,selects,indices)
        assert check_values(input,k,dim,selects)
        return True
    assert check_kthvalue(b,k,dim,c[0],c[1])

case1()
import os
import sys
import time

import numpy
import pandas

origin_columns = ["Id", "Project", "Framework", "DGV" ,"RT", "dataSize"]
origin_columns_set = set(origin_columns)

def count_size(data):
    return sum([numpy.array(item).nbytes() for item in data])

def DGV(_X,_Y):
    X = numpy.array(_X)
    Y = numpy.array(_Y)
    spec = numpy.isnan(X) | numpy.isnan(Y) | numpy.isinf(X) | numpy.isinf(Y)
    X = numpy.where(spec,1,X)
    Y = numpy.where(spec,1,Y)
    return numpy.average((X-Y)/Y)

def Gather_Data(Project,input,test_torch,test_tensorflow,test_paddle):
    ret = {"Project":Project}
    st = time.time_ns()
    print(input)
    torch_output = test_torch(*input)
    p1 = time.time_ns()
    tensorflow_output = test_tensorflow(*input)
    p2 = time.time_ns()
    paddle_output = test_paddle(*input)
    p3 = time.time_ns()

    ret["PyTorch RT"] = p1-st
    ret["TensorFlow RT"] = p2-p1
    ret["PaddlePaddle RT"] = p3-p2
    ret["dataSize"] = count_size(input)
    outputs = {"PyTorch":torch_output,"TensorFlow":tensorflow_output,"PaddlePaddle":paddle_output}
    for namea in outputs:
        for nameb in outputs:
            if namea == nameb :
                continue
            name = namea+"&"+nameb+"& DGV"
            ret[name] = DGV(outputs[namea],outputs[nameb])
    return ret

def Gather(Objetct):
    def __init__(self,loads=True):
        self.data = pandas.DataFrame(columns=origin_columns)
        self.Id = 0
        if loads:
            self.load()

    def load(self):
        self.data = pandas.read_json("gather_data.json")
        self.Id = max(self.data["Id"])

    def save(self):
        self.data.to_json("gather_data.json", force_ascii=False)

    def put(self, item, saves=True):
        self.Id += 1
        item["Id"] = self.Id
        for name in item:
            if name not in origin_columns_set:
                raise RuntimeError(f"No such data column like [{name}]")
        self.append(item)
        if saves:
            self.save()

    def clean(self):
        self.data = pandas.DataFrame(columns=origin_columns)
        self.Id = 0
        self.save()

def clean():
    it = Gather(loads=False)
    it.save()

#def run():
#    now_dir = r"D:\learn\code\oss-fuzz\projects\joint_tpkc\fuzzer"
#    for i in range(8):
#        print(f"清理中...{8-i}")
#        time.sleep(1)
#    clean()
#    for filename in os.listdir(now_dir):
#        if not filename.endswith("_fuzzer.py"):
#            continue
#        filedir = os.path.join(now_dir,filename)
#        os.system(fr"python filedir --gather True --runTimes 10000 --deadline 10000")

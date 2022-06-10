import json
import os
import sys
import time

import numpy
import pandas

origin_columns = ["Id", "Project", "Framework", "DGV", "RT", "dataSize"]
origin_columns_set = set(origin_columns)


def count_size(data):
    return sum([numpy.array(item).nbytes for item in data])


# def DGV(_X,_Y):
#    X = numpy.array(_X)
#    Y = numpy.array(_Y)
#    spec = numpy.isnan(X) | numpy.isnan(Y) | numpy.isinf(X) | numpy.isinf(Y)
#    X = numpy.where(spec,1,X)
#    Y = numpy.where(spec,1,Y)
#    shape = numpy.product(Y.shape)
#    #print("sum",numpy.sum((X - Y) ** 2))
#    return numpy.sqrt(numpy.sum((X-Y)**2))/shape

vc = 0


def Gather_Data(Project, input, test_torch, test_tensorflow, test_paddle):
    global vc
    vc += 1
    # print(f"{Project} case : {vc}")
    ret = {"Project": Project}
    st = time.perf_counter_ns()
    torch_output = test_torch(*input)
    p1 = time.perf_counter_ns()
    tensorflow_output = test_tensorflow(*input)
    p2 = time.perf_counter_ns()
    paddle_output = test_paddle(*input)
    p3 = time.perf_counter_ns()

    ret["PyTorch RT"] = p1 - st
    ret["TensorFlow RT"] = p2 - p1
    ret["PaddlePaddle RT"] = p3 - p2
    ret["dataSize"] = count_size(input)
    # outputs = {"PyTorch":torch_output,"TensorFlow":tensorflow_output,"PaddlePaddle":paddle_output}
    # for namea in outputs:
    #    for nameb in outputs:
    #        if namea == nameb :
    #            continue
    #        name = namea+"&"+nameb+"& DGV"
    #        ret[name] = DGV(outputs[namea],outputs[nameb])
    return ret


class Gather:
    def __init__(self, loads=True):
        self.data = pandas.DataFrame(columns=origin_columns)
        self.Id = 0
        if loads:
            self.load()

    def load(self):
        self.data = pandas.read_json("gather_data.json")
        if "Id" not in self.data:
            self.Id = 0
        else:
            self.Id = max(self.data["Id"])

    def save(self):
        self.data.to_json("gather_data.json", force_ascii=False)

    def put(self, item, saves=True):
        self.Id += 1
        item["Id"] = self.Id
        # for name in item:
        #    if name not in origin_columns_set:
        #        raise RuntimeError(f"No such data column like [{name}]")
        self.data.append(item, ignore_index=True)
        if saves:
            self.save()

    def clean(self):
        self.data = pandas.DataFrame(columns=origin_columns)
        self.Id = 0
        self.save()


def clean():
    it = Gather(loads=False)
    it.save()


def run():
    now_dir = r"D:\learn\code\oss-fuzz\projects\joint_tpkc\fuzzer"
    for i in range(5):
        print(f"清理中...{5 - i}")
        time.sleep(1)
    data = []
    for filename in sorted(os.listdir(now_dir)):
        if not filename.endswith("_fuzzer.py"):
            continue
        filedir = os.path.join(now_dir, filename)
        print(f"\n>>>running {filename}")
        os.system(fr"python {filedir} > out.txt")
        with open("out.txt", "r") as fin:
            for line in fin.readlines():
                # print(eval(line.strip()), data)
                try:
                    data.append(eval(line.strip()))
                    # print(data)
                except Exception as e:
                    continue
        with open("collect.txt", "w") as fout:
            print(data, file=fout)


if __name__ == "__main__":
    run()

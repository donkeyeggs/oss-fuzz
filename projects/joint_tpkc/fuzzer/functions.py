import datetime
import platform
import random
import time
from itertools import product


def PLATFORM():
    return platform.system().lower()


def self(x):
    return x


def shape_space(shape):
    return product(*[range(i) for i in shape])


def index_except(index, dim):
    return index[:dim] + index[dim + 1:]


def index_join(index, dim, i):
    return index[:dim] + [int(i)] + index[dim:]


def get(array, index):
    v = array
    for i in index:
        v = v[i]
    return v


def trans(tensor, cond=self):
    ret = []
    for index in shape_space(tensor.shape):
        item = get(tensor, index)
        ret.append((list(index), cond(item)))
    return ret


def new_name(lens=6):
    random.seed(time.time())
    return ''.join(random.choices("0123456789", k=lens))


def now_time(formats="%Y%m%d-%H.%M.%S"):
    return datetime.datetime.now().strftime(formats)
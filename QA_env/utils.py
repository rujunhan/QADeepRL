### All functions directly adapted from https://github.com/allenai/bi-att-flow/

from itertools import zip_longest
from functools import reduce
from operator import mul
import numpy as np
import json
from collections import deque
from tqdm import tqdm

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER


def grouper(iterable, n, fillvalue=None, shorten=False, num_groups=None):
    args = [iter(iterable)] * n
    out = zip_longest(*args, fillvalue=fillvalue)
    out = list(out)
    if num_groups is not None:
        default = (fillvalue, ) * n
        assert isinstance(num_groups, int)
        out = list(each for each, _ in zip_longest(out, range(num_groups), fillvalue=default))
    if shorten:
        assert fillvalue is None
        out = (tuple(e for e in each if e is not None) for each in out)
    return out

def mytqdm(list_, desc="", show=True):
    if show:
        pbar = tqdm(list_)
        pbar.set_description(desc)
        return pbar
    return list_


def json_pretty_dump(obj, fh):
    return json.dump(obj, fh, sort_keys=True, indent=2, separators=(',', ': '))


def index(l, i):
    return index(l[i[0]], i[1:]) if len(i) > 1 else l[i[0]]


def fill(l, shape, dtype=None):
    out = np.zeros(shape, dtype=dtype)
    stack = deque()
    stack.appendleft(((), l))
    while len(stack) > 0:
        indices, cur = stack.pop()
        if len(indices) < shape:
            for i, sub in enumerate(cur):
                stack.appendleft([indices + (i,), sub])
        else:
            out[indices] = cur
    return out


def short_floats(o, precision):
    class ShortFloat(float):
        def __repr__(self):
            return '%.{}g'.format(precision) % self

    def _short_floats(obj):
        if isinstance(obj, float):
            return ShortFloat(obj)
        elif isinstance(obj, dict):
            return dict((k, _short_floats(v)) for k, v in obj.items())
        elif isinstance(obj, (list, tuple)):
            return tuple(map(_short_floats, obj))
        return obj

    return _short_floats(o)


def argmax(x):
    return np.unravel_index(x.argmax(), x.shape)



import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
import math
from collections import OrderedDict
import time
import matplotlib.pyplot as plt
import argparse
import random
from args import parse_args
from datastream import load_data
#from utils import *


def main():

    args = parse_args()
    batch = load_data(args)    

    count = 0
    for item in batch:
        print(item)
        count += 1
        if count > 10:
            break


if __name__ == '__main__':
    main()

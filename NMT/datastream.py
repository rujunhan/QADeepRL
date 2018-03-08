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
import random


class load_data():
    def __init__(self, args):

        self.batch_size = args.batch_size
        #self.data_size = args.data_size

        #self.vocab = np.load(args.source+args.vocab).item()

        fname = args.source+args.data_file
        print(fname)
        self.source = open(fname, 'r')

        self.end_of_data = False

        self.count = 0


    def reset(self):
        self.source.seek(0)
        self.count = 0
        raise StopIteration


    def __iter__(self):
        return(self)

    def __next__(self):

        data = []
        count = 0

        while True:
            line = self.source.readline().strip().split(',')
            source = line[0]
            target = line[1]

            if line == '':
                print("end of file!")
                self.reset()
                break

            self.count += 1

            #label 
            #if len(line) < 2:
            #    continue
            count += 1
            #text_list = text.split(" ")

            #elif label in self.labels:
            #    count += 1

            #for i in range(self.skips, len(text_list) - self.skips):

            #    out_text = text_list[i-self.skips:i] + text_list[i+1:i+self.skips+1]

            #    in_text = text_list[i]

            #    data.append((label, in_text, out_text))
            data.append(line)
            if count >= self.batch_size:
                break
            
        #in_idxs, out_idxs, covars = self.create_batch(data, self.vocab)
        
        #return in_idxs, out_idxs, covars

        return data

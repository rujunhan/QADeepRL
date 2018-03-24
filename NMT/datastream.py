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
        self.max_sent_len = args.max_sent_len

        #self.vocab = np.load(args.source+args.vocab).item()                                                                                     
        fname = args.source+args.data_file

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

        count = 0

        batch_s = []
        batch_t = []

        while True:
            line = self.source.readline().strip().split(',')
            source = line[0].split(' ')
            target = line[1].split(' ')

            if line == '':
                print("end of file!")
                self.reset()
                break

            self.count += 1

            count += 1
                       batch_s.append(source)
            batch_t.append(target)

            if count >= self.batch_size:
                break

        batch_s = self.mask_sent(batch_s)
        batch_t = self.mask_sent(batch_t)

        return Variable(torch.LongTensor(batch_s), requires_grad=False), Variable(torch.LongTensor(batch_t), requires_grad=False)


    def mask_sent(self, raw_batch):

        max_sent_len = max([len(x) for x in raw_batch])

        batch = []
        for sent in raw_batch:

            sent = [int(x) for x in sent]

            if len(sent) >= max_sent_len:
                sent = sent[:max_sent_len]
            else:
                sent.extend([int(x) for x in np.zeros(max_sent_len - len(sent))])

            batch.append(sent)

        return np.stack(batch, axis=0)
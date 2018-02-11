import numpy as np
import logging
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
from glob import glob
import math
from collections import OrderedDict
from layers import *
import time
import random
import itertools


class BiDAF(nn.Module):
    def __init__(self, config, rep=True):

        super(BiDAF, self).__init__()

        self.config = config

        ## need to figure this out later
        self.global_step = 0
        
        self.N = config.batch_size
        self.M = config.max_num_sents
        self.JX = config.max_sent_size
        self.JQ = config.max_ques_size
        self.VW = config.word_vocab_size
        self.VC = config.char_vocab_size
        self.W = config.max_word_size
        
        # if M > 1, input_size = M * JX * emb, to be fixed later
        input_size = config.word_emb_size

        if config.use_char_emb:
            input_size += config.out_channel_dims

        self.HighwayMLP = HighwayMLP(input_size, gate_bias=config.wd)

        # 2d character-CNN layer
        self.char_cnn = nn.Conv2d(config.char_emb_size, config.out_channel_dims, (1, config.filter_heights))
        if not config.share_cnn_weights:
            self.char_cnn_q = nn.Conv2d(config.char_emb_size, config.out_channel_dims, (1, config.filter_heights))

        # batch_size * max_num_sents * max_sent_size                                                                       
        self.x = Variable(torch.LongTensor(self.N, self.M, self.JX).zero_())
        self.cx = Variable(torch.LongTensor(self.N, self.M, self.JX, self.W).zero_())
        self.x_mask = Variable(torch.IntTensor(self.N, self.M, self.JX).zero_())
        self.q = Variable(torch.LongTensor(self.N, self.JQ).zero_())
        self.cq = Variable(torch.LongTensor(self.N, self.JQ, self.W).zero_())
        self.q_mask = Variable(torch.IntTensor(self.N, self.JQ).zero_())
        
        self.y = Variable(torch.IntTensor(self.N, self.M, self.JX))
        self.y2 = Variable(torch.IntTensor(self.N, self.M, self.JX))
        #self.is_train = tf.placeholder('bool', [], name='is_train')
        #self.new_emb_mat = tf.placeholder('float', [None, config.word_emb_size], name='new_emb_mat')

        ## Word embedding layer 
        if config.use_word_emb:
            self.emb = nn.Embedding(config.word_vocab_size, config.word_emb_size, sparse=True)
            self.emb.weight = Parameter(torch.FloatTensor(config.emb_mat))
            self.emb.weight.requires_grad = config.finetune

        ## Character embedding layer
        if config.use_char_emb:
            self.char_emb = nn.Embedding(config.char_vocab_size, config.char_emb_size, sparse=True)
            self.char_emb.weight.requires_grad = True

        ## Context layers
        ## hidden size / num_layers can be tunable, but the original paper doesn't use it
        self.context = nn.LSTM(input_size,
                               input_size,               # hidden size
                               1,                        # num_context_layers,
                               bias = False,
                               batch_first = True,
                               dropout = config.input_keep_prob,
                               bidirectional = True)

        if not config.share_lstm_weights:
            self.context_q = nn.LSTM(input_size,
                                     input_size,         # hidden size
                                     1,                  # num_context_layers,                                              
                                     bias = False,
                                     batch_first = True,
                                     dropout = config.input_keep_prob,
                                     bidirectional = True)

    def forward(self, batches):
        
        #self.x = self.get_data_feed(batches)
        #print(self.x)
        for i in batches:
            
            # get data
            self.get_data_feed(i[1], self.config)

            dc, dw, dco = self.config.char_emb_size, self.config.word_emb_size, self.config.char_out_size
            
            # word embedding layer
            if self.config.use_word_emb:
            
                # not sure how the author treat num sentence differently                                    
                # since he consider the entire paragraph as a long sent                                                     
                # needs to figure out since dim=1 - M is num_sent                                                            
                Ax = self.emb(self.x.view(self.N, -1)).view(self.N, self.JX, -1)
                Aq = self.emb(self.q)

            # character embedding layer                                                                                      
            if self.config.use_char_emb:

                Acx = self.char_emb(self.cx.view(self.N, -1)).view(self.N, self.JX, self.W, -1)
                Acq = self.char_emb(self.cq.view(self.N, -1)).view(self.N, self.JQ, self.W, -1)

                # Add dropout layer here + fix dimensions                                                                     
                # character-cnn + maxpool                                                                                     
                xx = self.char_cnn(Acx.permute(0, 3, 1, 2)).max(dim=3)[0].permute(0, 2, 1)
                
                # not sure how the author treat num sentence differently
                # since he consider the entire paragraph as a long sent
                # needs to figure out since dim=1 - M is num_sent
                #xx = xx.unsqueeze(1)

                if self.config.share_cnn_weights:
                    qq = self.char_cnn(Acq.permute(0, 3, 1, 2)).max(dim=3)[0].permute(0, 2, 1)
                else:
                    qq = self.char_cnn_q(Acq.permute(0, 3, 1, 2)).max(dim=3)[0].permute(0, 2, 1)
            
                xx = torch.cat((Ax, xx), -1)
                qq = torch.cat((Aq, qq), -1)

            else:
                xx = Ax
                qq = Aq

            if self.highway:
                xx = self.highway(xx, self.config.highway_num_layers)
                qq = self.highway(qq, self.config.highway_num_layers)

            print('\nxx:', xx.size())
            print('qq:', qq.size())

            # context layer
            h, _ = self.context(xx)
            if self.config.share_lstm_weights:
                u, _ = self.context(qq)
            else:
                u, _ = self.context_q(qq)

            print("h:", h.size())
            print("u:", u.size())

        return

    def highway(self, cur, num_layers):

        for i in range(num_layers):
            pre = cur
            cur = self.HighwayMLP(pre)
        
        return cur

    def get_data_feed(self, batch, config, supervised = True):
        # for each batch of raw data the model gets
        # convert it to PyTorch tensors that are ready for training
        X = batch.data['x']
        CX = batch.data['cx']


        if supervised:
            y = np.zeros([self.N, self.M, self.JX])
            y2 = np.zeros([self.N, self.M, self.JX])
            
            for i, (xi, cxi, yi) in enumerate(zip(X, CX, batch.data['y'])):
                start_idx, stop_idx = random.choice(yi)
                j, k = start_idx
                j2, k2 = stop_idx
                if config.single:
                    X[i] = [xi[j]]
                    CX[i] = [cxi[j]]
                    j, j2 = 0, 0
                if config.squash:
                    offset = sum(map(len, xi[:j]))
                    j, k = 0, k + offset
                    offset = sum(map(len, xi[:j2]))
                    j2, k2 = 0, k2 + offset
                y[i, j, k] = 1
                y2[i, j2, k2-1] = 1
                self.y = y
                self.y2 = y2

        def _get_word(word):
            d = batch.shared['word2idx']
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in d:     
                    return d[each]

            #if config.use_glove_for_unk:
            #    d2 = batch.shared['new_word2idx']
            #    for each in (word, word.lower(), word.capitalize(), word.upper()):
            #        if each in d2:
            #            return d2[each] + len(d)
            return 1

        def _get_char(char):
            d = batch.shared['char2idx']
            if char in d:
                return d[char]
            return 1

        for i, xi in enumerate(X):
            if self.config.squash:
                xi = [list(itertools.chain(*xi))]
            for j, xij in enumerate(xi):
                if j == config.max_num_sents:
                    break
                for k, xijk in enumerate(xij):
                    if k == config.max_sent_size:
                        break
                    each = _get_word(xijk)
                    assert isinstance(each, int), each
                    self.x[i, j, k] = each
                    self.x_mask[i, j, k] = 1

        for i, cxi in enumerate(CX):
            if self.config.squash:
                cxi = [list(itertools.chain(*cxi))]
            for j, cxij in enumerate(cxi):
                if j == config.max_num_sents:
                    break
                for k, cxijk in enumerate(cxij):
                    if k == config.max_sent_size:
                        break
                    for l, cxijkl in enumerate(cxijk):
                        if l == config.max_word_size:
                            break
                        self.cx[i, j, k, l] = _get_char(cxijkl)

        for i, qi in enumerate(batch.data['q']):
            for j, qij in enumerate(qi):
                self.q[i, j] = _get_word(qij)
                self.q_mask[i, j] = 1


        for i, cqi in enumerate(batch.data['cq']):
            for j, cqij in enumerate(cqi):
                for k, cqijk in enumerate(cqij):
                    self.cq[i, j, k] = _get_char(cqijk)
                    if k + 1 == config.max_word_size:
                        break
        return

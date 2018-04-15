import numpy as np
from tqdm import tqdm
import argparse
import json
import math
import os
import shutil
from pprint import pprint
from read_data import read_data, get_squad_data_filter, update_config
from args import parse_args
from model import *
from evaluator import GPUF1Evaluator


def _config_debug(config):
    if config.debug:
        config.num_steps = 2
        config.eval_period = 1
        config.log_period = 1
        config.save_period = 1
        config.val_num_batches = 2
        config.test_num_batches = 2

def eval_model(model, trn_data, dev_data, config):

    model.eval()

    dev_num_steps = int(math.ceil(dev_data.num_examples / (config.batch_size * config.num_gpus)))
    trn_num_steps = int(math.ceil(trn_data.num_examples / (config.batch_size * config.num_gpus)))
    
    evaluator = GPUF1Evaluator(config, model)

    e_trn = evaluator.get_evaluation_from_batches(tqdm(trn_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=trn_num_steps, shuffle=True, cluster=config.cluster), total=trn_num_steps))
    e_dev = evaluator.get_evaluation_from_batches(tqdm(dev_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=dev_num_steps, shuffle=True, cluster=config.cluster), total=dev_num_steps))
    print(e_trn)
    print(e_dev)
    
def main():
    
    config = parse_args()
    data_filter = get_squad_data_filter(config)

    train_data = read_data(config, 'train', config.load, data_filter=data_filter)
    dev_data = read_data(config, 'dev', True, data_filter=data_filter)
    
    #print("Total vocabulary for training is %s" % config.word_vocab_size)
    #print(train_data.shared['x'][0][0])
    #print(train_data.shared['x'][0][1])

    #print(train_data.data['*x'][0])
    update_config(config, [train_data, dev_data])
    #_config_debug(config)
    
    print("Total vocabulary for training is %s" % config.word_vocab_size) 
    
    # from all
    word2vec_dict = train_data.shared['lower_word2vec'] if config.lower_word else train_data.shared['word2vec']
    # from filter-out set
    word2idx_dict = train_data.shared['word2idx']
    
    # filter-out set idx-vector
    idx2vec_dict = {word2idx_dict[word]: vec for word, vec in word2vec_dict.items() if word in word2idx_dict}
    print("{}/{} unique words have corresponding glove vectors.".format(len(idx2vec_dict), len(word2idx_dict)))
    
    # <null> and <unk> do not have corresponding vector so random.
    emb_mat = np.array([idx2vec_dict[idx] if idx in idx2vec_dict
                        else np.random.multivariate_normal(np.zeros(config.word_emb_size), np.eye(config.word_emb_size))
                        for idx in range(config.word_vocab_size)])
    
    config.emb_mat = emb_mat
    config.new_emb_mat = train_data.shared['new_emb_mat']
    
    print(emb_mat.shape, config.new_emb_mat.shape)

    ## Initialize model
    model = BiDAF(config)

    if config.use_gpu:
        model.cuda()

    #optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=config.init_lr)
    print("learning rate is: %.4f" % config.init_lr)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.init_lr)
    
    ## Begin training
    num_steps = config.num_steps or int(math.ceil(train_data.num_examples / (config.batch_size * config.num_gpus))) * config.num_epochs
    global_step = 0
    
    train_loss = []
    count = 0
    for batches in tqdm(train_data.get_multi_batches(config.batch_size, config.num_gpus,
                                                     num_steps=num_steps, shuffle=True, cluster=config.cluster), total=num_steps):
            
        model.train()
        model.zero_grad()
        
        model(batches)
        model.loss = model.build_loss()

        model.loss.backward()
        optimizer.step()
        
        if config.test_run:
            eval_model(model, train_data, dev_data, config)
            break
        else:
            if count % 500 == 0:
                #print("train loss is: %.4f" % model.loss.data.cpu().numpy()[0])
                eval_model(model, train_data, dev_data, config)
                #print("eval loss is: %.4f \n" % eval_loss)
        count += 1
    return 


    #count = 0
    #for idx, ds in train_data.get_batches(config.batch_size, num_batches=None, shuffle=False, cluster=False):
    #    if count > 0:
    #        break
        #for i in idx:
            #print(i)
            #print(ds.data)
    #    count += 1
    #return

if __name__ == "__main__":
    main()


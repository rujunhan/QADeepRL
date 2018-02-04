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

def main():
    config = parse_args()
    data_filter = get_squad_data_filter(config)
    train_data = read_data(config, 'train', config.load, data_filter=data_filter)
    count = 0
    for idx, ds in train_data.get_batches(config.batch_size, num_batches=None, shuffle=False, cluster=False):
        if count > 1:
            break
        for i in idx:
            print(i)
            print(ds.data)
        count += 1
    return


if __name__ == "__main__":
    main()
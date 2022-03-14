import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

import os
import math
import argparse
import random

from models import *

device = 'cuda'

class DatasetFromFile(Dataset):
    def __init__(self, filename):
        examples = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line == '':
                    continue
                line = [int(x) for x in line.split(',')]
                examples.append(line)
        x = torch.LongTensor(examples)
        self.x = x

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)


def init():
    global device
    global CUDA_CORE

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(2020)
    np.random.seed(2020)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset_path', default='', type=str)
    arg_parser.add_argument('--dataset', default='', type=str)
    arg_parser.add_argument('--device', default='cuda', type=str)
    arg_parser.add_argument('--cuda_core', default='0', type=str)
    arg_parser.add_argument('--model_path', default='', type=str)
    arg_parser.add_argument('--batch_size', default=32, type=int)

    args = arg_parser.parse_args()

    device = args.device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_core

    return args


def load_data(dataset_path, dataset,
            load_train=True, load_valid=True, load_test=True):
    dataset_path += '{}/'.format(dataset)
    train_path = dataset_path + '{}.train.data'.format(dataset)
    valid_path = dataset_path + '{}.valid.data'.format(dataset)
    test_path = dataset_path + '{}.test.data'.format(dataset)

    train, valid, test = None, None, None

    if load_train:
        train = DatasetFromFile(train_path)
    if load_valid:
        valid = DatasetFromFile(valid_path)
    if load_test:
        test = DatasetFromFile(test_path)

    return train, valid, test


def evaluate_model(model, test,
                batch_size, dataset_name):
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)

    model = model.to(device)
    model.eval()
    K = 10
    neg_sample_num = 100
    m = test.x.shape[1]

    acc = 0.0
    xs = test.x.tolist()
    ########
    basket0 = [356, 2571, 593, 260, 2959, 527, 1196, 110, 50, 1210]
    x0 = [0] * m
    for i in basket0:
        x0[i-1] = 1
    xs = [x0]
    ########
    for i in tqdm(range(len(xs))):
        x = xs[i]
        basket = [idx for idx, z in enumerate(x) if z == 1]
        neg_sample_cands = [idx for idx, z in enumerate(x) if z == 0]
        # p = random.sample(basket, 1)[0]
        # x[p] = 0

        # neg_samples = [p] + random.sample(neg_sample_cands, neg_sample_num)

        # x_candidates = []
        # for j in neg_samples:
        #     y = x[:]
        #     y[j] = 1
        #     x_candidates.append(y)
        x_candidates = []
        for j in range(0, m):
            y = x[:]
            y[j] = 1
            x_candidates.append(y)

        with torch.no_grad():
            y_batches = []
            for j in range(0, len(x_candidates), batch_size):
                x_batch = torch.LongTensor(x_candidates[j:j+batch_size]).to(device)
                y_batch = model(x_batch).tolist()
                y_batches.append(y_batch)

        y = [z for batch in y_batches for z in batch]
        y = [(j, z) for j, z in enumerate(y)]

        y = sorted(y, key = lambda z : z[1], reverse=True)
        ###        
        y = [(j+1, z) for j, z in y]
        print(y[:K])
        y = [(j, z) for j, z in y if j not in basket0]
        print(y[:K])
        ###
        # if 0 in [j for j, z in y[:K]]:
        #     acc += 1.0

    acc /= len(xs)

    print(f'{dataset_name} {acc}')


def main():
    random.seed(10)

    args = init()

    _, _, test = load_data(args.dataset_path, args.dataset)

    model = torch.load(args.model_path)

    evaluate_model(model, test,
        args.batch_size, args.dataset)

if __name__ == '__main__':
    main()
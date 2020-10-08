from __future__ import  absolute_import
import os

import gensim
import json
import ipdb
import matplotlib
import numpy
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, VRDDataset
from data.vrd_dataset import NoAnnotaion
from model import FasterRCNNVGG16
import torch as t
from torch import nn
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from model import VGG16PREDICATES
from utils.load_w2v import load_from_word2vec
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
from utils.encoding import DataParallelModel, DataParallelCriterion
import argparse

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

def train(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    t.distributed.init_process_group(backend="nccl", init_method="env://")
    t.cuda.set_device(args.local_rank)
    device = t.device("cuda", args.local_rank)
    opt._parse(kwargs)


    dataset = VRDDataset(opt)
    train_sampler = t.utils.data.distributed.DistributedSampler(dataset)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=False, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers,
                                  sampler=train_sampler)

    # word2vec_map = load_from_word2vec("test_word2vec.txt")
    word2vec_db = json.load(open("w2v.json"))

    faster_rcnn = FasterRCNNVGG16()
    faster_rcnn_trainer = FasterRCNNTrainer(faster_rcnn)
    faster_rcnn_trainer.load(opt.faster_rcnn_model)
    vrd_trainer = VGG16PREDICATES(faster_rcnn_trainer, word2vec_db, dataset.db.triplets).to(device)
    vrd_trainer = nn.parallel.DistributedDataParallel(vrd_trainer, find_unused_parameters=True, device_ids=[args.local_rank], output_device=args.local_rank)
    optimizer = t.optim.Adam(vrd_trainer.parameters())

    for epoch in range(opt.vrd_epoch):
        total_loss = 0
        for ii, (img, D) in tqdm(enumerate(dataloader)):
            if len(img) == 0:
                continue
            if D == [] or D[0] == []:
                continue

            img = img.cuda().float()

            loss = vrd_trainer(img, D)
            total_loss += sum(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(total_loss / (ii + 1))

if __name__ == '__main__':
    # import fire

    # fire.Fire()
    train()

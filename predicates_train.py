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

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

def train(**kwargs):
    opt._parse(kwargs)

    dataset = VRDDataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)

    word2vec_map = load_from_word2vec("test_word2vec.txt")
    word2vec_db = {"obj": [], "rel": []}
    for obj in dataset.db.label_names:
        if obj in word2vec_map:
            word2vec_db["obj"].append(word2vec_map[obj])
        else:
            objs = obj.split(" ")
            v = []
            for i in objs:
                v.append(word2vec_map[i])
            v = numpy.array(v)
            word2vec_db["obj"].append(numpy.average(v, axis=1))
    for rel in dataset.db.predicates_name:
        if rel in word2vec_map:
            word2vec_db["rel"].append(word2vec_map[rel])
        else:
            rels = rel.split(" ")
            v = []
            for i in rels:
                v.append(word2vec_map[i])
            v = numpy.array(v)
            word2vec_db["rel"].append(numpy.average(v, axis=1))

    faster_rcnn = FasterRCNNVGG16()
    faster_rcnn_trainer = FasterRCNNTrainer(faster_rcnn)
    faster_rcnn_trainer.load(opt.faster_rcnn_model)
    vrd_trainer = VGG16PREDICATES(faster_rcnn_trainer, word2vec_db, dataset.db.triplets).cuda()
    optimizer = t.optim.Adam(vrd_trainer.parameters())


    w2v = {
        "obj": [],
        "rel": []
    }
    for obj in dataset.db.label_names:
        w2v["obj"].append(word2vec_db.get_vector(obj))

    for rel in dataset.db.predicates_name:
        w2v["rel"].append(word2vec_db.get_vector(rel))

    for epoch in range(opt.vrd_epoch):
        for ii, (img, D) in tqdm(enumerate(dataloader)):
            if len(img) == 0:
                continue

            img, D = img.cuda().float(), D.cuda()

            total_loss = vrd_trainer(img, D)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


if __name__ == '__main__':
    import fire

    fire.Fire()

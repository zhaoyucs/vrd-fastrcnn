import os
import json

import numpy as np

from .util import read_image


class NoAnnotaion(Exception):
    pass


class VRDBboxDataset:
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir

        json_file = os.path.join(data_dir, "annotations_{0}.json".format(split))
        self.data_json = json.load(open(json_file))
        self.id_list = list(self.data_json.keys())

        self.label_names = json.load(open(os.path.join(data_dir, "objects.json")))
        self.predicates_name = json.load(open(os.path.join(data_dir, "predicates.json")))

        self.img_dir = os.path.join(data_dir, "sg_dataset/sg_{0}_images".format(split))

        #
        self.use_difficult = False
        self.return_difficult = False

    def __len__(self):
        return len(self.data_json)

    def get_example(self, i):
        anno = self.data_json[self.id_list[i]]

        if not anno:
            raise NoAnnotaion

        bbox = list()
        label = list()
        for pair in anno:
            # bbox=[ymin,xmin,ymax,xmax]
            _bb = pair["subject"]["bbox"]
            bbox.append([_bb[0], _bb[2], _bb[1], _bb[3]])
            _bb = pair["object"]["bbox"]
            bbox.append([_bb[0], _bb[2], _bb[1], _bb[3]])
            label.append(pair["subject"]["category"])
            label.append(pair["object"]["category"])
        bbox = np.stack(bbox).astype(np.float32)
        label = np.stack(label).astype(np.int32)

        img_file = os.path.join(self.img_dir, self.id_list[i])
        img = read_image(img_file, color=True)

        return img, bbox, label, 0

    __getitem__ = get_example


class VRDFullDataset:
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir

        json_file = os.path.join(data_dir, "annotations_{0}.json".format(split))
        self.data_json = json.load(open(json_file))
        self.id_list = list(self.data_json.keys())

        self.label_names = json.load(open(os.path.join(data_dir, "objects.json")))
        self.predicates_name = json.load(open(os.path.join(data_dir, "predicates.json")))

        self.img_dir = os.path.join(data_dir, "sg_dataset/sg_{0}_images".format(split))

        # all relationship triplets
        # (i, j, k)
        self.triplets = []
        for _, item in self.data_json.items():
            for anno in item:
                R = (anno["subject"]["category"], anno["object"]["category"], anno["predicate"])
                if not R in self.triplets:
                    self.triplets.append(R)

    def get_example(self, i):
        anno = self.data_json[self.id_list[i]]
        D_list = []
        for r in anno:
            i = r["subject"]["category"]
            j = r["object"]["category"]
            k = r["predicate"]
            O1 = [r["subject"]["bbox"][0], r["subject"]["bbox"][2], r["subject"]["bbox"][1], r["subject"]["bbox"][3]]
            O2 = [r["object"]["bbox"][0], r["object"]["bbox"][2], r["subject"]["bbox"][1], r["subject"]["bbox"][3]]
            D_list.append(((i, j, k), O1, O2))

        img_file = os.path.join(self.img_dir, self.id_list[i])
        img = read_image(img_file, color=True)
        return img, D_list

    __getitem__ = get_example


if __name__ == '__main__':
    test = VRDBboxDataset(r"F:\json_dataset_vrd")

from tqdm import tqdm
import torch as t
from utils.config import opt
from torchvision.models import vgg16
from torch.utils import data as data_
from data.dataset import Dataset, TestDataset, VRDDataset
from model.faster_rcnn_vgg16 import VGG16PREDICATES_PRE_TRAIN



def train(**kwargs):
    opt._parse(kwargs)

    dataset = VRDDataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)


    # vrd_trainer = nn.DataParallel(vrd_trainer, device_ids=[0])
    model = VGG16PREDICATES_PRE_TRAIN()
    optimizer = t.optim.Adam(model.parameters())

    for epoch in range(opt.vrd_epoch):
        for ii, (img, D) in tqdm(enumerate(dataloader)):
            if len(img) == 0:
                continue
            if D == [] or D[0] == []:
                continue

            img = img.cuda().float()

            total_loss = model(img, D)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        print(total_loss)
        # eval

if __name__ == '__main__':
    import fire

    fire.Fire()

        



if __name__ == '__main__':
    import fire

    fire.Fire()

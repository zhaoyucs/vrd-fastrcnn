from __future__ import absolute_import
import numpy as np
import torch as t
from torch import nn
import torchvision
from torchvision.models import vgg16
from torchvision.ops import RoIPool
from torchvision.transforms import CenterCrop

from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from utils import array_tool as at
from utils.config import opt
from scipy.spatial.distance import cosine


def decom_vgg16(**kwargs):
    # the 30th layer of features is relu of conv5_3
    if opt.caffe_pretrain:
        model = vgg16(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
    else:
        model = vgg16(not opt.load_path, **kwargs)

    features = list(model.features)[:30]
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier


def full_vgg16(**kwargs):
    # the 30th layer of features is relu of conv5_3
    # if opt.caffe_pretrain:
    #     model = vgg16(pretrained=False)
    #     if not opt.load_path:
    #         model.load_state_dict(t.load(opt.caffe_pretrain_path))
    # else:
    #     model = vgg16(not opt.load_path, **kwargs)
    
    # if opt.caffe_pretrain:
    #     model = vgg16(pretrained=False)
    #     if not opt.load_path:
    #         model.load_state_dict(t.load(opt.caffe_pretrain_path))
    # else:
    #     model = vgg16(not opt.load_path, **kwargs)
    model = vgg16(**kwargs)
    # features = list(model.features)
    # classifier = model.classifier
    #
    # classifier = list(classifier)
    # del classifier[6]
    # if not opt.use_drop:
    #     del classifier[5]
    #     del classifier[2]
    # classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    # for layer in model.features[:10]:
    #     for p in layer.parameters():
    #         p.requires_grad = False

    return model

class _test_vgg16(nn.Module):
    def __init__(self):
        super(_test_vgg16, self).__init__()
        self.model = vgg16(pretrained=False)
        if not opt.load_path:
            self.model.load_state_dict(t.load(opt.caffe_pretrain_path))
    
    @t.no_grad()
    def predict(self, x):
        return self.model(x)


def convert_coords(sc, oc):
    sx, sy, sw, sh = sc
    ox, oy, ow, oh = oc
    rx = min(sx, ox)
    ry = min(sy, oy)
    rw = max(sx + sw, ox + ow) - rx
    rh = max(sy + sh, oy + oh) - ry
    return rx, ry, rw, rh


def union_bbox(sub, obj):
    # bbox = [ymin, xmin, ymax, xmax]
    sub_y_min, sub_x_min, sub_y_max, sub_x_max = sub[0]
    obj_y_min, obj_x_min, obj_y_max, obj_x_max = obj[0]

    return t.tensor([min(sub_y_min, obj_y_min), min(sub_x_min, obj_x_min), max(sub_y_max, obj_y_max), max(sub_x_max, obj_x_max)])


def get_ruid(O1, O2, k):
    return (O1, O2), convert_coords(O1[1], O2[1]), k


class FasterRCNNVGG16(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self,
                 n_fg_class=100,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
        extractor, classifier = decom_vgg16()

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
        )


class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores

class VGG16PREDICATES_PRE_TRAIN(nn.Module):
    def __init__(self):
        super(VGG16PREDICATES_PRE_TRAIN, self).__init__()
        self.model = full_vgg16(num_classes=70).cuda()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, D_gt):

        finall_loss = 0
        for R, O1, O2 in D_gt:
            i, j, k = R
            u_bbox = union_bbox(O1, O2)
            mask = t.ones_like(x).bool()
            mask[:, :, u_bbox[0]:u_bbox[2], u_bbox[1]:u_bbox[3]] = False
            region = x.masked_fill(mask, 0)
            score = self.model(region)
            loss = self.loss(score, t.tensor(k).cuda())
            finall_loss += loss

        return finall_loss

class VGG16PREDICATES(nn.Module):
    def __init__(self, faster_rcnn, word2vec=None, D_samples=[], K_samples=500000, lamb1=0.05, lamb2=0.001):

        super(VGG16PREDICATES, self).__init__()
        self.faster_rcnn = faster_rcnn

        # self.extractor, self.classifier = full_vgg16()
        self.cnn_obj = full_vgg16(num_classes=100)
        self.cnn_rel = _test_vgg16()

        self.w2v = word2vec
        self.n = len(word2vec['obj'])
        self.k = len(word2vec['rel'])
        # parameter for V()
        self.Z = nn.Parameter(t.Tensor(self.k, 1000))
        # bias
        self.s =nn.Parameter(t.Tensor(self.k, 1))
        # parameter for f()
        self.W = nn.Parameter(t.Tensor(self.k, 600))
        # bias
        self.b = nn.Parameter(t.Tensor(self.k, 1))

        self.lamb1 = lamb1
        self.lamb2 = lamb2

        # sample number for Eq.4
        self.K_samples = K_samples
        if D_samples:
            self.sample_R_pairs(D_samples)

        self.init_w2v_tab()
        self.init_params()

    def init_params(self):
        nn.init.orthogonal_(self.Z)
        nn.init.orthogonal_(self.s)
        nn.init.orthogonal_(self.W)
        nn.init.orthogonal_(self.b)
        self.f_dict = {}
        self.V_dict = {}
        self.init_w2v_tab

    def forward(self, x, D_gt):

        img_size = x.shape[2:]

        # bboxes, labels,  scores = self.faster_rcnn.faster_rcnn.predict(x, [x.shape[2:]])

        # D = [i,j,k], O1, O2

        loss = self.loss(x, D_gt)
        return loss

    def init_w2v_tab(self):
        N, K = (range(self.n), range(self.k))

        co = lambda i1, i2: cosine(self.w2v['obj'][i1],   self.w2v['obj'][i2])
        cr = lambda k1, k2: cosine(self.w2v['rel'][k1],   self.w2v['rel'][k2])

        # TODO the bug here is what *could* have been causing the issues earlier!!!
        self.w2v_nt = t.Tensor([[co(i1, i2) for i2 in N] for i1 in N])
        self.w2v_vt = t.Tensor([[cr(k1, k2) for k2 in K] for k1 in K])

    def w2v_dist(self, R1, R2):
        i1, j1, k1 = R1
        i2, j2, k2 = R2
        return self.w2v_nt[i1, i2] + self.w2v_nt[j1, j2] + self.w2v_vt[k1, k2]

    def word_vec(self, i, j):
        return t.cat((t.tensor(self.w2v['obj'][i]).cuda(), t.tensor(self.w2v['obj'][j]).cuda()))

    def d(self, R1, R2):
        """
        Distance between two predicate triplets.

        """
        d_rel = self.func_f(R1) - self.func_f(R2)
        d_obj = self.w2v_dist(R1, R2)
        d = (d_rel ** 2) / d_obj
        return d if (d > 1e-10) else 1e-10

    def func_f(self, R):
        """
        Reduce relationship <i,j,k> to scalar language space.

        """
        i, j, k = R

        # if self.f_full_tab is not None:
        #     return self.f_full_tab[i, j, k]

        # if R not in self.f_dict:
        #     wvec = self.word_vec(i, j)
        #     f = t.dot(self.W[k].T, wvec) + self.b[k]
        #     self.f_dict[R] = f

        # return self.f_dict[R]
        wvec = self.word_vec(i, j)
        f = t.dot(self.W[k].squeeze(0), wvec) + self.b[k]
        return f

    # def func_f_full(self):
    #     if self.f_full_tab is None:
    #         w_i = self.w2v['obj'][None, ...]  # (1, N, 300)
    #         w_i = np.concatenate([w_i, np.zeros_like(w_i)], axis=2)  # (1, N, 600)
    #         w_j = self.w2v['obj'][:, np.newaxis, :]  # (N, 1, 300)
    #         w_j = np.concatenate([np.zeros_like(w_j), w_j], axis=2)  # (1, N, 600)
    #
    #         n, k = (self.n, self.k)
    #         B = np.reshape(np.tile(self.b, n ** 2), (k, n, n)).T  # (1,)  ->  (N, N, K)
    #
    #         tile_wv = w_i + w_j  # np auto-tiles `w_i`, `w_j` to (N, N, 300)
    #         F = np.tensordot(tile_wv, self.W, axes=(2, 1)) + B  # (N, N, 600)  x  (K, 600)
    #
    #         self.f_full_tab = F
    #         return F
    #     else:
    #         return self.f_full_tab

    def func_V(self, img, R, O1, O2, verbose=False):
        """
        Reduce relationship <i,j,k> to scalar visual space.

        """
        i, j, k = R
        u_bbox = union_bbox(O1, O2)
        # region = img[:, :, u_bbox[0]:u_bbox[2], u_bbox[1]:u_bbox[3]]
        mask = t.ones_like(img).bool()
        mask[:, :, u_bbox[0]:u_bbox[2], u_bbox[1]:u_bbox[3]] = False
        region = img.masked_fill(mask, 0)
        # _h = self.extractor(region)
        # print(_h.shape)
        # fc7 = self.classifier(_h)
        fc7 = self.cnn_rel.predict(region)

        # ruid = get_ruid(O1, O2, k)
        # _bboxes, _labels, _scores = self.faster_rcnn.faster_rcnn.predict(img, [img.shape[2:]])
        mask1 = t.ones_like(img).bool()
        mask1[:, :, O1[0][0]:O1[0][2], O1[0][1]:O1[0][3]] = False
        region1 = img.masked_fill(mask1, 0)
        score1 = self.cnn_obj(region1)
        P_i = score1[0][i]

        mask2 = t.ones_like(img).bool()
        mask2[:, :, O2[0][0]:O2[0][2], O2[0][1]:O2[0][3]] = False
        region2 = img.masked_fill(mask2, 0)
        score2 = self.cnn_obj(region2)
        P_j = score2[0][j]

        # P_i = _scores[0][i]
        # P_j = _scores[j]
        P_k = t.mm(self.Z[k], fc7.permute(1,0)) + self.s[k]
        # try:
        #     P_i = self.obj_probs[O1[:2]][i]
        #     P_j = self.obj_probs[O2[:2]][j]
        # except:
        #     import ipdb;
        #     ipdb.set_trace()
        #
        # # V_dict keeps a table of previously computed values by input
        # if ruid not in self.V_dict:
        #     rf = ruid2feats(ruid)
        #     try:
        #         cnn = self.rel_feats[rf]
        #     except:
        #         import ipdb;
        #         ipdb.set_trace()
        #     Z, s = (self.Z, self.s)
        #     self.V_dict[ruid] = np.dot(Z[k], cnn) + s[k]
        #
        # P_k = self.V_dict[ruid]

        if verbose:
            print('V: i {}  j {}  k {}'.format(P_i, P_j, P_k))

        return P_i * P_j * P_k

    def sample_R_pairs(self, triplets):
        # TODO other possibility is that we sample pairs in same image only
        m = self.K_samples
        samples = np.random.randint(0, len(triplets), m*2)
        R_samples = [triplets[s] for s in samples]
        self.R_pairs = zip(R_samples[:m], R_samples[m:])

    def func_K(self):
        """
        Eq (4): randomly sample relationship pairs and minimize variance.

        """
        # import ipdb; ipdb.set_trace()
        # R_rand = lambda: (randint(self.n), randint(self.n), randint(self.k))
        # R_samples = ((R_rand(), R_rand()) for n in range(self.K_samples))
        R_dists = t.tensor([self.d(R1, R2) for R1, R2 in self.R_pairs])
        return t.var(R_dists)

    def func_L(self, D):
        """
        Likelihood of relationships

        D: triplets of training data

        """
        # import ipdb; ipdb.set_trace()
        Rs, O1s, O2s = zip(*D)
        fn = lambda R1, R2: max(self.func_f(R1) - self.func_f(R2) + 1, 0)
        return sum(fn(R1, R2) for R1 in Rs for R2 in Rs)

    def func_C(self, img, D_gt):
        """
        Rank loss function

        """
        # import ipdb; ipdb.set_trace()
        C = 0.0
        for R, O1, O2 in D_gt:
            c_ = [self.func_V(img, R_, O1_, O2_) * self.func_f(R_) for R_, O1_, O2_ in D_gt
                  if (R_ != R) and (not O1_.equal(O1) or not O2_.equal(O2))]
            if c_ is not None and c_ != []:
                c_max = max(c_)
                c = self.func_V(img, R, O1, O2) * self.func_f(R)
                C += max(0, 1 - c + c_max)
        return C

    def loss(self, img, D_gt):
        """
        Final objective loss function.

        """
        C = self.func_C(img, D_gt)
        L = self.lamb1 * self.func_L(D_gt)
        K = self.lamb2 * self.func_K()
        return C + L + K


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()

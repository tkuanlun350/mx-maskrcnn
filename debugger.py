"""
This script help to verify mask label.
"""
from rcnn.pycocotools.mask import decode

import cPickle as pkl
import cv2
import numpy as np
import os
import cPickle
import random

with open('model/res50-fpn/oar/alternate_2/cache/OAR_mask_rcnn1.pkl') as f:
    maskdb = pkl.load(f)

with open('model/res50-fpn/oar/alternate_2/cache/OAR_roidb_rcnn1.pkl') as f:
    roidb = pkl.load(f)

if not os.path.exists('check'):
    os.mkdir('check')

def _mask_umap(mask_targets, mask_labels, mask_inds, num_rois, num_classes):
    _mask_targets = np.zeros((num_rois, num_classes, 28, 28), dtype=np.int8)
    _mask_weights = np.zeros((num_rois, num_classes, 1, 1), dtype=np.int8)
    _mask_targets[mask_inds, mask_labels] = mask_targets
    _mask_weights[mask_inds, mask_labels] = 1
    return _mask_targets, _mask_weights  # [num_rois, num_classes, 28, 28]

n_img = 0
for index, roi_rec in enumerate(roidb):

    im_path = roi_rec['image']
    mask_rec = maskdb[im_path]
    boxes = roi_rec['boxes']
    isflipped = roi_rec['flipped']
    if not isflipped:
        continue
    # im = cv2.imread(im_path)
    with open(im_path, 'rb') as f:
        _file = cPickle.load(f)
    im = _file['image']
    #print('process {}'.format(im_path))

    if isflipped:
        im = im[:, ::-1, :]

    mask_targets = mask_rec['mask_targets']
    mask_labels = mask_rec['mask_labels']
    mask_inds = mask_rec['mask_inds']
    num_classes = roi_rec['gt_overlaps'].shape[1]
    num_rois =  boxes.shape[0]
    mask_targets_decoded = np.concatenate([decode(encoded_mask).reshape([1, 28, 28]) for encoded_mask in mask_targets])
    _mask_targets, _mask_weights = _mask_umap(mask_targets_decoded, mask_labels, mask_inds, num_rois, num_classes)
    if isflipped:
        _mask_targets = np.flip(_mask_targets, -1)
    n_mask = 0
    for mask, idx in zip(mask_targets, mask_inds):
        box = boxes[idx]
        box = box.astype(np.int32)
        if (box[3]-box[1])*(box[2]-box[0]) < 1000:
            continue
        rgb = im[box[1]:box[3], box[0]:box[2], :]
        for _i in range(3):
            im = (im / np.max(im).astype(np.float32)) * 255
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), color=color, thickness=2)
        cv2.imwrite('check/{}_{}.jpg'.format(n_img, n_mask), im)

        mask = decode(mask)
        mask = mask*255
        if isflipped:
            mask = mask[:, ::-1]
        cv2.imwrite('check/{}_{}_mask.png'.format(n_img, n_mask), mask)

        mask_label = np.where(_mask_weights[idx]!=0)[0]
        _mask = _mask_targets[idx][mask_label]
        cv2.imwrite('check/{}_{}_mask_.png'.format(n_img, n_mask), _mask[0]*255)

        n_mask+=1

n_img+=1
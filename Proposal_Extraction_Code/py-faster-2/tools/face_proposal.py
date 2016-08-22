#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths

from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from PIL import Image
import math

CLASSES = ('__background__',
           'face')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
               'zf_faster_rcnn_iter_130000.caffemodel')}

count = 1
# save_dir='/data2/jiali/program/proposalDataBase/face-0.5-0.3/'
data_root='/data2/jiali/DataSet/wider_face/WIDER_train/'
save_dir='/data3/jiali/clutter-proposal/orginalPic/'
flabel=open(os.path.join(save_dir,'label-0.5-0.3.txt'),'w')

def checkIOU(bbox,gt):
    w=min(gt[2],bbox[2])-max(gt[0],bbox[0])
    h=min(gt[3],bbox[3])-max(gt[1],bbox[1])

    I=max(w,0)*max(h,0)
    U=(bbox[2]-bbox[0]+1)*(bbox[3]-bbox[1]+1)+(gt[2]-gt[0]+1)*(gt[3]-gt[1]+1)-I
    # print I/U
    return I/U


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]

    # transfer BGR to RGB
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],  #width
                          bbox[3] - bbox[1], fill=False,  #height
                          edgecolor='green', linewidth=3.5)
        )
        # ax.text(bbox[0], bbox[1] - 2,
        #         '{:s} {:.3f}'.format(class_name, score),
        #         bbox=dict(facecolor='blue', alpha=0.5),
        #         fontsize=14, color='white')

    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                               thresh),
    #               fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(net, image_name, gt):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(data_root,'JPEGImages',im_name)[:-1]
    im = cv2.imread(im_file)
    os.makedirs(os.path.join(save_dir,'pos',image_name))
    os.makedirs(os.path.join(save_dir,'neg',image_name))

    # Detect all object classes and regress object bounds
    scores, boxes, proposal = im_detect(net, im)

    # get proposal
    cls_scores = scores[:, 1];
    proposal = np.hstaclk((proposal, cls_scores[:, np.newaxis])).astype(np.float32)

    # check IOU between each proposal and each groundtruth
    global count
    for i in xrange(0,len(proposal[:,-1])):
        flag=0
        bbox=proposal[i,:4]
        im_save=im[int(bbox[1]):int(bbox[3])+1, int(bbox[0]):int(bbox[2])+1, :]
        if im_save.size==0:
            continue
        save_name=('{:08d}.jpg').format(count)

        # when bbox matches with one of the ground_truths, then break--label 1 ; else label 0
        tmp=[]
        for j in xrange(0,len(gt)):
            IOU=checkIOU(bbox,gt[j])
            if IOU>=0.5:
                flag=1
                flabel.write(('{:08d}.jpg 1\n').format(count))
                cv2.imwrite(os.path.join(save_dir,'pos',image_name,save_name),im_save)
                count+=1
                break
            if IOU<0.3:
                tmp.append(1)

        if flag==0 and len(tmp)==len(gt):
            flabel.write(('{:08d}.jpg 0\n').format(count))
            cv2.imwrite(os.path.join(save_dir,'neg',image_name,save_name),im_save)
            count+=1



def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.SCALES = [1000, ] #shorter side
    cfg.TEST.MAX_SIZE = 2500  #longer side
    args = parse_args()

    prototxt = os.path.join('/data2/jiali/program/face_detection/py-faster-2/models/wider_face', NETS['zf'][0],
                            'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join(os.getcwd(), '../output', 'faster_rcnn_end2end', 'wider_train',
                              NETS['zf'][1])
    print prototxt
    print caffemodel
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _, _ = im_detect(net, im)

    fp=open(os.path.join(data_root,'file_path.txt'),'r')  # file_path
    files=fp.readlines()

    fgt=open(os.path.join(data_root,'gt.txt'),'r')

    for im_name in files:
        print im_name
        k=fgt.readline()
        gt=[]
        for m in xrange(0,int(k)):
            temp=fgt.readline()[:-1].split(' ')
            box=np.array([float(x) for x in temp])
            gt.append(box)

        demo(net, im_name, gt)

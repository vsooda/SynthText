#coding=utf-8
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os.path as osp
import random, os
import cv2
import codecs
#import Image
from PIL import Image
import math
import traceback, itertools
from card_commom import *
from text_utils import RenderFont


def bbs_to_rects(bbs):
    rects = []
    for i in xrange(bbs.shape[2]):
        bb = bbs[:, :, i]
        left = int(bb[0,0])
        right = int(bb[0,1])
        top = int(bb[1,0])
        bottom = int(bb[1,3])
        #cv2.rectangle(dst, (left,top), (right,bottom), (255,0,0))
        rect = (left, top, right, bottom)
        rects.append(rect)
    return rects

def charrects_to_wordrect(rects):
    print len(rects)
    xmin = 10000
    ymin = 10000
    xmax = -1
    ymax = -1
    for left, top, right, bottom in rects:
        if left < xmin:
            xmin = left
        if top < ymin:
            ymin = top
        if right > xmax:
            xmax = right
        if bottom > ymax:
            ymax = bottom
    return (xmin, ymin, xmax, ymax)

def char2wordBB(charBB, text):
    """
    Converts character bounding-boxes to word-level
    bounding-boxes.

    charBB : 2x4xn matrix of BB coordinates
    text   : the text string

    output : 2x4xm matrix of BB coordinates,
             where, m == number of words.
    """
    wrds = text.split()
    bb_idx = np.r_[0, np.cumsum([len(w) for w in wrds])]
    wordBB = np.zeros((2,4,len(wrds)), 'float32')

    for i in xrange(len(wrds)):
        cc = charBB[:,:,bb_idx[i]:bb_idx[i+1]]

        # fit a rotated-rectangle:
        # change shape from 2x4xn_i -> (4*n_i)x2
        cc = np.squeeze(np.concatenate(np.dsplit(cc,cc.shape[-1]),axis=1)).T.astype('float32')
        rect = cv2.minAreaRect(cc.copy())
        box = np.array(cv2.boxPoints(rect))

        # find the permutation of box-coordinates which
        # are "aligned" appropriately with the character-bb.
        # (exhaustive search over all possible assignments):
        cc_tblr = np.c_[cc[0,:],
                        cc[-3,:],
                        cc[-2,:],
                        cc[3,:]].T
        perm4 = np.array(list(itertools.permutations(np.arange(4))))
        dists = []
        for pidx in xrange(perm4.shape[0]):
            d = np.sum(np.linalg.norm(box[perm4[pidx],:]-cc_tblr,axis=1))
            dists.append(d)
        wordBB[:,:,i] = box[perm4[np.argmin(dists)],:].T

    return wordBB

def do_crop(img, rects, texts, base_name, word_path, word_label):
    assert len(texts) == len(rects)
    index = 0
    with codecs.open(word_label, 'a', encoding='utf-8') as f:
        for left, top, right, bottom in rects:
            text_region = img[top:bottom, left:right]
            save_name = "%s_%02d.jpg" % (base_name, index)
            cv2.imwrite(word_path+save_name, text_region)
            f.write('%s %s\n' % (save_name, texts[index]))
            index = index + 1


if __name__ == '__main__':
    text_render = RenderFont()
    img_name = 'template/template.jpg'
    smu_name = 'template/smu2.jpg'
    img = cv2.imread(img_name)
    smu = cv2.imread(smu_name)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    print mask.shape
    h = img.shape[0]
    w = img.shape[1]
    margin_ratio = 8.0
    h_margin = int(h / margin_ratio)
    w_margin = int(w / margin_ratio)
    mask[h_margin:h-h_margin,w_margin:w-w_margin] = 255
    mask = mask.astype('float') / 255
    # cv2.imshow('orig', img)
    # cv2.imshow('mask', mask)
    # cv2.waitKey()
    # text_mask_orig = text_mask.copy()
    # bb_orig = bb.copy()
    # text_mask = self.warpHomography(text_mask, H, rgb.shape[:2][::-1])
    # bb = self.homographyBB(bb, Hinv)
    verbose = False


    color_mat = np.zeros(img.shape[:], dtype=np.float32)
    color_mat[:,:,2] = 1.0
    word_label = 'word.txt'
    word_path = 'card/'

    if not os.path.exists(word_path):
        os.makedirs(word_path)

    if word_path is not None:
        if os.path.exists(word_label):
            os.remove(word_label)

    index = 0
    max_num = 10
    while index < max_num:
        print index
        font = text_render.font_state.sample()
        font = text_render.font_state.init_font(font)
        text_mask, loc, bbs, text_pack = text_render.render_plate(font, mask)
        if text_mask is None:
            continue
        texts = text_pack.split()
        #cv2.rectangle(text_mask, bb)
        img_32f = img.astype('float')/255
        text_mask = text_mask.astype('float')/255
        text_mask_3 = np.zeros(img.shape[:], dtype=np.float32)
        text_mask_3[:,:,0] = text_mask
        text_mask_3[:,:,1] = text_mask
        text_mask_3[:,:,2] = text_mask
        dst = img_32f * (1-text_mask_3) + color_mat * text_mask_3
        #cv2.imshow('orig', img)
        dst = (dst * 255).astype('uint8')
        ibb = [bbs]
        charbbs =  np.concatenate(ibb, axis=2)
        wordbbs = char2wordBB(charbbs, text_pack)

        #xmin, ymin, xmax, ymax = charrects_to_wordrect(rects)
        if verbose:
            rects = bbs_to_rects(wordbbs)
            for xmin, ymin, xmax, ymax in rects:
                cv2.rectangle(dst, (xmin, ymin), (xmax, ymax), (255, 0, 255))

        rotation_dst, wordbbs = rot(dst, r(20) - 10, dst.shape, wordbbs, 20)
        rotation_dst, wordbbs = roll(rotation_dst, 30, (rotation_dst.shape[1], rotation_dst.shape[0]), wordbbs)

        rects = bbs_to_rects(wordbbs)
        if verbose:
            for xmin, ymin, xmax, ymax in rects:
                cv2.rectangle(rotation_dst, (xmin, ymin), (xmax, ymax), (255, 255, 255))

        #rects = bbs_to_rects(bbs)
        #xmin, ymin, xmax, ymax = charrects_to_wordrect(rects)
        #cv2.rectangle(rotation_dst, (xmin, ymin), (xmax, ymax), (255, 255, 255))
        base_name = 'card_%07d' % index
        rotation_dst = AddSmudginess(rotation_dst, smu)
        rotation_dst = AddGauss(rotation_dst, 1 + r(2))
        rotation_dst = addNoise(rotation_dst)
        do_crop(rotation_dst, rects, texts, base_name, word_path, word_label)


        if verbose:
            cv2.imshow('uint', dst)
            cv2.imshow('dst_rot', rotation_dst)
            cv2.waitKey()
        index = index + 1

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
import multiprocessing


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

def do_crop(img, rects, texts, base_name, crop_path, label_name):
    assert len(texts) == len(rects)
    index = 0
    #label_name = label_path + base_name + '.txt'
    with codecs.open(label_name, 'a', encoding='utf-8') as f:
        for left, top, right, bottom in rects:
            w_margin = int((right - left) / 20)
            h_margin = int((bottom - top) / 10)
            text_region = img[top-h_margin:bottom+h_margin, left-w_margin:right+w_margin]
            save_name = "%s_%02d.jpg" % (base_name, index)
            cv2.imwrite(crop_path+save_name, text_region)
            f.write('%s %s\n' % (save_name, texts[index]))
            index = index + 1



class Synthesizer(object):
    def __init__(self):
        self.queueLock = None
        self.workQueue = None

    def synth_worker(self):
        text_render = RenderFont()
        hash = random.getrandbits(128)
        label_name = '%s/%s.txt' % (self.label_path, hash)
        while True:
            self.queueLock.acquire()
            if not self.workQueue.empty():
                index = self.workQueue.get()
                self.queueLock.release()
                self.do_synth(index, text_render, label_name)
            else:
                self.queueLock.release()
                break

    def do_synth(self, index, text_render, label_name):
        print index
        #try:
        img = self.img
        mask = self.mask
        smu = self.smu
        color_mat = self.color_mat
        verbose = self.verbose
        font = text_render.font_state.sample()
        font = text_render.font_state.init_font(font)
        text_mask, loc, bbs, text_pack = text_render.render_plate(font, mask)
        if text_mask is None:
            return
        texts = text_pack.split()
        # cv2.rectangle(text_mask, bb)
        text_mask = text_mask.astype('float') / 255
        text_mask_3 = np.zeros(img.shape[:], dtype=np.float32)
        text_mask_3[:, :, 0] = text_mask
        text_mask_3[:, :, 1] = text_mask
        text_mask_3[:, :, 2] = text_mask
        dst = self.img_32f * (1 - text_mask_3) + color_mat * text_mask_3
        # cv2.imshow('orig', img)
        dst = (dst * 255).astype('uint8')
        ibb = [bbs]
        charbbs = np.concatenate(ibb, axis=2)

        # xmin, ymin, xmax, ymax = charrects_to_wordrect(rects)
        if self.verbose:
            rects = bbs_to_rects(charbbs)
            for xmin, ymin, xmax, ymax in rects:
                cv2.rectangle(dst, (xmin, ymin), (xmax, ymax), (255, 0, 255))

        rotation_dst, charbbs = rot(dst, r(20) - 10, dst.shape, charbbs, 20)
        rotation_dst, charbbs = roll(rotation_dst, 30, (rotation_dst.shape[1], rotation_dst.shape[0]), charbbs)
        wordbbs = char2wordBB(charbbs, text_pack)

        rects = bbs_to_rects(wordbbs)
        if verbose:
            for xmin, ymin, xmax, ymax in rects:
                cv2.rectangle(rotation_dst, (xmin, ymin), (xmax, ymax), (255, 255, 255))

        # rects = bbs_to_rects(bbs)
        # xmin, ymin, xmax, ymax = charrects_to_wordrect(rects)
        # cv2.rectangle(rotation_dst, (xmin, ymin), (xmax, ymax), (255, 255, 255))
        base_name = 'card_%07d' % index
        rotation_dst = AddSmudginess(rotation_dst, smu)
        rotation_dst = AddGauss(rotation_dst, 1 + r(2))
        rotation_dst = addNoise(rotation_dst)
        do_crop(rotation_dst, rects, texts, base_name, self.crop_path, label_name)

        if verbose:
            cv2.imshow('uint', dst)
            cv2.imshow('dst_rot', rotation_dst)
            cv2.waitKey()
        # except:
        #     print '>>>>>>continue'

    def synth_data(self):
        img_name = 'template/template.jpg'
        smu_name = 'template/smu2.jpg'
        self.img = cv2.imread(img_name)
        self.img_32f = self.img.astype('float')/255
        self.smu = cv2.imread(smu_name)
        self.mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        h = self.img.shape[0]
        w = self.img.shape[1]
        margin_ratio = 8.0
        h_margin = int(h / margin_ratio)
        w_margin = int(w / margin_ratio)
        self.mask[h_margin:h-h_margin,w_margin:w-w_margin] = 255
        self.mask = self.mask.astype('float') / 255
        # cv2.imshow('orig', img)
        # cv2.imshow('mask', mask)
        # cv2.waitKey()
        # text_mask_orig = text_mask.copy()
        # bb_orig = bb.copy()
        # text_mask = self.warpHomography(text_mask, H, rgb.shape[:2][::-1])
        # bb = self.homographyBB(bb, Hinv)
        self.verbose = False


        self.color_mat = np.zeros(self.img.shape[:], dtype=np.float32)
        self.color_mat[:,:,2] = 1.0
        self.label_path = 'card_labels/'
        self.crop_path = 'card/'

        if not os.path.exists(self.crop_path):
            os.makedirs(self.crop_path)

        if not os.path.exists(self.label_path):
            os.makedirs(self.label_path)

        index = 0
        max_num = 1000
        thread_num = 8
        self.queueLock = multiprocessing.Lock()
        self.workQueue = multiprocessing.Queue(max_num)
        self.threads = [multiprocessing.Process(target=self.synth_worker) for i in range(thread_num)]
        for i in xrange(0, max_num):
            self.queueLock.acquire()
            self.workQueue.put(i)
            self.queueLock.release()

        for thread in self.threads:
            thread.daemon = True
            thread.start()
        while not self.workQueue.empty():
            pass
        for t in self.threads:
            t.join(timeout=None)


if __name__ == '__main__':
    synther = Synthesizer()
    synther.synth_data()


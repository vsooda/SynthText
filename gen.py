#coding=utf-8
# Author: Ankush Gupta
# Date: 2015

"""
Entry-point for generating synthetic text images, as described in:

@InProceedings{Gupta16,
      author       = "Gupta, A. and Vedaldi, A. and Zisserman, A.",
      title        = "Synthetic Data for Text Localisation in Natural Images",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition",
      year         = "2016",
    }
"""

import numpy as np
import h5py
import os, sys, traceback
import os.path as osp
from synthgen import *
from common import *
import wget, tarfile
import codecs


## Define some configuration variables:
NUM_IMG = -1 # no. of images to use for generation (-1 to use all available):
INSTANCE_PER_IMAGE = 1 # no. of times to use the same image
SECS_PER_IMG = 100 #max time per image in seconds

# path to the data-file, containing image, depth and segmentation:
DATA_PATH = 'data'
bg_data = '/home/sooda/data/ocr/SynthTextBg/'
DB_FNAME = osp.join(bg_data,'dset.h5')
# url of the data (google-drive public file):
DATA_URL = 'http://www.robots.ox.ac.uk/~ankush/data.tar.gz'
OUT_FILE = 'results/SynthText.h5'

def get_data():
  """
  Download the image,depth and segmentation data:
  Returns, the h5 database.
  """
  if not osp.exists(DB_FNAME):
    try:
      colorprint(Color.BLUE,'\tdownloading data (56 M) from: '+DATA_URL,bold=True)
      print
      sys.stdout.flush()
      out_fname = 'data.tar.gz'
      wget.download(DATA_URL,out=out_fname)
      tar = tarfile.open(out_fname)
      tar.extractall()
      tar.close()
      os.remove(out_fname)
      colorprint(Color.BLUE,'\n\tdata saved at:'+DB_FNAME,bold=True)
      sys.stdout.flush()
    except:
      print colorize(Color.RED,'Data not found and have problems downloading.',bold=True)
      sys.stdout.flush()
      sys.exit(-1)
  # open the h5 file and return:
  return h5py.File(DB_FNAME,'r')


def save_cut_pics(imgname, res, result_img_dir, count, label_file):
    with codecs.open(label_file, 'a', encoding='utf-8') as csv:
        ninstance = len(res)
        for k in xrange(ninstance):
            m = 0
            space = 0
            space_count = 0
            rgb = res[k]['img']
            charBB = res[k]['charBB']
            wordBB = res[k]['wordBB']
            txt = res[k]['txt']
            image = Image.fromarray(rgb)

            txt = [t.decode('utf-8').strip() for t in txt]
            txt_len = len(txt)

            for i in xrange(wordBB.shape[-1]):
                bb = wordBB[:, :, i]
                bb = np.c_[bb, bb[:, 0]]

                leftx = 1000
                lefty = 1000
                rightx = -1000
                righty = -1000

                for j in xrange(4):
                    if j == 0:
                        if bb[0, j] < leftx:
                            leftx = int(round(bb[0, j]))
                        if bb[1, j] < lefty:
                            lefty = int(round(bb[1, j]))
                    if j == 1:
                        if bb[0, j] > rightx:
                            rightx = int(round(bb[0, j]))
                        if bb[1, j] < lefty:
                            lefty = int(round(bb[1, j]))
                    if j == 2:
                        if bb[0, j] > rightx:
                            rightx = int(round(bb[0, j]))
                        if bb[1, j] > righty:
                            righty = int(round(bb[1, j]))
                    if j == 3:
                        if bb[0, j] < leftx:
                            leftx = int(round(bb[0, j]))
                        if bb[1, j] > righty:
                            righty = int(round(bb[1, j]))

                if leftx < 0:
                    leftx = 0
                if lefty < 0:
                    lefty = 0
                width, height = image.size
                if rightx > width:
                    rightx = int(round(width))
                if righty > height:
                    righty = int(round(height))

                box = (leftx, lefty, rightx, righty)
                region = image.crop(box)
                region.save(result_img_dir + str(count) + '.jpg')

                lines = txt[m].split('\n')
                lines = [line.strip() for line in lines]
                if len(lines) > 1:
                    if space_count == 0:
                        space = len(lines)
                    if m < txt_len - 1:
                        if space_count < space:
                            csv.write('%s %s\n' % (str(count) + '.jpg', lines[space_count]))
                            space_count += 1
                            count += 1
                            if space_count == space:
                                m += 1
                                space_count = 0
                    else:
                        if space_count < space:
                            csv.write('%s %s\n' % (str(count) + '.jpg', lines[space_count]))
                            space_count += 1
                            count += 1
                            if space_count == space:
                                m = 0
                                space_count = 0
                else:
                    if m < txt_len - 1:
                        csv.write('%s %s\n' % (str(count) + '.jpg', txt[m]))
                        m += 1
                    elif m == txt_len - 1:
                        csv.write('%s %s\n' % (str(count) + '.jpg', txt[m]))
                        m = 0
                    count += 1
    return count



def add_res_to_db(imgname,res,db):
  """
  Add the synthetically generated text image instance
  and other metadata to the dataset.
  """
  ninstance = len(res)
  for i in xrange(ninstance):
    dname = "%s_%d"%(imgname, i)
    db['data'].create_dataset(dname,data=res[i]['img'])
    db['data'][dname].attrs['charBB'] = res[i]['charBB']
    db['data'][dname].attrs['wordBB'] = res[i]['wordBB']
    db['data'][dname].attrs['txt'] =  [a.encode('utf8') for a in res[i]['txt']]


def main(viz=False):
  # open databases:
  print colorize(Color.BLUE,'getting data..',bold=True)
  db = get_data()
  print colorize(Color.BLUE,'\t-> done',bold=True)

  # open the output h5 file:
  out_db = h5py.File(OUT_FILE,'w')
  out_db.create_group('/data')
  print colorize(Color.GREEN,'Storing the output in: '+OUT_FILE, bold=True)
  index = 20000
  label_file = 'result.txt'
  result_img_dir = 'cut_pics/'
  if not os.path.exists(result_img_dir):
      os.makedirs(result_img_dir)

  # get the names of the image files in the dataset:
  imnames = sorted(db['image'].keys())
  N = len(imnames)
  global NUM_IMG
  if NUM_IMG < 0:
    NUM_IMG = N
  start_idx,end_idx = 0,min(NUM_IMG, N)

  RV3 = RendererV3(DATA_PATH,max_time=SECS_PER_IMG)
  for i in xrange(start_idx,end_idx):
    imname = imnames[i]
    print imname
    try:
      # get the image:
      img = Image.fromarray(db['image'][imname][:])
      # get the pre-computed depth:
      #  there are 2 estimates of depth (represented as 2 "channels")
      #  here we are using the second one (in some cases it might be
      #  useful to use the other one):
      depth = db['depth'][imname][:].T
      depth = depth[:,:,1]
      # get segmentation:
      seg = db['seg'][imname][:].astype('float32')
      area = db['seg'][imname].attrs['area']
      label = db['seg'][imname].attrs['label']

      # re-size uniformly:
      sz = depth.shape[:2][::-1]
      img = np.array(img.resize(sz,Image.ANTIALIAS))
      seg = np.array(Image.fromarray(seg).resize(sz,Image.NEAREST))

      print colorize(Color.RED,'%d of %d'%(i,end_idx-1), bold=True)
      res = RV3.render_text(img,depth,seg,area,label,
                            ninstance=INSTANCE_PER_IMAGE,viz=viz)
      if len(res) > 0:
        # non-empty : successful in placing text:
        add_res_to_db(imname,res,out_db)
        #index = save_cut_pics(imname,res, result_img_dir, index, label_file)
      # visualize the output:
      if viz:
        if 'q' in raw_input(colorize(Color.RED,'continue? (enter to continue, q to exit): ',True)):
          break
    except:
      traceback.print_exc()
      print colorize(Color.GREEN,'>>>> CONTINUING....', bold=True)
      continue
  db.close()
  out_db.close()


if __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
  parser.add_argument('--viz',action='store_true',dest='viz',default=False,help='flag for turning on visualizations')
  args = parser.parse_args()
  main(args.viz)

import numpy as np
import h5py
import os, sys, traceback
import os.path as osp
import wget, tarfile
import cv2
from PIL import Image
import cPickle as cp




def add_more_data_into_dset(DB_FNAME,more_img_file_path,more_depth_path,more_seg_path, filter_name_path):
  db=h5py.File(DB_FNAME,'w')
  depth_db=h5py.File(more_depth_path,'r')
  seg_db=h5py.File(more_seg_path,'r')
  db.create_group('image')
  db.create_group('depth')
  db.create_group('seg')
  f = open(filter_name_path, 'r')
  imnames = cp.load(f)
  f.close()
  print len(imnames)
  for imname in imnames:
    if imname.endswith('.jpg'):
      full_path=more_img_file_path+imname
      print full_path,imname
      try:
          seg_data = seg_db['mask'][imname]
          j=Image.open(full_path)
          imgSize=j.size
          rawData=j.tobytes()
          img=Image.frombytes('RGB',imgSize,rawData)
          #img = img.astype('uint16')
          db['image'].create_dataset(imname,data=img)
          db['depth'].create_dataset(imname,data=depth_db[imname])
          db['seg'].create_dataset(imname,data=seg_data)
          db['seg'][imname].attrs['area']=seg_data.attrs['area']
          db['seg'][imname].attrs['label']=seg_data.attrs['label']
      except:
          print 'fail'
          continue
  db.close()
  depth_db.close()
  seg_db.close()


# path to the data-file, containing image, depth and segmentation:
base_path = "/home/sooda/data/ocr/SynthTextBg/"
DB_FNAME = base_path + 'dset_8000.h5'

#add more data into the dset
more_depth_path = base_path + 'depth.h5'
more_seg_path = base_path + 'seg.h5'
more_img_file_path = base_path + 'bg_img/'
filter_name_path = base_path + 'imnames.cp'

add_more_data_into_dset(DB_FNAME,more_img_file_path,more_depth_path,more_seg_path, filter_name_path)

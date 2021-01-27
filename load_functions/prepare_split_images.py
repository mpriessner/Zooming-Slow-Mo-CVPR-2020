#prepare_split_aug_images.py
#UPDATED CHECK
from skimage import io
import numpy as np
from tqdm import tqdm
import shutil
import os
from aicsimageio import AICSImage, imread
import shutil
import time
import numpy
import random
from aicsimageio import AICSImage, imread
from aicsimageio.writers import png_writer 
from tqdm import tqdm
from google.colab.patches import cv2_imshow
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from tqdm import tqdm
from timeit import default_timer as timer
import imageio
import tifffile 
from aicsimageio.transforms import reshape_data
from datetime import datetime

from tempfile import mkstemp
from shutil import move, copymode
from os import fdopen, remove


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def make_folder_with_date(save_location, name):
  today = datetime.now()
  if today.hour < 12:
    h = "00"
  else:
    h = "12"
  sub_save_location = save_location + "/" + today.strftime('%Y%m%d%H')+ "_"+ today.strftime('%H%M%S')+ "_%s"%name
  os.mkdir(sub_save_location)
  return sub_save_location


def diplay_img_info(img, divisor, use_RGB):
  ### display image data
    nr_z_slices = img.shape[1]
    nr_timepoints = img.shape[0]
    x_dim = img.shape[-2]
    y_dim = img.shape[-2] 
    x_div = x_dim//divisor
    y_div = y_dim//divisor
    print(img.shape)
    print("The Resolution is: " + str(x_dim))
    print("The number of z-slizes is: " + str(nr_z_slices))
    print("The number of timepoints: " + str(nr_timepoints))
    if use_RGB:
        nr_channels = img.shape[-1]
        print("The number of channels: " + str(nr_channels))
        nr_channels = 1
    else:
        nr_channels = 1
    return nr_z_slices, nr_channels, nr_timepoints, x_dim, y_dim, x_div, y_div 

def correct_channels(img):
  '''For 2D + T (with or without RGB) a artificial z channel gets created'''
  if img.shape[-1] ==3:
    use_RGB = True
  else:
    use_RGB = False
  if len(img.shape) ==4 and use_RGB:
    t, x, y, c = img.shape
    zeros = np.zeros((t,1,y,x,c))
    zeros[:,0,:,:,:] = img
    img = zeros
  elif len(img.shape) ==3 and not use_RGB:
    t, x, y = img.shape
    zeros = np.zeros((t,1,y,x))
    zeros[:,0,:,:] = img
    img = zeros
  return img, use_RGB
    

def change_train_file(zoomfactor, model_path):
  """This function changes the resolution value in the file: Vimeo7_dataset.py"""
  file_path_2 = "/content/ZoomInterpolation/codes/test_new.py"
  fh_2, abs_path_2 = mkstemp()
  with fdopen(fh_2,'w') as new_file:
    with open(file_path_2) as old_file:
      for counter, line in enumerate(old_file):
        if counter ==27:
          new_file.write(f"    scale = {zoomfactor}\n")
        elif counter == 34:
          new_file.write(f"    model_path = '{model_path}'\n")
        else:
          new_file.write(line)
  copymode(file_path_2, abs_path_2)
  #Remove original file
  remove(file_path_2)
  #Move new file
  move(abs_path_2, file_path_2) 

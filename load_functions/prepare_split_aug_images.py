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


def make_folder_with_date(save_location, name):
  today = datetime.now()
  if today.hour < 12:
    h = "00"
  else:
    h = "12"
  sub_save_location = save_location + "/" + today.strftime('%Y%m%d%H')+ "_"+ today.strftime('%H%M%S')+ "_%s"%name
  os.mkdir(sub_save_location)
  return sub_save_location


def diplay_img_info(img, divisor):
  ### display image data
    image_resolution = img.shape[-1]
    nr_z_slices = img.shape[2]
    nr_channels = img.shape[0]
    nr_timepoints = img.shape[1]
    x_dim = img.shape[-1]
    y_dim = img.shape[-2] 
    x_div = x_dim//divisor
    y_div = y_dim//divisor
    print(img.shape)
    print("The Resolution is: " + str(image_resolution))
    print("The number of z-slizes is: " + str(nr_z_slices))
    print("The number of timepoints: " + str(nr_timepoints))
    print("The number of channels: " + str(nr_channels))
    return nr_z_slices, nr_channels, nr_timepoints, x_dim, y_dim, x_div, y_div 


def rotation_aug(source_img, name, path, flip=False):
    print(source_img.shape)
    # Source Rotation
    source_img_90 = np.rot90(source_img,axes=(2,3))
    source_img_180 = np.rot90(source_img_90,axes=(2,3))
    source_img_270 = np.rot90(source_img_180,axes=(2,3))
    # Add a flip to the rotation
    if flip == True:
      source_img_lr = np.fliplr(source_img)
      source_img_90_lr = np.fliplr(source_img_90)
      source_img_180_lr = np.fliplr(source_img_180)
      source_img_270_lr = np.fliplr(source_img_270)

      #source_img_90_ud = np.flipud(source_img_90)
    # Save the augmented files
    # Source images
    with OmeTiffWriter(path + "/" + name + ".tif") as writer2:
        writer2.save(source_img, dimension_order='TZYX')  
    with OmeTiffWriter(path + "/" + name +'_90.tif') as writer2:
        writer2.save(source_img_90, dimension_order='TZYX')  
    with OmeTiffWriter(path + "/" + name +'_180.tif') as writer2:
        writer2.save(source_img_180, dimension_order='TZYX')  
    with OmeTiffWriter(path + "/" + name +'_270.tif') as writer2:
      writer2.save(source_img_270, dimension_order='TZYX')  
    # Target images
   
    if flip == True:
      with OmeTiffWriter(path + "/" + name + '_lr.tif') as writer2:
        writer2.save(source_img_lr, dimension_order='TZYX')  
      with OmeTiffWriter(path + "/" + name + '_90_lr.tif') as writer2:
          writer2.save(source_img_90_lr, dimension_order='TZYX')  
      with OmeTiffWriter(path + "/" + name + '_180_lr.tif') as writer2:
          writer2.save(source_img_180_lr, dimension_order='TZYX')  
      with OmeTiffWriter(path + "/" + name + '_270_lr.tif') as writer2:
        writer2.save(source_img_270_lr, dimension_order='TZYX')  

 
def flip(source_img, name, path):
    source_img_lr = np.fliplr(source_img)
    with OmeTiffWriter(path + "/" + name + ".tif") as writer2:
        writer2.save(source_img, dimension_order='TZYX')  
    with OmeTiffWriter(path + "/" + name + '_lr.tif') as writer2:
      writer2.save(source_img_lr, dimension_order='TZYX')

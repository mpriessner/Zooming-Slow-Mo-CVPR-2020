#WORKING
import os
import sys
sys.path.insert(0,'/content/ZoomInterpolation/load_functions')
from skimage import io
import numpy as np
from tqdm import tqdm
import shutil
from aicsimageio import AICSImage, imread
import time
import random
from aicsimageio import AICSImage, imread
from aicsimageio.writers import png_writer 
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from timeit import default_timer as timer
import imageio
import tifffile 
from aicsimageio.transforms import reshape_data
from datetime import datetime

def downsample_z_creation(img_path_list, file_num, sub_save_location):
    os.chdir(sub_save_location)
    t, z, y_dim,x_dim, img, use_RGB = load_img(img_path_list[file_num])
    # folder_steps = str(file_num) + "_steps"
    img_nr = img_path_list[file_num].split("/")[-1].split(".")[0].split("-")[1][:3]
    fr_nr = img_path_list[file_num].split("/")[-1].split(".")[0].split("-")[2][:2]

    #create new directory-path
    for num_t in tqdm(range(0,t)):
        folder_name = "i-{}_".format(img_nr) + "f-{}_".format(fr_nr) + "t-%03d"%(num_t)
        os.chdir(sub_save_location)
        folder = os.path.join(sub_save_location,folder_name)
        os.mkdir(folder)
        os.chdir(folder)
        for num_z in range(z):
          if (num_z % 2) == 0:
            #create new directory-path
            file_name = ("dz_%03d" %(num_z))

            # #here put the image pngs into the folder (instead of creating the folder)
            # #convert image to unit8 otherwise warning
            if use_RGB == False:
              img_save_1 = img[num_t,num_z, :, :] 
              img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
              img_save_1 = convert(img_save_1, 0, 255, np.uint8)
            elif use_RGB == True:
              img_save_1 = img[num_t,num_z, :, :, :] 
              img_save_1 = convert(img_save_1, 0, 255, np.uint8)
              # # saving images as PNG
            io.imsave("{}.png".format(file_name), img_save_1)

          #save the last slide on top labeled with x
          if num_z == z-1 and (num_z % 2) != 0:
            file_name = ("dz_%03d" %(num_z))

            # #here put the image pngs into the folder (instead of creating the folder)
            # #convert image to unit8 otherwise warning 
            if use_RGB == False:
              img_save_1 = img[num_t,num_z, :, :] 
              img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
              img_save_1 = convert(img_save_1, 0, 255, np.uint8)
            elif use_RGB == True:
              img_save_1 = img[num_t,num_z, :, :, :] 
              img_save_1 = convert(img_save_1, 0, 255, np.uint8)              
            # # saving images as PNG
            io.imsave("{}-x.png".format(file_name), img_save_1)


def downsample_t_creation(img_path_list, file_num, sub_save_location):
    os.chdir(sub_save_location)
    t, z, y_dim,x_dim, img, use_RGB = load_img(img_path_list[file_num])
    # folder_steps = str(file_num) + "_steps"
    img_nr = img_path_list[file_num].split("/")[-1].split(".")[0].split("-")[1][:3]
    fr_nr = img_path_list[file_num].split("/")[-1].split(".")[0].split("-")[2][:2]

    #create new directory-path
    for num_z in tqdm(range(0,z)):
        folder_name = "i-{}_".format(img_nr) + "f-{}_".format(fr_nr) + "z-%03d"%(num_z)
        os.chdir(sub_save_location)
        folder = os.path.join(sub_save_location,folder_name)
        os.mkdir(folder)
        os.chdir(folder)
        for num_t in range(t):
          if (num_t % 2) == 0:
            #create new directory-path
            file_name = ("dt_%03d" %(num_t))

            # #here put the image pngs into the folder (instead of creating the folder)
            # #convert image to unit8 otherwise warning
            if use_RGB == False:
              img_save_1 = img[num_t,num_z, :, :] 
              img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
              img_save_1 = convert(img_save_1, 0, 255, np.uint8)
            elif use_RGB == True:
              img_save_1 = img[num_t,num_z, :, :, :] 
              img_save_1 = convert(img_save_1, 0, 255, np.uint8)
              # # saving images as PNG
            io.imsave("{}.png".format(file_name), img_save_1)

          #save the last slide on top labeled with x
          if num_t == t-1 and (num_t % 2) != 0:
            file_name = ("dt_%03d" %(num_t))

            if use_RGB == False:
              # #here put the image pngs into the folder (instead of creating the folder)
              # #convert image to unit8 otherwise warning
              img_save_1 = img[num_t,num_z, :, :] 
              img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
              img_save_1 = convert(img_save_1, 0, 255, np.uint8)
            elif use_RGB == True:
              img_save_1 = img[num_t,num_z, :, :, :] 
              img_save_1 = convert(img_save_1, 0, 255, np.uint8)
              # # saving images as PNG
            io.imsave("{}-x.png".format(file_name), img_save_1)

            
def upsample_t_creation(img_path_list, file_num, sub_save_location, folder_option):
    # to differentiate between zoom and normal upsampling in t dim
    if folder_option =="zoom":
      marker = "z"
    else:
      marker = "u"
    os.chdir(sub_save_location)
    t, z, y_dim,x_dim, img, use_RGB = load_img(img_path_list[file_num])
    # folder_steps = str(file_num) + "_steps"
    img_path = img_path_list[file_num]
    img_nr = img_path.split("/")[-1].split(".")[0].split("-")[1][:3]
    fr_nr = img_path.split("/")[-1].split(".")[0].split("-")[2][:2]
    
    #create new directory-path
    for num_z in tqdm(range(0,z)):   # dim_2 = zdimension
        folder_name = "i-{}_".format(img_nr) + "f-{}_".format(fr_nr) + "z-%03d"%(num_z) # z doesn't need to be the z dimension because it is also used for the t dimension
        os.chdir(sub_save_location)
        folder = os.path.join(sub_save_location,folder_name)
        os.mkdir(folder_name)
        os.chdir(folder_name)
        for num_t in range(t):
          #create new directory-path
          file_name = (f"{marker}t_%03d" %(num_t))

          # #here put the image pngs into the folder (instead of creating the folder)
          # #convert image to unit8 otherwise warning

          if use_RGB == False:
            img_save_1 = img[num_t,num_z, :, :] 
            img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
            img_save_1 = convert(img_save_1, 0, 255, np.uint8)
          elif use_RGB == True:
            img_save_1 = img[num_t,num_z, :, :, :] 
            img_save_1 = convert(img_save_1, 0, 255, np.uint8)
            # # saving images as PNG
          io.imsave("{}.png".format(file_name), img_save_1)
            # writer1.save(img_save_1)



def upsample_z_creation(img_path_list, file_num, sub_save_location):
    os.chdir(sub_save_location)
    t, z, y_dim,x_dim, img, use_RGB = load_img(img_path_list[file_num]) #dim_1=t, dim_2=z
    # folder_steps = str(file_num) + "_steps"
    img_nr = img_path_list[file_num].split("/")[-1].split(".")[0].split("-")[1][:3]
    fr_nr = img_path_list[file_num].split("/")[-1].split(".")[0].split("-")[2][:2]
    # folder_file_path = os.path.join(sub_save_location,file_to_folder_name)
    # os.mkdir(folder_file_path)

    #create new directory-path
    for num_t in tqdm(range(0,t)):
        folder_name = "i-{}_".format(img_nr) + "f-{}_".format(fr_nr) + "z-%03d"%(num_t)
        os.chdir(sub_save_location)
        folder = os.path.join(sub_save_location,folder_name)
        os.mkdir(folder_name)
        os.chdir(folder_name)
        for num_z in range(z):
          #create new directory-path
          file_name = ("uz_%03d"%(num_z))
          # #convert image to unit8 otherwise warning

          if use_RGB == False:
            img_save_1 = img[num_t,num_z, :, :] 
            img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
            img_save_1 = convert(img_save_1, 0, 255, np.uint8)
          elif use_RGB == True:
            img_save_1 = img[num_t,num_z, :, :, :] 
            img_save_1 = convert(img_save_1, 0, 255, np.uint8)
            # # saving images as PNG
          io.imsave("{}.png".format(file_name), img_save_1)

            
def get_img_path_list(img_path_list, img_folder_path):
  ''' Creates a list of image-path that will be used for loading the images later'''
  flist = os.listdir(img_folder_path)
  flist.sort()
  for i in flist:
    img_slice_path = os.path.join(img_folder_path, i)
    img_path_list.append(img_slice_path)
  return img_path_list
# img_path_list = get_img_path_list_T(img_path_list, filepath, folder_list)
# img_path_list


def load_img(img_path):
    img = io.imread(img_path)
    if img.shape[-1]==3:
      use_RGB = True
      t, z, y_dim, x_dim, _ = img.shape 
      print("This image will be processed as a RGB image")
    else:
      use_RGB = False
      t, z, y_dim, x_dim = img.shape 
    print("The image dimensions are: " + str(img.shape))
    return t, z, y_dim,x_dim, img, use_RGB
  
    
def make_folder_with_date(save_location, name):
  today = datetime.now()
  if today.hour < 12:
    h = "00"
  else:
    h = "12"
  sub_save_location = save_location + "/" + today.strftime('%Y%m%d')+ "_"+ today.strftime('%H%M%S')+ "_%s"%name
  os.mkdir(sub_save_location)
  return sub_save_location


def create_3D_image(img, x_dim, y_dim):
# creates 3D image with 3 times the same values for RGB because the NN was generated for normal rgb images dim(3,x,y)
  # print(img.shape)
  image_3D = np.zeros((x_dim,y_dim,3))
  image_3D[:,:,0] = img
  image_3D[:,:,1] = img
  image_3D[:,:,2] = img
  return image_3D


def convert(img, target_type_min, target_type_max, target_type):
  # this function converts images from float32 to unit8 
    imin = img.min()
    imax = img.max()
    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

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



#############################################################################

def data_preparation_for_zoominterpolation(folder_option, save_location):
    if folder_option == "upsample-t":
      name = "upsample-t"
      img_path_list = []
      img_path_list = get_img_path_list(img_path_list, split_img_folder_path) 
      sub_save_location = make_folder_with_date(save_location, name)
      for file_num in range(len(img_path_list)):
        upsample_t_creation(img_path_list, file_num, sub_save_location, folder_option)

    elif folder_option == "upsample-z":
      name = "upsample-z"
      img_path_list = []
      img_path_list = get_img_path_list(img_path_list, split_img_folder_path) 
      sub_save_location = make_folder_with_date(save_location, name)
      for file_num in range(len(img_path_list)):
        upsample_z_creation(img_path_list, file_num, sub_save_location,)

    elif folder_option == "downsample-t":
      name = "downsample-t"
      img_path_list = []
      img_path_list = get_img_path_list(img_path_list, split_img_folder_path) 
      sub_save_location = make_folder_with_date(save_location, name)
      for file_num in range(len(img_path_list)):
        downsample_t_creation(img_path_list, file_num, sub_save_location,)

    elif folder_option == "downsample-z":
      name = "downsample-z"
      img_path_list = []
      img_path_list = get_img_path_list(img_path_list, split_img_folder_path) 
      sub_save_location = make_folder_with_date(save_location, name)
      for file_num in range(len(img_path_list)):
        downsample_z_creation(img_path_list, file_num, sub_save_location)
        
    elif folder_option == "zoom":
      name = "zoom"
      img_path_list = []
      img_path_list = get_img_path_list(img_path_list, split_img_folder_path) 
      sub_save_location = make_folder_with_date(save_location, name)
      for file_num in range(len(img_path_list)):
        upsample_t_creation(img_path_list, file_num, sub_save_location, folder_option)
    return sub_save_location


from preparation_for_training import change_Sakuya_arch

def prepare_files_for_zoominterpolation_step(sub_save_location, pretrained_model_path, use_fine_tuned_models):
    img_folder_path_interpolate = sub_save_location

    !rm -rf "/content/ZoomInterpolation/test_example"
    shutil.copytree(img_folder_path_interpolate,"/content/ZoomInterpolation/test_example")
    os.chdir("/content/ZoomInterpolation/codes")

    if use_fine_tuned_models:
      if zoomfactor ==1:
        change_train_file(zoomfactor, pretrained_model_path)
        change_Sakuya_arch(zoomfactor)
      elif zoomfactor ==2:
        change_train_file(zoomfactor, pretrained_model_path)
        change_Sakuya_arch(zoomfactor)
      elif zoomfactor ==4:
        change_train_file(zoomfactor, pretrained_model_path)
        change_Sakuya_arch(zoomfactor)
    else:
      pretrained_model_path_1x = "/content/ZoomInterpolation/experiments/pretrained_models/pretrained_1x.pth"
      pretrained_model_path_2x = "/content/ZoomInterpolation/experiments/pretrained_models/pretrained_2x.pth"
      pretrained_model_path_4x = "/content/ZoomInterpolation/experiments/pretrained_models/pretrained_4x.pth"
      if zoomfactor ==1:
        change_train_file(zoomfactor, pretrained_model_path_1x)
        change_Sakuya_arch(zoomfactor)
      elif zoomfactor ==2:
        change_train_file(zoomfactor, pretrained_model_path_2x)
        change_Sakuya_arch(zoomfactor)
      elif zoomfactor ==4:
        change_train_file(zoomfactor, pretrained_model_path_4x)
        change_Sakuya_arch(zoomfactor)
  return img_folder_path_interpolate

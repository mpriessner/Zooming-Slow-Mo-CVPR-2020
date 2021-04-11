# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 08:22:06 2021

@author: Martin_Priessner
"""
import os
import os
import math
from tqdm import tqdm
import cv2 
import numpy as np
from skimage import io
import shutil
import random
import pandas as pd
from tqdm import tqdm


#### Load the necessary functions
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

def generate_mod_LR(up_scale, sourcedir, savedir, train_guide, test_guide, continue_loading, N_frames, log_path):
    """This function generates the high and low resulution images in a given output folder"""

    create_folder_list_from_txt_guide(train_guide, test_guide)

    save_HR = os.path.join(savedir, 'HR')
    save_LR = os.path.join(savedir, 'LR')
 
    saveHRpath = os.path.join(savedir, 'HR', 'x' + str(up_scale))
    saveLRpath = os.path.join(savedir, 'LR', 'x' + str(up_scale))

    if not os.path.isdir(sourcedir):
        print('Error: No source data found')
        exit(0)
      
    # Create folder system
    if continue_loading == False:
        print("Restart loading")
        if os.path.isdir(savedir):
          shutil.rmtree(savedir)
          os.mkdir(savedir)
        else:
          os.mkdir(savedir)

        os.mkdir(save_HR)
        os.mkdir(save_LR)
        
        os.mkdir(saveHRpath)
        prep_folder_structure(saveHRpath)

        os.mkdir(saveLRpath)
        prep_folder_structure(saveLRpath)

        # copy the set_guide text files in each folder (HR, LR)
        train_guide_HR = saveHRpath[:-3]+"/sep_trainlist.txt"
        train_guide_LR = saveLRpath[:-3]+"/sep_trainlist.txt"

        test_guide_HR = saveHRpath[:-3]+"/sep_testlist.txt"
        test_guide_LR = saveLRpath[:-3]+"/sep_testlist.txt"

        shutil.copy(train_guide, train_guide_HR)
        shutil.copy(train_guide, train_guide_LR)

        shutil.copy(test_guide, test_guide_HR)
        shutil.copy(test_guide, test_guide_LR)
        with open(log_path, "w") as f:
            f.write("start")
        with open(log_path, "a") as f:
            f.write(f'Created new folders: {savedir} \n')
            f.write(f'Created new folders: {save_HR}\n')
            f.write(f'Created new folders: {save_LR}\n')
            f.write(f'Created new folders: {saveHRpath}\n')
            f.write(f'Created new file: {train_guide_HR}\n')
            f.write(f'Created new file: {test_guide_LR}\n')
    else:
        with open(log_path, "w") as f:
            f.write("start")
    filepaths = get_all_filepaths(sourcedir, N_frames)
    print(f"number of files: {len(filepaths)}")
    num_files = len(filepaths)

 # # prepare data with augementation
    for i in tqdm(range(num_files)):
        filename = filepaths[i]
        file_folder_path = filename[-18:]
        # check if file was already processed
        file_checker_path = os.path.join(saveHRpath, file_folder_path)
        if os.path.exists(file_checker_path):
          with open(log_path, "a") as f:
            f.write(f"File already exists: {file_checker_path}\n")
          continue
        else: 
          try:
            with open(log_path, "a") as f:
              f.write('No.{} -- Processing {}\n'.format(i, filename))
            # read image
            image = cv2.imread(filename)

            width = int(np.floor(image.shape[1] / up_scale))
            height = int(np.floor(image.shape[0] / up_scale))
            # modcrop
            if len(image.shape) == 3:
                image_HR = image[0:up_scale * height, 0:up_scale * width, :]
            else:
                image_HR = image[0:up_scale * height, 0:up_scale * width]
            # LR
            image_LR = imresize_np(image_HR, 1 / up_scale, True)
            file_folder_path = filename[-18:]
            cv2.imwrite(os.path.join(saveHRpath, file_folder_path), image_HR)
            cv2.imwrite(os.path.join(saveLRpath, file_folder_path), image_LR)
          except:
            with open(log_path, "a") as f:
              f.write('No.{} -- failed {}\n'.format(i, filename))     

    return save_HR, save_LR

def get_img_dim(img):
  """This function gets the right x,y,t,z dimensions and if it is an RGB image or not"""
  if (img.shape[-1] ==3 and len(img.shape) ==5):
    use_RGB = True
    t_dim, z_dim, y_dim, x_dim, _ = img.shape
  elif (img.shape[-1] ==3 and len(img.shape) ==4):
    use_RGB = True
    t_dim, y_dim, x_dim, channel = img.shape
    zeros = np.zeros((t_dim,1,y_dim,x_dim,channel))
    zeros[:,0,:,:,:] = img
    img = zeros
    t_dim, z_dim, y_dim, x_dim, _ = img.shape
  elif (img.shape[-1] !=3 and len(img.shape) ==3):  # create a 4th dimension
    use_RGB = False
    t, y, x = img.shape
    zeros = np.zeros((t,1,y,x))
    zeros[:,0,:,:] = img
    img = zeros
    t_dim, z_dim, y_dim, x_dim= img.shape
  elif (img.shape[-1] !=3 and len(img.shape) ==4):
    use_RGB = False
    t_dim, z_dim, y_dim, x_dim = img.shape
  return t_dim, z_dim, y_dim, x_dim, use_RGB

def get_all_filepaths_in_folder(folder_path):
    '''This function gets the paths from each file in folder and subfolder of a given location'''
    flist = []
    for path, subdirs, files in tqdm(os.walk(folder_path)):
          for name in files:
            flist.append(os.path.join(path, name))
    return flist




######################## SELECT SOURCE FOLDER ########################
#@markdown Provide the folder with the training data
# Define the necessary paths needed later
Source_path = r'E:\Outsourced_Double\BF_data_for_training\SRFBN\1024'#@param {type:"string"}

Parent_path = os.path.dirname(Source_path)
test_train_seq_path = os.path.join(Parent_path, "sequences")

######################## SELECT test_train_split  ########################
# Paramenters
test_train_split = 0.1 #@param {type:"slider", min:0.1, max:1, step:0.1}
N_frames = 7 

# create seq_lists *.txt
train_seq_txt = os.path.join(Parent_path, "sep_trainlist.txt")
test_seq_txt = os.path.join(Parent_path, "sep_testlist.txt")
with open(train_seq_txt, "w") as f:
  f.write("")
with open(test_seq_txt, "w") as f:
  f.write("")

# delete test_train_folder if already exists
if os.path.isdir(test_train_seq_path):
  shutil.rmtree(test_train_seq_path)
os.mkdir(test_train_seq_path)

#get all files in the selected folder
flist = get_all_filepaths_in_folder(Source_path)

# split the different images and save them in the sequence folder with a given folderstructure
# and create the test train split seq txt files
for counter_1, file_path in tqdm(enumerate(flist)):
  os.chdir(test_train_seq_path)
  file_folder = "%05d"%(counter_1+1)
  print(file_folder)
  os.mkdir(file_folder)
  file_folder_path = os.path.join(test_train_seq_path, file_folder)
  os.chdir(file_folder_path)

  img = io.imread(file_path)
  # makes 3D into 4D dataset if needed
  img, _ = correct_channels(img)
  t_dim, z_dim, y_dim, x_dim, use_RGB = get_img_dim(img)

  #calculate how many folders need to be created to cover all the images
  N_folders_per_slice = math.ceil(t_dim/N_frames)
  counter_2 = 1
  for z in tqdm(range(z_dim)):
    if (use_RGB and len(img.shape) ==5) or (use_RGB and len(img.shape) ==4):
       img_slice = img[:, z, :, :, :]
    else:
      img_slice = img[:, z, :, :]
   
    for seq in range(1,N_folders_per_slice):
      seq_folder = "%04d"%(counter_2)
      seq_folder_path = os.path.join(file_folder_path, seq_folder)
      os.mkdir(seq_folder_path)
      counter_2 += 1
      #create lever to randomly shift samples to test or train depending on the chosen split 
      test_train_lever = random.uniform(0, 1)
      if test_train_lever < test_train_split:
        with open(test_seq_txt, "a") as f:
          f.write(f"{file_folder}/{seq_folder}\n")
      else:
        with open(train_seq_txt, "a") as f:
          f.write(f"{file_folder}/{seq_folder}\n")

      #save a given number of images as png in the new folder
      for im_num in range(1, N_frames+1):
        # print(((seq-1)*N_frames+(im_num-1)), y_dim, x_dim)
        png_img_path = os.path.join(seq_folder_path, f"im{im_num}.png")
        if use_RGB:
            png_img = img_slice[((seq-1)*N_frames+(im_num-1)), :, :, :]
            img_channels = png_img
        else:
            png_img = img_slice[((seq-1)*N_frames+(im_num-1)), :, :]
            img_channels = np.zeros((y_dim, x_dim, 3))
            img_channels[:,:,0] = png_img
            img_channels[:,:,1] = png_img
            img_channels[:,:,2] = png_img
        io.imsave(png_img_path, img_channels)
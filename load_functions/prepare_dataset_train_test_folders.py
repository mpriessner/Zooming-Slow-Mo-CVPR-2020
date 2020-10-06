import os
from aicsimageio import AICSImage, imread
import shutil
import time
import numpy
import random
from aicsimageio import AICSImage, imread
from aicsimageio.writers import png_writer 
import numpy as np
from tqdm import tqdm
from google.colab.patches import cv2_imshow
from aicsimageio.writers.ome_tiff_writer import OmeTiffWriter
from tqdm import tqdm
from timeit import default_timer as timer
import imageio
import tifffile 
from aicsimageio.transforms import reshape_data
from datetime import datetime
import math

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
    img = AICSImage(img_path)
    img = img.get_image_data("TZYX", S=0)  # in my case channel is the Time
    print("The image dimensions are: " + str(img.shape))
    t, z, y_dim, x_dim = img.shape[:]
    return t, z, y_dim,x_dim, img
  

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
  image_3D = np.zeros((3,x_dim,y_dim))
  image_3D[0] = img
  image_3D[1] = img
  image_3D[2] = img
  return image_3D

def convert(img, target_type_min, target_type_max, target_type):
  # this function converts images from float32 to unit8 
    imin = img.min()
    imax = img.max()
    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


''' the following funcitons prepare the dataset in a way that it form the necessary folder system for the NN to handle the data correctly
'''
def upsample_z_creation(img_path_list, file_num, sub_save_location):
    os.chdir(sub_save_location)
    t, z, y_dim,x_dim, img = load_img(img_path_list[file_num])
    folder_steps = str(file_num) + "_steps"
    folder_steps = os.path.join(sub_save_location,folder_steps)
    os.mkdir(folder_steps)

    # txt_name_log = open(destination + "/name_log.txt", "a")
    # txt_name_log.write("{}, {}\n".format(new_folder_name, img_path_list[file_num]), )
    # txt_name_log.close()

    #create new directory-path
    for t_num in tqdm(range(0,t)):
        for z_num in range(z-1):
          #create new directory-path
          file_folder = ("f_%03d" %(file_num) + "-"+"t_%03d" %(t_num) +"-" +"z_%03d"%(z_num))
          os.chdir(folder_steps)
          os.mkdir(file_folder)
          steps_path_folder = os.path.join(folder_steps, file_folder)
          os.chdir(steps_path_folder)

          # #here put the image pngs into the folder (instead of creating the folder)
          # #convert image to unit8 otherwise warning
          img_save_1 = reshape_data(img, "TZXY","XY", T=t_num, Z=z_num)
          img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
          img_save_1 = convert(img_save_1, 0, 255, np.uint8)

          img_save_2 = reshape_data(img, "TZXY","XY", T=t_num, Z=z_num+1)
          img_save_2 = create_3D_image(img_save_2, x_dim, y_dim)
          img_save_2 = convert(img_save_2, 0, 255, np.uint8)

          # # saving images as PNG
          with png_writer.PngWriter("im1.png") as writer1:
            writer1.save(img_save_1)
          with png_writer.PngWriter("im3.png") as writer2:
            writer2.save(img_save_2)

def upsample_t_creation(img_path_list, file_num, sub_save_location):
    os.chdir(sub_save_location)
    t, z, y_dim,x_dim, img = load_img(img_path_list[file_num])
    folder_steps = str(file_num) + "_steps"
    folder_steps = os.path.join(sub_save_location,folder_steps)
    os.mkdir(folder_steps)

    # txt_name_log = open(destination + "/name_log.txt", "a")
    # txt_name_log.write("{}, {}\n".format(new_folder_name, img_path_list[file_num]), )
    # txt_name_log.close()

    images_jump =2
    #create new directory-path
    for z_num in tqdm(range(0,z)):
        for t_num in range(t-1):
          #create new directory-path
          file_folder = ("f_%03d" %(file_num)+"-" +"z_%03d"%(z_num) + "-"+"t_%03d" %(t_num))
          os.chdir(folder_steps)
          os.mkdir(file_folder)
          steps_path_folder = os.path.join(folder_steps, file_folder)
          os.chdir(steps_path_folder)

          # #here put the image pngs into the folder (instead of creating the folder)
          # #convert image to unit8 otherwise warning
          img_save_1 = reshape_data(img, "TZXY","XY", T=t_num, Z=z_num)
          img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
          img_save_1 = convert(img_save_1, 0, 255, np.uint8)

          img_save_2 = reshape_data(img, "TZXY","XY", T=t_num+1, Z=z_num)
          img_save_2 = create_3D_image(img_save_2, x_dim, y_dim)
          img_save_2 = convert(img_save_2, 0, 255, np.uint8)

          # # saving images as PNG
          with png_writer.PngWriter("im1.png") as writer1:
            writer1.save(img_save_1)
          with png_writer.PngWriter("im3.png") as writer2:
            writer2.save(img_save_2)

def perform_prep_predict_z_creation(img_path_list, file_num,  sub_save_location):
    os.chdir(sub_save_location)
    t, z, y_dim,x_dim, img = load_img(img_path_list[file_num])
    folder_gt = str(file_num) + "_gt"
    folder_gt = os.path.join(sub_save_location,folder_gt)
    os.mkdir(folder_gt)
    folder_steps = str(file_num) + "_steps"
    folder_steps = os.path.join(sub_save_location,folder_steps)
    os.mkdir(folder_steps)

    # txt_name_log = open(destination + "/name_log.txt", "a")
    # txt_name_log.write("{}, {}\n".format(new_folder_name, img_path_list[file_num]), )
    # txt_name_log.close()

    images_jump =2
   #create new directory-path
    for t_num in tqdm(range(0,t)):
        for z_num in range(math.ceil(z/images_jump)-1): # rounds up to then remove the last one to not overshoot in the counting
        #create new directory-path
          file_folder = ("f_%03d" %(file_num) + "-"+"t_%03d" %(t_num)+"-" +"z_%03d"%(z_num))
          os.chdir(folder_gt)
          os.mkdir(file_folder)
          os.chdir(folder_steps)
          os.mkdir(file_folder)
          GT_path_folder = os.path.join(folder_gt, file_folder)
          steps_path_folder = os.path.join(folder_steps, file_folder)
          os.chdir(steps_path_folder)

          #here put the image pngs into the folder (instead of creating the folder)
          #convert image to unit8 otherwise warning
          first = z_num* images_jump
          second = z_num*images_jump+images_jump
          img_save_1 = reshape_data(img, "TZXY","XY", T=t_num, Z=first)
          img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
          img_save_1 = convert(img_save_1, 0, 255, np.uint8)

          img_save_3 = reshape_data(img, "TZXY","XY", T=t_num, Z=second)
          img_save_3 = create_3D_image(img_save_3, x_dim, y_dim)
          img_save_3 = convert(img_save_3, 0, 255, np.uint8)

          img_save_2 = reshape_data(img, "TZXY","XY", T=t_num, Z=first+1)
          img_save_2 = create_3D_image(img_save_2, x_dim, y_dim)
          img_save_2 = convert(img_save_2, 0, 255, np.uint8)

          # saving images as PNG
          with png_writer.PngWriter("im1.png") as writer1:
            writer1.save(img_save_1)
          with png_writer.PngWriter("im3.png") as writer2:
            writer2.save(img_save_3)

          os.chdir(GT_path_folder)
          with png_writer.PngWriter("im2.png") as writer2:
            writer2.save(img_save_2)

def perform_prep_predict_t_creation(img_path_list, file_num,  sub_save_location):
    os.chdir(sub_save_location)
    t, z, y_dim,x_dim, img = load_img(img_path_list[file_num])
    folder_gt = str(file_num) + "_gt"
    folder_gt = os.path.join(sub_save_location,folder_gt)
    os.mkdir(folder_gt)
    folder_steps = str(file_num) + "_steps"
    folder_steps = os.path.join(sub_save_location,folder_steps)
    os.mkdir(folder_steps)

    # txt_name_log = open(destination + "/name_log.txt", "a")
    # txt_name_log.write("{}, {}\n".format(new_folder_name, img_path_list[file_num]), )
    # txt_name_log.close()

    images_jump =2
   #create new directory-path
    for z_num in tqdm(range(0,z)):
        for t_num in range(math.ceil(t/images_jump)-1):
        #create new directory-path
          file_folder = ("f_%03d" %(file_num)+"-" +"z_%03d"%(z_num) + "-"+"t_%03d" %(t_num))
          os.chdir(folder_gt)
          os.mkdir(file_folder)
          os.chdir(folder_steps)
          os.mkdir(file_folder)
          GT_path_folder = os.path.join(folder_gt, file_folder)
          steps_path_folder = os.path.join(folder_steps, file_folder)
          os.chdir(steps_path_folder)

          #here put the image pngs into the folder (instead of creating the folder)
          #convert image to unit8 otherwise warning
          first = t_num* images_jump
          second = t_num*images_jump+images_jump
          img_save_1 = reshape_data(img, "TZXY","XY", T=first, Z=z_num)
          img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
          img_save_1 = convert(img_save_1, 0, 255, np.uint8)

          img_save_3 = reshape_data(img, "TZXY","XY", T=second, Z=z_num)
          img_save_3 = create_3D_image(img_save_3, x_dim, y_dim)
          img_save_3 = convert(img_save_3, 0, 255, np.uint8)

          img_save_2 = reshape_data(img, "TZXY","XY", T=first+1, Z=z_num)
          img_save_2 = create_3D_image(img_save_2, x_dim, y_dim)
          img_save_2 = convert(img_save_2, 0, 255, np.uint8)

          # saving images as PNG
          with png_writer.PngWriter("im1.png") as writer1:
            writer1.save(img_save_1)
          with png_writer.PngWriter("im3.png") as writer2:
            writer2.save(img_save_3)

          os.chdir(GT_path_folder)
          with png_writer.PngWriter("im2.png") as writer2:
            writer2.save(img_save_2)

def perform_max_t_creation(img_path_list, file_num,  sub_save_location, split_training_test):
    os.chdir(sub_save_location)
    sub_folder = "sequences"
    sequence_path = os.path.join(sub_save_location, sub_folder)
    if not os.path.exists(sequence_path):
      os.mkdir(sequence_path)
      os.chdir(sequence_path)
    else:
      os.chdir(sequence_path)
    t, z, y_dim,x_dim, img = load_img(img_path_list[file_num])

    # txt_name_log = open(destination + "/name_log.txt", "a")
    # txt_name_log.write("{}, {}\n".format(new_folder_name, img_path_list[file_num]), )
    # txt_name_log.close()

    for z_num in tqdm(range(0,z)):
    #create new directory-path
      file_folder = "%02d" % (z_num+1+file_num*z)
      z_folder = os.path.join(sequence_path, file_folder)
      os.mkdir(z_folder)  
      os.chdir(z_folder)

      for t_num in range(0,t-2):
        slice_folder = "%04d" % (t_num+1)
        three_t_folder = os.path.join(z_folder, slice_folder)
        os.mkdir(three_t_folder)  
        os.chdir(three_t_folder)
        #add new folder to txt-file
        decision_train_test = random.random()
        if decision_train_test < split_training_test:
          txt_file_train = open(sub_save_location + "/tri_trainlist.txt", "a")
          txt_file_train.write("{}/{}\n".format(file_folder,slice_folder))
          txt_file_train.close()
        else:
          txt_file_test = open(sub_save_location + "/tri_testlist.txt", "a")
          txt_file_test.write("{}/{}\n".format(file_folder,slice_folder))
          txt_file_test.close()
          
        #converting images in rgb and uint8 to save it like that
        img_save_1 = reshape_data(img, "TZXY","XY", T=t_num, Z=z_num)
        img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
        img_save_1 = convert(img_save_1, 0, 255, np.uint8)
        img_save_2 = reshape_data(img, "TZXY","XY", T=t_num+1, Z=z_num)
        img_save_2 = create_3D_image(img_save_2, x_dim, y_dim)
        img_save_2 = convert(img_save_2, 0, 255, np.uint8)
        img_save_3 = reshape_data(img, "TZXY","XY", T=t_num+2, Z=z_num)
        img_save_3 = create_3D_image(img_save_3, x_dim, y_dim)
        img_save_3 = convert(img_save_3, 0, 255, np.uint8)
        with png_writer.PngWriter("im1.png") as writer2:
          writer2.save(img_save_1)
        with png_writer.PngWriter("im2.png") as writer2:
          writer2.save(img_save_2)
        with png_writer.PngWriter("im3.png") as writer2:
          writer2.save(img_save_3)
        print("{}/{}\n".format(file_folder,slice_folder))

def perform_max_z_creation(img_path_list, file_num,  sub_save_location, split_training_test):
    os.chdir(sub_save_location)
    sub_folder = "sequences"
    sequence_path = os.path.join(sub_save_location, sub_folder)
    if not os.path.exists(sequence_path):
      os.mkdir(sequence_path)
      os.chdir(sequence_path)
    else:
      os.chdir(sequence_path)
    t, z, y_dim,x_dim, img = load_img(img_path_list[file_num])

    # txt_name_log = open(destination + "/name_log.txt", "a")
    # txt_name_log.write("{}, {}\n".format(new_folder_name, img_path_list[file_num]), )
    # txt_name_log.close()

    for t_num in tqdm(range(0,t)):
    #create new directory-path
      file_folder = "%02d" % (t_num+1+file_num*t)
      t_folder = os.path.join(sequence_path, file_folder)
      os.mkdir(t_folder)  
      os.chdir(t_folder)

      for z_num in range(0,z-2):
        slice_folder = "%04d" % (z_num+1)
        three_z_folder = os.path.join(t_folder, slice_folder)
        os.mkdir(three_z_folder)  
        os.chdir(three_z_folder)
        #add new folder to txt-file
        decision_train_test = random.random()
        if decision_train_test < split_training_test:
          txt_file_train = open(sub_save_location + "/tri_trainlist.txt", "a")
          txt_file_train.write("{}/{}\n".format(file_folder,slice_folder))
          txt_file_train.close()
        else:
          txt_file_test = open(sub_save_location + "/tri_testlist.txt", "a")
          txt_file_test.write("{}/{}\n".format(file_folder,slice_folder))
          txt_file_test.close()
          
        #converting images in rgb and uint8 to save it like that
        img_save_1 = reshape_data(img, "TZXY","XY", T=t_num, Z=z_num)
        img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
        img_save_1 = convert(img_save_1, 0, 255, np.uint8)
        img_save_2 = reshape_data(img, "TZXY","XY", T=t_num, Z=z_num+1)
        img_save_2 = create_3D_image(img_save_2, x_dim, y_dim)
        img_save_2 = convert(img_save_2, 0, 255, np.uint8)
        img_save_3 = reshape_data(img, "TZXY","XY", T=t_num, Z=z_num+2)
        img_save_3 = create_3D_image(img_save_3, x_dim, y_dim)
        img_save_3 = convert(img_save_3, 0, 255, np.uint8)

        with png_writer.PngWriter("im1.png") as writer2:
          writer2.save(img_save_1)
        with png_writer.PngWriter("im2.png") as writer2:
          writer2.save(img_save_2)
        with png_writer.PngWriter("im3.png") as writer2:
          writer2.save(img_save_3)
        print("{}/{}\n".format(file_folder,slice_folder))

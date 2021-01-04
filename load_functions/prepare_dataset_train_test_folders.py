#WORKING
import os
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
    t, z, y_dim,x_dim, img = load_img(img_path_list[file_num])
    # folder_steps = str(file_num) + "_steps"
    img_nr = img_path_list[file_num].split("/")[-1].split(".")[0].split("-")[1][:3]
    fr_nr = img_path_list[file_num].split("/")[-1].split(".")[0].split("-")[2][:2]
    p_nr = img_path_list[file_num].split("/")[-1].split(".")[0].split("-")[3][:2]

    #create new directory-path
    for num_1 in tqdm(range(0,t)):
        folder_name = "i-{}_".format(img_nr) + "f-{}_".format(fr_nr) + "p-{}_".format(p_nr) + "t-%03d"%(num_1)
        os.chdir(sub_save_location)
        folder = os.path.join(sub_save_location,folder_name)
        os.mkdir(folder)
        os.chdir(folder)
        for num_2 in range(z):
          if (num_2 % 2) == 0:
            #create new directory-path
            file_name = ("z_%03d" %(num_2))

            # #here put the image pngs into the folder (instead of creating the folder)
            # #convert image to unit8 otherwise warning
            img_save_1 = reshape_data(img, "TZXY","XY", T=num_1, Z=num_2)
            img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
            img_save_1 = convert(img_save_1, 0, 255, np.uint8)
            # # saving images as PNG
            with png_writer.PngWriter("{}.png".format(file_name)) as writer1:
              writer1.save(img_save_1)

          #save the last slide on top labeled with x
          if num_2 == z-1 and (num_2 % 2) != 0:
            file_name = ("z_%03d" %(num_2))

            # #here put the image pngs into the folder (instead of creating the folder)
            # #convert image to unit8 otherwise warning
            img_save_1 = reshape_data(img, "TZXY","XY", T=num_1, Z=num_2)
            img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
            img_save_1 = convert(img_save_1, 0, 255, np.uint8)
            # # saving images as PNG
            with png_writer.PngWriter("{}-x.png".format(file_name)) as writer1:
              writer1.save(img_save_1)


def downsample_t_creation(img_path_list, file_num, sub_save_location):
    os.chdir(sub_save_location)
    t, z, y_dim,x_dim, img = load_img(img_path_list[file_num])
    # folder_steps = str(file_num) + "_steps"
    img_nr = img_path_list[file_num].split("/")[-1].split(".")[0].split("-")[1][:3]
    fr_nr = img_path_list[file_num].split("/")[-1].split(".")[0].split("-")[2][:2]
    p_nr = img_path_list[file_num].split("/")[-1].split(".")[0].split("-")[3][:2]

    #create new directory-path
    for num_2 in tqdm(range(0,z)):
        folder_name = "i-{}_".format(img_nr) + "f-{}_".format(fr_nr) + "p-{}_".format(p_nr) + "z-%03d"%(num_2)
        os.chdir(sub_save_location)
        folder = os.path.join(sub_save_location,folder_name)
        os.mkdir(folder)
        os.chdir(folder)
        for num_1 in range(t):
          if (num_1 % 2) == 0:
            #create new directory-path
            file_name = ("t_%03d" %(num_1))

            # #here put the image pngs into the folder (instead of creating the folder)
            # #convert image to unit8 otherwise warning
            img_save_1 = reshape_data(img, "TZXY","XY", T=num_1, Z=num_2)
            img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
            img_save_1 = convert(img_save_1, 0, 255, np.uint8)
            # # saving images as PNG
            with png_writer.PngWriter("{}.png".format(file_name)) as writer1:
              writer1.save(img_save_1)

          #save the last slide on top labeled with x
          if num_1 == t-1 and (num_1 % 2) != 0:
            file_name = ("t_%03d" %(num_1))

            # #here put the image pngs into the folder (instead of creating the folder)
            # #convert image to unit8 otherwise warning
            img_save_1 = reshape_data(img, "TZXY","XY", T=num_1, Z=num_2)
            img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
            img_save_1 = convert(img_save_1, 0, 255, np.uint8)
            # # saving images as PNG
            with png_writer.PngWriter("{}-x.png".format(file_name)) as writer1:
              writer1.save(img_save_1)

            
def upsample_t_creation(img_path_list, file_num, sub_save_location):
    os.chdir(sub_save_location)
    t, z, y_dim,x_dim, img = load_img(img_path_list[file_num])
    # folder_steps = str(file_num) + "_steps"
    img_nr = img_path_list[file_num].split("/")[-1].split(".")[0].split("-")[1][:3]
    fr_nr = img_path_list[file_num].split("/")[-1].split(".")[0].split("-")[2][:2]
    p_nr = img_path_list[file_num].split("/")[-1].split(".")[0].split("-")[3][:2]
    
    #create new directory-path
    for num_2 in tqdm(range(0,z)):   # dim_2 = zdimension
        folder_name = "i-{}_".format(img_nr) + "f-{}_".format(fr_nr) + "p-{}_".format(p_nr) + "z-%03d"%(num_2) # z doesn't need to be the z dimension because it is also used for the t dimension
        os.chdir(sub_save_location)
        folder = os.path.join(sub_save_location,folder_name)
        os.mkdir(folder_name)
        os.chdir(folder_name)
        for num_1 in range(t):
          #create new directory-path
          file_name = ("t_%03d" %(num_1))

          # #here put the image pngs into the folder (instead of creating the folder)
          # #convert image to unit8 otherwise warning
          img_save_1 = reshape_data(img, "TZXY","XY", T=num_1, Z=num_2)
          img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
          img_save_1 = convert(img_save_1, 0, 255, np.uint8)

          # # saving images as PNG
          with png_writer.PngWriter("{}.png".format(file_name)) as writer1:
            writer1.save(img_save_1)


def upsample_z_creation(img_path_list, file_num, sub_save_location):
    os.chdir(sub_save_location)
    t, z, y_dim,x_dim, img = load_img(img_path_list[file_num]) #dim_1=t, dim_2=z
    # folder_steps = str(file_num) + "_steps"
    img_nr = img_path_list[file_num].split("/")[-1].split(".")[0].split("-")[1][:3]
    fr_nr = img_path_list[file_num].split("/")[-1].split(".")[0].split("-")[2][:2]
    p_nr = img_path_list[file_num].split("/")[-1].split(".")[0].split("-")[3][:2]
    # folder_file_path = os.path.join(sub_save_location,file_to_folder_name)
    # os.mkdir(folder_file_path)

    #create new directory-path
    for num_1 in tqdm(range(0,t)):
        folder_name = "i-{}_".format(img_nr) + "f-{}_".format(fr_nr) + "p-{}_".format(p_nr) + "z-%03d"%(num_1)
        os.chdir(sub_save_location)
        folder = os.path.join(sub_save_location,folder_name)
        os.mkdir(folder_name)
        os.chdir(folder_name)
        for num_2 in range(z):
          #create new directory-path
          file_name = ("z_%03d"%(num_2))

          # #here put the image pngs into the folder (instead of creating the folder)
          # #convert image to unit8 otherwise warning
          img_save_1 = reshape_data(img, "TZXY","XY", T=num_1, Z=num_2)
          img_save_1 = create_3D_image(img_save_1, x_dim, y_dim)
          img_save_1 = convert(img_save_1, 0, 255, np.uint8)

          # # saving images as PNG
          with png_writer.PngWriter("{}.png".format(file_name)) as writer1:
            writer1.save(img_save_1)

            
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
    img = img.get_image_data("TZXY", S=0)  # in my case channel is the Time
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


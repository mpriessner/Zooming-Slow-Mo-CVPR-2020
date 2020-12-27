import os
from skimage import io
import glob 
import cv2
from tqdm import tqdm
import os
import shutil
from aicsimageio import AICSImage, imread
from aicsimageio.transforms import reshape_data
from aicsimageio.writers import png_writer 
import numpy as np
def get_file_list(folder_path):
  """This function takes a folder_path and returns a list of files sorted"""
  # get a list of files in the folder
  flist = os.listdir(folder_path)
  flist.sort()
  return flist

def get_folder_list(source_path):
  """ This function creates a list of folders from a given source path"""
  folder_list = [x[0] for x in os.walk(source_path)]
  folder_list.sort()
  folder_list = folder_list[1:]
  return folder_list

#def create_template_dict(source_path):
#  # get dimensions of data
#  folder_list = get_folder_list(source_path)
#  flist = get_file_list(folder_list[0])
#  file_path = os.path.join(folder_list[0], flist[0])
#  img = AICSImage(file_path)
#  img = img.get_image_data("YX", Z=0, C=0, S=0, T=0)
#  xy_dim = img.shape[-1]
#  folder_dim = len(folder_list)
#  file_dim = len(flist)
#  #create a dictionary which contains all 3d lists of zeros for storing all the images
#  master_dict = {}
#  for folder_path in folder_list:
#    folder_name = folder_path.split("/")[-1]
#    master_dict[folder_name] =np.zeros((file_dim,xy_dim,xy_dim))
#  return master_dict, file_dim, folder_dim, xy_dim

def save_image(temp_img, folder_option, slice_count, file_count, save_location_image, file_name):
  """ This function saves the temp image and re-structures the channels in the right order for the z-dimension"""
  temp_img_final = temp_img[1:,:,:,:]
  # if folder_option == "upsample-z":
  #   temp_img_final = np.swapaxes(temp_img_final, 0, 1)
  io.imsave("/content/temp.tif",temp_img_final)
  img = AICSImage("/content/temp.tif")
  img = img.get_image_data("CSTZYX")
  if folder_option == "upsample-z" or folder_option == "downsample-z":
    img= reshape_data(img, "CSTZYX","STCZYX")
  elif folder_option == "upsample-t" or folder_option == "downsample-t":
    img= reshape_data(img, "CSTZYX","SZTCYX")
  # io.imsave(f"/{save_location}/{name}.tif",img)
  io.imsave(save_location_image+f"/{file_name}_file_{file_count}_%02d.tif"%slice_count, img)
  

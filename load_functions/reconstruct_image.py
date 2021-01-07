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
import h5py


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


def save_image(temp_img, folder_option, slice_count, file_count, save_location_image, file_name):
  """ This function saves the temp image and re-structures the channels in the right order for the z-dimension"""
  
  temp_img_final = temp_img[1:,:,:,:]
  # if folder_option == "upsample-z":
  #   temp_img_final = np.swapaxes(temp_img_final, 0, 1)
  io.imsave("/content/temp.tif",temp_img_final)
  img = AICSImage("/content/temp.tif")
  img = img.get_image_data("CSTZYX")
  
  if folder_option == "upsample-z" or folder_option == "downsample-z":
    img= reshape_data(img, "CSTZYX","SCTZYX")
    io.imsave(save_location_image+f"/{file_name}_Z.tif", img)

  elif folder_option == "upsample-t" or folder_option == "downsample-t" or folder_option == "zoom":
    img= reshape_data(img, "CSTZYX","SZTCYX")
    io.imsave(save_location_image+f"/{file_name}_T.tif", img)
    

def save_as_h5py(img_list, permutation_list, fraction_list, zt_list, file_nr, interpolate_location, multiplyer, product_image_shape):
    '''this function saves the the single images of each 4D file into one h5py file'''
    zt_dim = len(zt_list)
    xy_dim = int(product_image_shape/multiplyer)
    h5py_safe_location_list = []
    # saving all the images in the xyz dimension in a h5py file
    
    for image in img_list:
      h5py_safe_location = f"/content/ZoomInterpolation/results/{image}.hdf5"
      h5py_safe_location_list.append(h5py_safe_location)
      with h5py.File(h5py_safe_location, 'w') as f:
        
        for permutation in permutation_list:
          for zt in tqdm(zt_list):
            temp_img_3D = np.zeros((len(file_nr), multiplyer*xy_dim, multiplyer*xy_dim))
            for single_file_nr, single_file in enumerate(file_nr):
              temp_img_2D = np.zeros((multiplyer*xy_dim, multiplyer*xy_dim))
              counter_x = 0
              counter_y = 0
              for num, fraction in enumerate(fraction_list):
                if counter_x == multiplyer:
                  counter_x = 0
                  counter_y+=1
                key = f"{image}_{fraction}_{permutation}_{zt}/{single_file}"
                # print(key)
                img_path = os.path.join(interpolate_location, key)
                img = AICSImage(img_path)
                img = img.get_image_data("YX", Z=0, C=0, S=0, T=0)
                img = img.astype('uint8')
                temp_img_2D[counter_x*xy_dim:(counter_x+1)*xy_dim,counter_y*xy_dim:(counter_y+1)*xy_dim] = img
                counter_x += 1
              temp_img_3D[single_file_nr,:,:] = temp_img_2D
            name = f"{image}_{permutation}_{zt}"
            f.create_dataset(f"{name}", data=np.array(temp_img_3D, dtype=np.uint8))

    return h5py_safe_location_list

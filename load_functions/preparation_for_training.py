import os
import sys
sys.path.insert(0,'/content/ZoomInterpolation/codes')
import cv2
import numpy as np
from tqdm import tqdm
from data.util import imresize_np


def prep_folder_structure(root, new_path):
  '''this function creates the same folder and subfolder structure as provided in the sequences folder in a 
  new given location path'''
  for path, subdirs, files in os.walk(root):
      for dir in subdirs:
        if len(dir)==5:
          new_path_sub = os.path.join(new_path, dir)
          os.mkdir(os.path.join(new_path, dir))
        else:
          os.mkdir(os.path.join(new_path_sub, dir))

def get_all_filepaths(folder_path):
    '''This function gets the paths from each file in folder and subfolder of a given location'''
    flist = []
    for path, subdirs, files in tqdm(os.walk(folder_path)):
          for name in files:
            flist.append(os.path.join(path, name))
    return flist

def generate_mod_LR_bic(up_scale, sourcedir, savedir):
    # params: upscale factor, input directory, output directory
    saveHRpath = os.path.join(savedir, 'HR', 'x' + str(up_scale))
    saveLRpath = os.path.join(savedir, 'LR', 'x' + str(up_scale))
    saveBicpath = os.path.join(savedir, 'Bic', 'x' + str(up_scale))
    print(sourcedir)
    print(not os.path.isdir(sourcedir))
    if not os.path.isdir(sourcedir):
        print('Error: No source data found')
        exit(0)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    if not os.path.isdir(os.path.join(savedir, 'HR')):
        os.mkdir(os.path.join(savedir, 'HR'))
    if not os.path.isdir(os.path.join(savedir, 'LR')):
        os.mkdir(os.path.join(savedir, 'LR'))
    if not os.path.isdir(os.path.join(savedir, 'Bic')):
        os.mkdir(os.path.join(savedir, 'Bic'))

    if not os.path.isdir(saveHRpath):
        os.mkdir(saveHRpath)
        prep_folder_structure(sourcedir, saveHRpath)
    else:
        print('It will cover ' + str(saveHRpath))
        prep_folder_structure(sourcedir, saveHRpath)

    if not os.path.isdir(saveLRpath):
        os.mkdir(saveLRpath)
        prep_folder_structure(sourcedir, saveLRpath)
    else:
        print('It will cover ' + str(saveLRpath))
        prep_folder_structure(sourcedir, saveLRpath)

    if not os.path.isdir(saveBicpath):
        os.mkdir(saveBicpath)
        prep_folder_structure(sourcedir, saveBicpath)
    else:
        print('It will cover ' + str(saveBicpath))
        prep_folder_structure(sourcedir, saveBicpath)

    filepaths = get_all_filepaths(sourcedir)
    print(len(filepaths))
    # filepaths = [f for f in os.listdir(sourcedir) if f.endswith('.png')]
    num_files = len(filepaths)

    # # prepare data with augementation
    for i in tqdm(range(num_files)):
        filename = filepaths[i]
        print('No.{} -- Processing {}'.format(i, filename))
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
        # bic
        image_Bic = imresize_np(image_LR, up_scale, True)
        file_folder_path = filename[-18:]
        cv2.imwrite(os.path.join(saveHRpath, file_folder_path), image_HR)
        cv2.imwrite(os.path.join(saveLRpath, file_folder_path), image_LR)
        cv2.imwrite(os.path.join(saveBicpath, file_folder_path), image_Bic)

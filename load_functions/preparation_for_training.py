import os
import os.path as osp
import sys
sys.path.insert(0,'/content/ZoomInterpolation/codes')
import cv2
import numpy as np
from tqdm import tqdm
from data.util import imresize_np
import shutil


def split_test_train_sequences_data(inPath, outPath, guide):
  """This function splits the sequences folder into the test and train folder with the given format
  based on the guide txt files"""
  if os.path.isdir(outPath):
    shutil.rmtree(outPath)
  f = open(guide, "r")
  lines = f.readlines()
  for l in tqdm(lines):
      line = l.replace('\n','')
      this_folder = os.path.join(inPath, line)
      dest_folder = os.path.join(outPath, line)
      # print(this_folder)
      shutil.move(this_folder, dest_folder)
  print('Done')

def prep_folder_structure(root, new_path):
  '''this function creates the same folder and subfolder structure as provided in the sequences folder in a 
  new given new_location path'''
  file_folder_list = []
  for path, subdirs, files in os.walk(root):
      for dir in subdirs:
        if len(dir)==5: # filter all file folders with 5 digits
          file_folder_list.append(dir)
  for file_folder in file_folder_list:
    file_folder_path = os.path.join(new_path, file_folder)   
    os.mkdir(file_folder_path)
    for path, subdirs, files in os.walk(root):
      for sub_folder in subdirs:
        if len(sub_folder)==4: # filter all sequence folders with 4 digits
          new_path_sub = os.path.join(file_folder_path, sub_folder)
          if not os.path.isdir(new_path_sub): # to not repeat the same folders several times
            new_path_sub = os.path.join(file_folder_path, sub_folder)
            os.mkdir(new_path_sub)

def get_all_filepaths(folder_path):
    '''This function gets the paths from each file in folder and subfolder of a given location'''
    flist = []
    for path, subdirs, files in tqdm(os.walk(folder_path)):
          for name in files:
            flist.append(os.path.join(path, name))
    return flist

def generate_mod_LR_bic(up_scale, sourcedir, savedir, train_guide, test_guide):
    # params: upscale factor, input directory, output directory
    saveHRpath = os.path.join(savedir, 'HR', 'x' + str(up_scale))
    saveLRpath = os.path.join(savedir, 'LR', 'x' + str(up_scale))
    saveBicpath = os.path.join(savedir, 'Bic', 'x' + str(up_scale))

    save_HR = os.path.join(savedir, 'HR')
    save_LR = os.path.join(savedir, 'LR')
    save_Bic = os.path.join(savedir, 'Bic')

    # print(sourcedir)
    # print(not os.path.isdir(sourcedir))
    if not os.path.isdir(sourcedir):
        print('Error: No source data found')
        exit(0)
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    if not os.path.isdir(save_HR):
        os.mkdir(save_HR)
    if not os.path.isdir(save_LR):
        os.mkdir(save_LR)
    if not os.path.isdir(save_Bic):
        os.mkdir(save_Bic)

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
      
    # copy the set_guide text files in each folder (HR, LR, Bic)
    train_guide_HR = saveHRpath[:-3]+"/sep_trainlist.txt"
    train_guide_LR = saveLRpath[:-3]+"/sep_trainlist.txt"
    train_guide_Bic = saveBicpath[:-3]+"/sep_trainlist.txt"

    test_guide_HR = saveHRpath[:-3]+"/sep_testlist.txt"
    test_guide_LR = saveLRpath[:-3]+"/sep_testlist.txt"
    test_guide_Bic = saveBicpath[:-3]+"/sep_testlist.txt"

    shutil.copy(train_guide, train_guide_HR)
    shutil.copy(train_guide, train_guide_LR)
    shutil.copy(train_guide, train_guide_Bic)

    shutil.copy(test_guide, test_guide_HR)
    shutil.copy(test_guide, test_guide_LR)
    shutil.copy(test_guide, test_guide_Bic)


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
    return save_HR, save_LR, save_Bic

#############################Prepare LMBD data ##################################
import os,sys
import os.path as osp
import glob
import pickle
from multiprocessing import Pool
import numpy as np
import lmdb
import cv2
from tqdm import tqdm
sys.path.insert(0,'/content/ZoomInterpolation/codes')
import data.util as data_util
import utils.util as util


def reading_image_worker(path, key):
    '''worker for reading images'''
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return (key, img)

def save_to_lmbd(img_folder, test_or_train, H_dst, W_dst, batch, mode, scale_factor):
    '''create lmdb for the Vimeo90K-7 frames dataset, each image with fixed size
    GT: [3, 256, 448]
        Only need the 4th frame currently, e.g., 00001_0001_4
    LR: [3, 64, 112]
        With 1st - 7th frames, e.g., 00001_0001_1, ..., 00001_0001_7
    key:
        Use the folder and subfolder names, w/o the frame index, e.g., 00001_0001
    '''
    #### configurations
    n_thread = 40

    # define the septest/trainlist & lmdb_save_path
    # path_parent = os.path.dirname(img_folder)

    if test_or_train == "test":
      txt_file = os.path.join(img_folder,"sep_testlist.txt")
      lmdb_save_path = os.path.join(img_folder, f"vimeo7_{test_or_train}_x{scale_factor}_{mode}.lmdb")
      img_folder_selected = os.path.join(img_folder, f"test_{scale_factor}")
      if os.path.isdir(lmdb_save_path):
        shutil.rmtree(lmdb_save_path)
    if test_or_train == "train":
      txt_file = os.path.join(img_folder,"sep_trainlist.txt")
      lmdb_save_path = os.path.join(img_folder, f"vimeo7_{test_or_train}_x{scale_factor}_{mode}.lmdb")
      img_folder_selected = os.path.join(img_folder, f"train_{scale_factor}")
      if os.path.isdir(lmdb_save_path):
        shutil.rmtree(lmdb_save_path)

    ########################################################
    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with \'lmdb\'.")
    #### whether the lmdb file exist
    if osp.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    #### read all the image paths to a list
    print('Reading image path list ...')
    with open(txt_file) as f:
        train_l = f.readlines()
        train_l = [v.strip() for v in train_l]
    all_img_list = []
    keys = []
    for line in tqdm(train_l):
        folder = line.split('/')[0]
        sub_folder = line.split('/')[1]
        file_l = glob.glob(osp.join(img_folder_selected, folder, sub_folder) + '/*')
        all_img_list.extend(file_l)
        for j in range(7):
            keys.append('{}_{}_{}'.format(folder, sub_folder, j + 1))
    all_img_list = sorted(all_img_list)
    keys = sorted(keys)
    if mode == 'HR': 
        all_img_list = [v for v in all_img_list if v.endswith('.png')]
        keys = [v for v in keys]

    print('Calculating the total size of images...')
    data_size = sum(os.stat(v).st_size for v in all_img_list)

    #### read all images to memory (multiprocessing)
    print('Read images with multiprocessing, #thread: {} ...'.format(n_thread))
    
    #### create lmdb environment
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
    txn = env.begin(write=True)  # txn is a Transaction object

    #### write data to lmdb
    pbar = util.ProgressBar(len(all_img_list))

    i = 0
    for path, key in tqdm(zip(all_img_list, keys)):
        pbar.update('Write {}'.format(key))
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        key_byte = key.encode('ascii')
        H, W, C = img.shape  # fixed shape
        assert H == H_dst and W == W_dst and C == 3, 'different shape.'
        txn.put(key_byte, img)
        i += 1
        if  i % batch == 1:
            txn.commit()
            txn = env.begin(write=True)

    txn.commit()
    env.close()
    print('Finish reading and writing {} images.'.format(len(all_img_list)))
            
    print('Finish writing lmdb.')

    #### create meta information
    meta_info = {}
    if mode == 'HR':
        meta_info['name'] = 'Vimeo7_train_GT'
    elif mode == 'LR':
        meta_info['name'] = 'Vimeo7_train_LR7'
    meta_info['resolution'] = '{}_{}_{}'.format(3, H_dst, W_dst)
    key_set = set()
    for key in keys:
        a, b, _ = key.split('_')
        key_set.add('{}_{}'.format(a, b))
    meta_info['keys'] = key_set
    pickle.dump(meta_info, open(osp.join(lmdb_save_path, 'Vimeo7_train_keys.pkl'), "wb"))
    print('Finish creating lmdb meta info.')

from PIL import Image
from Utils import utils 
from torch.utils.data import DataLoader
import random
import numpy as np
from torch.utils.data import Dataset


def pre_data(cfg_proj, cfg_m):

    img_x_train_names = utils.getFilesInPath(cfg_m.train_x_dir, suffix = "tiff", contain_t = None)
    img_y_train_names = utils.getFilesInPath(cfg_m.train_y_dir, suffix = "tiff", contain_t = None)
    if cfg_proj.isTrain:
        if cfg_m.train_ratio != 1:
            index_split = int(len(img_x_train_names)*cfg_m.train_ratio)
            training_data = CustomImageDataset(img_x_train_names[:index_split], img_y_train_names[:index_split], cfg_m)
            valid_data = CustomImageDataset(img_x_train_names[index_split:], img_y_train_names[index_split:], cfg_m)
            train_dataloader = DataLoader(training_data, cfg_m.batch_size)
            valid_dataloader = DataLoader(valid_data, cfg_m.batch_size)
        else:
            training_data = CustomImageDataset(img_x_train_names, img_y_train_names, cfg_m, cfg_m.flag_augment)
            train_dataloader = DataLoader(training_data, cfg_m.batch_size)
            valid_dataloader = None
    else:
        train_dataloader, valid_dataloader = None, None

    img_x_test_names = utils.getFilesInPath(cfg_m.test_x_dir, suffix = "tiff", contain_t = None)
    img_y_test_names = utils.getFilesInPath(cfg_m.test_y_dir, suffix = "tiff", contain_t = None)
    test_data = CustomImageDataset(img_x_test_names, img_y_test_names, cfg_m, isTrain = False)
    test_dataloader = DataLoader(test_data, 1)

    return train_dataloader, valid_dataloader, test_dataloader



class CustomImageDataset(Dataset):
    def __init__(self, img_x_names, img_y_names, cfg_m, flag_augment = False, isTrain = True):
        self.img_x_names = img_x_names
        self.img_y_names = img_y_names if len(img_y_names) > 0 else [None for i in range(len(img_x_names))]
        self.patch_size = cfg_m.patch_size
        self.batch_size = cfg_m.batch_size
        self.flag_augment = flag_augment
        self.isTrain = isTrain
        if self.isTrain:
            self.patch_x_all, self.patch_y_all = self.get_all_patches()

    def __len__(self):
        return len(self.patch_x_all) if self.isTrain else len(self.img_x_names) 

    def __getitem__(self, idx):
        if self.isTrain:
            In_X, In_Y = self.patch_x_all[idx], self.patch_y_all[idx]
        else:
            In_X, In_Y = self.get_img_patch(self.img_x_names[idx], self.img_y_names[idx])
        return In_X, In_Y


    def get_all_patches(self):
        patch_x_all, patch_y_all = [], []
        for i in range(len(self.img_x_names)):
            X, Y = self.get_img_patch(self.img_x_names[i], self.img_y_names[i])
            patch_x_all.append(X)
            patch_y_all.append(Y)
        patch_x_all = np.concatenate(patch_x_all, axis = 0)
        patch_y_all = np.concatenate(patch_y_all, axis = 0)
        return patch_x_all, patch_y_all

    def get_img_patch(self, img_x_name, img_y_name):
        img_x_patches, _ , _ = utils.crop2patch(img_x_name, self.patch_size, flag_is_x=True)
        if img_y_name is not None:          
            img_y_patches, _ , _ = utils.crop2patch(img_y_name, self.patch_size, flag_is_x=False)

        #get non empty patch index
        index_img_non_ept = []
        for i,img in enumerate(img_x_patches):
            if len(np.unique(img)) >1:
                index_img_non_ept.append(i)
        
        img_x_patches = [img_x_patches[i] for i in index_img_non_ept]
        if img_y_name is not None:
            img_y_patches = [img_y_patches[i] for i in index_img_non_ept]

        if img_y_name is None:
            img_y_patches = [-1 for i in range(len(img_x_patches))]

        if self.flag_augment:
            for id, (x, y) in enumerate(zip(img_x_patches, img_y_patches)):
                if random.random() < 0.5:
                    rot_angle = random.randrange(-20, 20)
                    x = Image.fromarray(x.reshape(x.shape[1], x.shape[2]))
                    x = np.array(x.rotate(rot_angle))
                    x = x[np.newaxis, :, :]
                    if img_y_name is not None:
                        y = Image.fromarray(y.reshape(y.shape[1], y.shape[2]))
                        y = np.array(y.rotate(rot_angle))
                        y = y[np.newaxis, :, :]
                img_x_patches[id] = x
                img_y_patches[id] = y

        return np.array(img_x_patches), np.array(img_y_patches)
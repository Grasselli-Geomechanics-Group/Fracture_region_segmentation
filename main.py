import os
import argparse
from Data.pre_data import pre_data
from Trainer.pre_trainer import pre_trainer
from Utils.utils import dotdict
from time import localtime, strftime
import numpy as np


def init_env():
    cfg_m = dotdict()
    cfg_m.insert("Note", None)
    #hyper-para - training
    cfg_m.insert("epochs", 10)
    cfg_m.insert("learning_rate_init", 1e-2)
    cfg_m.insert("learning_rate_min", 1e-3)
    cfg_m.insert("betas", [0.5, 0.999])
    cfg_m.insert("weight_decay", 2.0e-4)
    cfg_m.insert("log_interval", 25)
    #other paras
    cfg_m.insert("img_ch", 1)
    cfg_m.insert("output_ch", 3)
    cfg_m.insert("batch_size", 32)  #num of patches when isTrain
    cfg_m.insert("train_ratio", 1.0) #train:valid = train_ratio:1-train_ratio
    cfg_m.insert("patch_size", 64)   #patch-img dimension
    cfg_m.insert("weights_loss", [0.01, 0.1, 0.89])
    #dataset root path
    cfg_m.insert("flag_augment", True)  # train data augmentation
    cfg_m.insert("train_x_dir", './Data/train/input/')
    cfg_m.insert("train_y_dir", './Data/train/output/')
    cfg_m.insert("test_x_dir", './Data/test/input/')
    cfg_m.insert("test_y_dir", './Data/test/output/')
    return cfg_m

def main(cfg_proj, cfg_m):
    model = pre_trainer(cfg_proj, cfg_m)
    dataloader_train, dataloader_valid, dataloader_test = pre_data(cfg_proj, cfg_m)
    if cfg_proj.isTrain:
        model.train(dataloader_train, dataloader_valid)
    model.test(dataloader_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seg_Image")
    parser.add_argument("--gpu", type=str, default="0", required=False)
    parser.add_argument("--seed", type=int, default=np.random.randint(0, 1000000), required=False)
    #proj cfg
    parser.add_argument("--backbone", type=str, default="U_Net", choices = ["U_Net", "R2U_Net"],  required=False)
    parser.add_argument("--solver_alg", type=str, default="Segment_model", required=False)  #Segment_model
    parser.add_argument("--flag_time", type=str, default = strftime("%Y-%m-%d_%H-%M-%S", localtime()), required=False)
    parser.add_argument("--model_to_load", type=str, default = None, required=False)    # if is not None, then the file of loaded para need to contain the str
    parser.add_argument("--isTrain", default=False, action= 'store_true')

    cfg_proj = parser.parse_args()
    cfg_m = init_env()
    os.environ["CUDA_VISIBLE_DEVICES"] = "%s"%(cfg_proj.gpu)
    main(cfg_proj, cfg_m)












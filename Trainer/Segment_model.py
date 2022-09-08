from PIL import Image
import os
import numpy as np
from Nets.pre_net import pre_net
from Utils.sync_batchnorm import convert_model
import torch
from torch import nn
from Utils import utils 
from Trainer.base_trainer import Base_Trainer
from tqdm import tqdm
from skimage import io


class Segment_model(Base_Trainer):

    def __init__(self, cfg_proj, cfg_m):
        Base_Trainer.__init__(self, cfg_proj, cfg_m, name = "Segment_model")
        # self.cfg_m = cfg_m

    # def train(self, dataloader_train, dataloader_valid, stage = "seg_train", flad_load_ckp = True):
    def train(self, dataloader_train, dataloader_valid, stage = "seg_train"):    

        u_model = pre_net(self.cfg_proj, self.cfg_m).to(self.device)
        loss_fn = nn.CrossEntropyLoss(weight = torch.FloatTensor(self.cfg_m.weights_loss).to(self.device))
        optimizer = torch.optim.Adam(u_model.parameters(), lr= self.cfg_m.learning_rate_init, betas = self.cfg_m.betas, weight_decay = self.cfg_m.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = len(dataloader_train)*self.cfg_m.epochs, eta_min = self.cfg_m.learning_rate_min)

        if self.cfg_proj.model_to_load is not None:
            u_model, optimizer, lr_scheduler, epoch_start = self.load_ckp(u_model, optimizer, lr_scheduler, stage)
        else:
            epoch_start = 0

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            u_model = convert_model(u_model)
            u_model = nn.DataParallel(u_model)
            u_model = u_model.to(self.device)

        if epoch_start < self.cfg_m.epochs:
            pbar = tqdm(initial=epoch_start, total = self.cfg_m.epochs)
            pbar.set_description("%s_%s, epoch = %d, loss = None"%(self.name, stage, 0))
            for epoch in range(epoch_start, self.cfg_m.epochs, 1):
                loss_defense_trace = []
                for batch, (X, y) in enumerate(dataloader_train):
                    if not self.cfg_proj.isTrain: X, y = X.view(X.shape[0]*X.shape[1], *X.shape[2:]), y.view(y.shape[0]*y.shape[1], *y.shape[2:])
                    X, y = X.float(), y.type(torch.LongTensor)
                    X, y = X.to(self.device), y.to(self.device)
                    pred = u_model(X)
                    pred = torch.reshape(pred, (pred.shape[0], pred.shape[1], -1))
                    y = torch.reshape(y, (y.shape[0], -1))
                    loss = loss_fn(pred, y)
                    loss_defense_trace.append(loss.item())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                loss_epoch_avg = np.mean(loss_defense_trace)
                pbar.update(1)
                str_record = "%s_%s, epoch = %d, loss = %.3f"%(self.name, stage, epoch, loss_epoch_avg)
                pbar.set_description(str_record)
                self.logger.info(str_record)
                if (epoch+1) >= self.cfg_m.log_interval and (epoch+1) % self.cfg_m.log_interval == 0 or (epoch+1) == self.cfg_m.epochs:
                    self.save_ckp(u_model, optimizer, lr_scheduler, (epoch+1), stage)
                if dataloader_valid is not None: self.test(dataloader_valid, stage = "seg_valid")
            pbar.close()

    def test(self, dataloader, stage = "seg_test"):
        u_model = pre_net(self.cfg_proj, self.cfg_m).to(self.device)
        if self.cfg_proj.model_to_load is None: 
            if self.cfg_proj.isTrain == False:
                print("Error! please assign the --model_to_load to load the model if not trainining")
                return
            self.cfg_proj.model_to_load = self.cfg_proj.flag_time    #load current memory
        u_model, _, _, _ = self.load_ckp(u_model, None, None, None)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            u_model = convert_model(u_model)
            u_model = nn.DataParallel(u_model)
            u_model = u_model.to(self.device)  

        num_batches = len(dataloader)
        u_model.eval()
        correct = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.view(X.shape[0]*X.shape[1], *X.shape[2:]), y.view(y.shape[0]*y.shape[1], *y.shape[2:])
                X, y = X.float(), y.type(torch.LongTensor)
                X, y = X.to(self.device), y.to(self.device)
                pred = u_model(X)
                pred = torch.reshape(pred, (pred.shape[0], pred.shape[1], -1))
                y = torch.reshape(y, (y.shape[0], -1))
                #check correction of each pixel, then avg
                pred = torch.argmax(pred, dim = 1)
                cor_m = torch.abs(pred - y)
                cor_m = (cor_m.cpu().numpy()).flatten()
                cor_m = len(np.where(cor_m == 0)[0])/len(cor_m)
                correct = correct + cor_m
        correct /= num_batches
        str_record = "%s_%s, Acc = %.2f%%" % (self.name, stage, correct*100)
        self.logger.info(str_record)
        print(str_record)
        self.output_full_img(u_model, dataloader, stage)


    def output_full_img(self, u_model, dataloader, stage):
        os.makedirs(os.path.join(self.log_sub_folder, stage), exist_ok=True)
        for img_name in dataloader.dataset.img_x_names:
            image_patches, imgSize, paddedShape = utils.crop2patch(img_name, self.cfg_m.patch_size, flag_is_x = True)
            num_patches_x, num_patches_y = int(paddedShape[1]/self.cfg_m.patch_size), int(paddedShape[0]/self.cfg_m.patch_size)
            image_patches = np.array(image_patches)
            
            u_model.eval()
            with torch.no_grad():
                img = torch.from_numpy(image_patches)
                img = img.to(self.device)
                img = img.float()
                pred_imgs_probability = u_model(img)
            pred_imgs = torch.argmax(pred_imgs_probability, dim = 1)
            pred_imgs = pred_imgs.cpu().numpy()
            
            #recover img from sliced imgs
            img_recover = np.zeros((num_patches_y*self.cfg_m.patch_size, num_patches_x*self.cfg_m.patch_size))
            for j in range(num_patches_y):
                for i in range(num_patches_x):
                    img_s = pred_imgs[num_patches_x*j + i]
                    img_s = np.reshape(img_s, (self.cfg_m.patch_size, self.cfg_m.patch_size))
                    img_recover[j*self.cfg_m.patch_size:(j+1)*self.cfg_m.patch_size, i*self.cfg_m.patch_size:(i+1)*self.cfg_m.patch_size] = img_s
            img_output = img_recover[:imgSize[1], :imgSize[0]]

            f_name_r = os.path.join(self.log_sub_folder, stage ,"r_%s"%os.path.basename(img_name))
            img_output = np.uint8(img_output)
            img_output[img_output==1]=128
            img_output[img_output==2]=255
            io.imsave(f_name_r, img_output)




import random
import torch
from PIL import Image
from glob import glob
import pandas as pd
import os


class Data_load(torch.utils.data.Dataset):
    def __init__(self, status,img_root,coarse_img_root, mask_root, img_transform, mask_transform,FERlabel=None):
        super(Data_load, self).__init__()
        self.status = status
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.FERlabel = FERlabel
        self.coarse = coarse_img_root
        self.paths = glob('{:s}/*'.format(img_root),recursive=True)
        self.paths2 = glob('{:s}/*'.format(self.coarse),recursive=True)
        self.mask_paths = glob('{:s}/*'.format(mask_root))

        self.N_mask = len(self.mask_paths)
        
        if self.status == "train":
            df = pd.read_csv(FERlabel, sep=' ', header=None,names=['name','label'])
            self.data = df[df['name'].str.startswith('train')]
        
            file_names = self.data.loc[:, 'name'].values
            self.label = self.data.loc[:, 'label'].values - 1

            self.file_paths = []
            self.file_paths2 = []
            for f in file_names:
                f = f.split(".")[0]
                f = f +"_aligned.jpg"
                path = os.path.join(img_root, f)
                path2 = os.path.join(coarse_img_root,f)
                self.file_paths.append(path)
                self.file_paths2.append(path2)

    def __getitem__(self, index):
        if self.status == "train":
            path = self.file_paths[index]
            path2 = self.file_paths2[index]
            gt_img = Image.open(path)
            coarse_img = Image.open(path2)
            img_name = self.file_paths[index].split('\\')[-1]
            label = self.label[index]
        else :
            gt_img = Image.open(self.paths[index])
            coarse_img = Image.open(self.paths2[index])
            img_name = self.paths[index].split('\\')[-1]
        
        gt_img = self.img_transform(gt_img.convert('RGB'))
        coarse_img = self.img_transform(coarse_img.convert('RGB'))
        
        #mask = Image.open(self.mask_paths[random.randint(0, self.N_mask - 1)])
        mask = Image.open(self.mask_paths[index%(self.N_mask)])

        mask = self.mask_transform(mask.convert('RGB'))

        if self.status == "train":
            return gt_img,coarse_img,mask,img_name,label
        else:
            return gt_img,coarse_img,mask,img_name

    def __len__(self):
        return len(self.paths)

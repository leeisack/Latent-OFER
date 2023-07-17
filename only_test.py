import os
from random import sample
import warnings
from tqdm import tqdm
import argparse

from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from sklearn.metrics import balanced_accuracy_score
from networks.OURS import OURS

def warn(*args, **kwargs):
    pass
warnings.warn = warn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='datasets/', help='Raf-DB dataset path.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=60, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=4, help='Number of attention head.')

    return parser.parse_args()


class DataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform = None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/label.txt'), sep=' ', header=None,names=['name','label']) #'datasets/AffectNet/Affect_test.txt' 'datasets/FED_RO_Label_with_test.txt'

        if phase == 'train':
            self.data = df[df['name'].str.startswith('train')]
        else:
            self.data = df[df['name'].str.startswith('test')]

        file_names = self.data.loc[:, 'name'].values
        self.label = self.data.loc[:, 'label'].values - 1 
        _, self.sample_counts = np.unique(self.label, return_counts=True)

        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0]
            f = f +".JPG"
            path = os.path.join(self.raf_path,'Images/', f) 
            self.file_paths.append(path)

        self.pt_paths = []
        for f in file_names:
            f = f.split(".")[0]
            f = f +".pt"
            path = os.path.join(self.raf_path, 'PT_files/', f)
            self.pt_paths.append(path)


        print('path : ', path)
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.label[idx]


        pt_path = self.pt_paths[idx]
        latent = torch.load(pt_path)

        box = []
        b = [46,47,48,49,50,51,52,53,60,61,62,63,64,65,66,67,74,75,76,77,78,79,80,81,88,89,90,91,92,93,94,95,102,103,104,105,106,107,108,109,116,117,118,119,120,121,122,123,130,131,132,133,134,135,136,137,144,145,146,147,148,149,150,151,158,159,160,161,162,163,164,165,172,173,174,175,176,177]
        for i in b:
            box.append(latent[0][i])

        box = torch.stack(box, dim=0)
        latent = box.flatten()
        latent = latent.flatten()
        latent = latent.clone().detach()

        if self.transform is not None:
            image = self.transform(image)
        
        return image, latent, label

def run_validation():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = OURS(num_head=args.num_head)
    checkpoint = torch.load('./checkpoints/KDEF_epoch22_acc0.89.pth')
    model.load_state_dict(checkpoint['model_state_dict'],strict=False)
    model.to(device)


    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])   

    val_dataset = DataSet(args.raf_path, phase = 'test', transform = data_transforms_val)   

    print('Validation set size:', val_dataset.__len__())
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)

    criterion_cls = torch.nn.CrossEntropyLoss()


    params = list(model.parameters())
    optimizer = torch.optim.SGD(params,lr=args.lr, weight_decay = 1e-4, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0
    e = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
    with torch.no_grad():
        running_loss = 0.0
        iter_cnt = 0
        bingo_cnt = 0
        sample_cnt = 0
        baccs = []
        y_pred, y_true = [],[]
        count = 0
        model.eval()
        for (imgs, latent, targets) in val_loader:
            imgs = imgs.to(device)
            latent = latent.to(device)
            targets = targets.to(device)
    
            out,feat,heads = model(imgs, latent)
            loss = criterion_cls(out,targets)

            running_loss += loss
            iter_cnt+=1
            a, predicts = torch.max(out, 1)
            correct_num  = torch.eq(predicts,targets)

            bingo_cnt += correct_num.sum().cpu()
            sample_cnt += out.size(0)
            y_pred.append(predicts)
            y_true.append(targets)            
            
            baccs.append(balanced_accuracy_score(targets.cpu().numpy(),predicts.cpu().numpy()))
        running_loss = running_loss/iter_cnt   
        scheduler.step()

        acc = bingo_cnt.float()/float(sample_cnt)
        acc = np.around(acc.numpy(),4)

        best_acc = max(acc,best_acc)

        bacc = np.around(np.mean(baccs),4)
        tqdm.write("Validation accuracy:%.4f. bacc:%.4f. Loss:%.3f" % (acc, bacc, running_loss))
        tqdm.write("best_acc:" + str(best_acc))

        
if __name__ == "__main__":        
    run_validation()
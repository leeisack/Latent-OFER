import os
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

        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/label.txt'), sep=' ', header=None,names=['name','label'])
        
        if phase == 'train':
            self.data = df[df['name'].str.startswith('train')]
        else:
            self.data = df[df['name'].str.startswith('test')]

        file_names = self.data.loc[:, 'name'].values
        self.label = self.data.loc[:, 'label'].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        _, self.sample_counts = np.unique(self.label, return_counts=True)

        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0]
            f = f +".JPG"
            path = os.path.join(self.raf_path, 'Images/', f)
            self.file_paths.append(path)

        self.pt_paths = []
        for f in file_names:
            f = f.split(".")[0]
            f = f +".pt"
            path = os.path.join(self.raf_path, 'PT_files/', f)
            self.pt_paths.append(path)

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

def run_training():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model = OURS(num_head=args.num_head)
    model.to(device)

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
                transforms.RandomRotation(20),
                transforms.RandomCrop(224, padding=32)
            ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02,0.25)),
        ])
    
    train_dataset = DataSet(args.raf_path, phase = 'train', transform = data_transforms)    
    
    print('Whole train set size:', train_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,  
                                               pin_memory = True,
    )
    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])   

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
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()

        for (imgs, latent, targets) in train_loader:
            iter_cnt += 1
            optimizer.zero_grad()

            imgs = imgs.to(device)
            targets = targets.to(device)
            latent = latent.to(device)
            out,feat,heads = model(imgs, latent)

            loss = criterion_cls(out,targets)

            loss.backward()
            optimizer.step()
            
            running_loss += loss
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss/iter_cnt
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (epoch, acc, running_loss,optimizer.param_groups[0]['lr']))
        
        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            baccs = []

            y_true = []
            y_pred = []

            model.eval()
            for (imgs, latent, targets) in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                latent = latent.to(device)
                
                out,feat,heads = model(imgs, latent)
                loss = criterion_cls(out,targets)
                running_loss += loss
                iter_cnt+=1
                _, predicts = torch.max(out, 1)
                correct_num  = torch.eq(predicts,targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)
                y_true.append(targets.cpu().numpy())
                y_pred.append(predicts.cpu().numpy())
                
                baccs.append(balanced_accuracy_score(targets.cpu().numpy(),predicts.cpu().numpy()))
            running_loss = running_loss/iter_cnt   
            scheduler.step()

            acc = bingo_cnt.float()/float(sample_cnt)
            acc = np.around(acc.numpy(),4)
            best_acc = max(acc,best_acc)

            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)

            bacc = np.around(np.mean(baccs),4)
            tqdm.write("[Epoch %d] Validation accuracy:%.4f. bacc:%.4f. Loss:%.3f" % (epoch, acc, bacc, running_loss))
            tqdm.write("best_acc:" + str(best_acc))

            if acc > 0.86 and acc == best_acc:
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),},
                            os.path.join('checkpoints', "rafdb_epoch"+str(epoch)+"_acc"+str(acc)+"_bacc"+str(bacc)+".pth"))
                tqdm.write('Model saved.')

        
if __name__ == "__main__":        
    run_training()
from torch import nn
import torch
import torch.nn.init as init
from torch.nn import functional as F
from torchvision import models
import numpy as np
np.set_printoptions(precision=3)
np.set_printoptions(threshold=np.inf)

class FER(nn.Module):
    def __init__(self, num_class=7,num_head=4, pretrained=True):
        super(FER, self).__init__()
        
        resnet = models.resnet18(pretrained)
        
        if pretrained:
            checkpoint = torch.load('./models/fer_model/resnet18_msceleb.pth')
            resnet.load_state_dict(checkpoint['state_dict'],strict=True)

        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.num_head = num_head
        for i in range(num_head):
            setattr(self,"cat_head%d" %i, CrossAttentionHead())
        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)

    def forward(self, x):
        x = self.features(x)

        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self,"cat_head%d" %i)(x))
        heads = torch.stack(heads).permute([1,0,2])
        if heads.size(1)>1:
            heads = F.log_softmax(heads,dim=1)


        out = self.fc(heads.sum(dim=1))
        out = self.bn(out)
        out = F.sigmoid(out)
        return out, x, heads

class CrossAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention()
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)         
    def forward(self, x):
        ca = self.ca(x)
        sa = self.sa(ca)

        return sa


class SpatialAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1,3),padding=(0,1)),
            nn.BatchNorm2d(512),
        )
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,1),padding=(1,0)),
            nn.BatchNorm2d(512),
        )
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        y = self.conv1x1(x)
        a = self.conv_3x3(y)
        b = self.conv_1x3(y)
        c = self.conv_3x1(y)

        y = self.relu(a + b + c)
        y = y.sum(dim=1,keepdim=True) 

        out = x*y
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        
        return out 

class ChannelAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 512),
            nn.Sigmoid()    
        )


    def forward(self, sa):
        sa2 = self.gap(sa)
        sa2 = sa2.view(sa2.size(0),-1)
        y = self.attention(sa2)
        y = y.unsqueeze(dim = -1)
        y = y.unsqueeze(dim = -1)

        out = sa * y
        return out
#-*-coding:utf-8-*-
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import functools
import torch.nn.functional as F
from torch.optim import lr_scheduler

from .hModel import Hybrid_model
from .InnerCos import InnerCos
from .InnerCos2 import InnerCos2


###############################################################################
# Functions
###############################################################################

def get_norm_layer(norm_type='instance'):
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, which_model_netG, opt, mask_global, norm='instance', use_dropout=False, init_type='normal', gpu_ids=[], init_gain=0.02):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    cosis_list = []
    cosis_list2 = []
    assembly_model = []
    if which_model_netG == 'unet_assemble':
        netG = UnetGeneratorHybrid(input_nc, output_nc, 8, opt, mask_global, assembly_model,cosis_list, cosis_list2,ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)

    return init_net(netG, init_type, init_gain, gpu_ids),cosis_list ,cosis_list2,assembly_model


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[], init_gain=0.02):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'feature':
        netD = PFDiscriminator()
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
              which_model_netD)
    
    return init_net(netD, init_type, init_gain, gpu_ids)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


class GANLoss(nn.Module):
    def __init__(self, gan_type='wgan_gp', target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if gan_type == 'wgan_gp':
            self.loss = nn.MSELoss()
        elif gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_type == 'vanilla':
            self.loss = nn.BCELoss()
        else:
            raise ValueError("GAN type [%s] not recognized." % gan_type)


    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, y_pred_fake,y_pred, target_is_real):
        target_tensor = self.get_target_tensor(y_pred_fake, target_is_real)
        if(target_is_real):
            errD = (torch.mean((y_pred - torch.mean(y_pred_fake) - target_tensor) ** 2) + torch.mean(
                (y_pred_fake - torch.mean(y_pred) + target_tensor) ** 2)) / 2
            return errD


        else:
            errG = (torch.mean((y_pred - torch.mean(y_pred_fake) + target_tensor) ** 2) + torch.mean(
                (y_pred_fake - torch.mean(y_pred) - target_tensor) ** 2)) / 2
            return errG



class UnetGeneratorHybrid(nn.Module):
    def __init__(self, input_nc, output_nc,  num_downs, opt, mask_global, assembly_model,cosis_list ,cosis_list2,ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGeneratorHybrid, self).__init__()

        unet_block = UnetSkipConnectionBlock_3(ngf * 8, ngf * 8,  ngf * 8, ngf * 8, submodule=None, norm_layer=norm_layer, innermost=True)

        for i in range(num_downs - 5):  
            unet_block = UnetSkipConnectionBlock_3(ngf*8, ngf*8, ngf*8, ngf*8, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock_3(ngf*8, ngf*8, ngf*8, ngf*8,submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
                    
        unet_assemble = Self_assembly(ngf * 4, ngf * 8,ngf*8,ngf*16, opt, assembly_model,cosis_list,cosis_list2,mask_global,submodule=unet_block, norm_layer=norm_layer)

        unet_block = UnetSkipConnectionBlock_3_outer(ngf*2,   ngf*4,  ngf*4,      ngf*8,  submodule=unet_assemble, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock_3_outer(ngf,     ngf*2,  ngf*2,      ngf*4,  submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock_3_outer(output_nc, ngf,  input_nc*2, ngf*2,  submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        syn = input[0,0:3]
        mask = input[0,3:6]
        syn=torch.flip(syn,[0,2])
        mask=torch.flip(mask,[0,2])
        input_flip=torch.cat((syn,mask),0)
        input_flip=input_flip.unsqueeze(0)
        input = torch.cat((input,input_flip),1)
        return self.model(input)

  
class UnetSkipConnectionBlock_3_outer(nn.Module):
    def __init__(self, outer_nc, inner_nc,input_nc,inner_flip,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock_3_outer, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv_3 = nn.Conv2d(input_nc, inner_flip, kernel_size=3,
                             stride=1, padding=1)
        downrelu_3 = nn.LeakyReLU(0.2, True)
        downnorm_3 = norm_layer(inner_flip, affine=True)
        uprelu_3 = nn.ReLU(True)
        upnorm_3 = norm_layer(outer_nc, affine=True)

        downconv = nn.Conv2d(input_nc, input_nc, kernel_size=4,
                             stride=2, padding=3, dilation=2)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(input_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        if outermost:
            upconv_3 = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=1,
                                        padding=1)
            down = [downconv_3]
            up = [uprelu, upconv_3]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]  
            up = [uprelu, upconv, upnorm]
            model = down + up

        else:
            upconv = nn.ConvTranspose2d(outer_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            upconv_3 = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=1,
                                        padding=1)
            down = [downrelu, downconv, downnorm,downrelu_3,downconv_3,downnorm_3]
            up = [uprelu_3,upconv_3,upnorm_3,uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost: 
            return self.model(x)
        else:
            x_latter = self.model(x)
            origin_cn = x.size(1)//2
            x = x[0,:origin_cn].unsqueeze(0)
            _, _, h, w = x.size()
            if h != x_latter.size(2) or w != x_latter.size(3):
                x_latter = F.upsample(x_latter, (h, w), mode='bilinear')
            return torch.cat([x_latter, x], 1)  

class UnetSkipConnectionBlock_3(nn.Module):
    def __init__(self, outer_nc, inner_nc,input_nc,inner_flip,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock_3, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv_3 = nn.Conv2d(input_nc, inner_flip, kernel_size=3,
                             stride=1, padding=1)
        downrelu_3 = nn.LeakyReLU(0.2, True)
        downnorm_3 = norm_layer(inner_flip, affine=True)
        uprelu_3 = nn.ReLU(True)
        upnorm_3 = norm_layer(outer_nc, affine=True)

        downconv = nn.Conv2d(input_nc, input_nc, kernel_size=4,
                             stride=2, padding=3, dilation=2)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(input_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        if outermost:
            upconv_3 = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=1,
                                        padding=1)
            down = [downconv_3]
            up = [uprelu, upconv_3]
            model = down + [submodule] + up

        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv] 
            up = [uprelu, upconv, upnorm]
            model = down + up

        else:
            upconv = nn.ConvTranspose2d(outer_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            upconv_3 = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=1,
                                        padding=1)
            down = [downrelu, downconv, downnorm,downrelu_3,downconv_3,downnorm_3]
            up = [uprelu_3,upconv_3,upnorm_3,uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:  
            return self.model(x)
        else:
            x_latter = self.model(x)
            _, _, h, w = x.size()
            if h != x_latter.size(2) or w != x_latter.size(3):
                x_latter = F.upsample(x_latter, (h, w), mode='bilinear')
            return torch.cat([x_latter, x], 1) 

class Self_assembly(nn.Module):
    def __init__(self, outer_nc, inner_nc,input_nc,inner_flip, opt,assembly_model,cosis_list, cosis_list2,mask_global,submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,use_dropout=False):
        super(Self_assembly, self).__init__()
        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc

        downconv_3 = nn.Conv2d(input_nc, inner_flip, kernel_size=3,
                             stride=1, padding=1)
        downrelu_3 = nn.LeakyReLU(0.2, True)
        downnorm_3 = norm_layer(inner_nc, affine=True)
        uprelu_3 = nn.ReLU(True)
        upnorm_3 = norm_layer(outer_nc, affine=True)

        downconv = nn.Conv2d(input_nc, input_nc, kernel_size=4,
                             stride=2, padding=3, dilation=2)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(input_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        self_assembly= Hybrid_model(opt.threshold, opt.fixed_mask, opt.shift_sz, opt.stride, opt.mask_thred, opt.triple_weight)
        self_assembly.set_mask(mask_global, 3, opt.threshold)
        assembly_model.append(self_assembly)

        innerCos = InnerCos(strength=opt.strength, skip=opt.skip)
        innerCos.set_mask(mask_global, opt) 
        cosis_list.append(innerCos)

        innerCos2 = InnerCos2(strength=opt.strength, skip=opt.skip)
        innerCos2.set_mask(mask_global, opt)  
        cosis_list2.append(innerCos2)

        if outermost:

            upconv_3 = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=1,
                                        padding=1)
            down = [downconv_3]
            up = [uprelu, upconv_3]
            model = down + [submodule] + up

        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv] 
            up = [uprelu, upconv, upnorm]
            model = down + up

        else:
            upconv = nn.ConvTranspose2d(outer_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            upconv_3 = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=3, stride=1,
                                        padding=1)
            down = [downrelu, downconv, downnorm,downrelu_3,downconv_3,self_assembly,innerCos,downnorm_3]
            up = [innerCos2,uprelu_3,upconv_3,upnorm_3,uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            x_latter = self.model(x)
            x = x[0,:256].unsqueeze(0)
            _, _, h, w = x.size()
            if h != x_latter.size(2) or w != x_latter.size(3):
                x_latter = F.upsample(x_latter, (h, w), mode='bilinear')
            return torch.cat([x_latter, x], 1)  

################################### D ###################################
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
class PFDiscriminator(nn.Module):
    def __init__(self):

       super(PFDiscriminator, self).__init__()


       self.model=nn.Sequential(
           nn.Conv2d(256, 512,kernel_size=4, stride=2,padding=1),
           nn.LeakyReLU(0.2, True),
           nn.Conv2d(512, 512,kernel_size=4, stride=2,padding=1),
           nn.InstanceNorm2d(512),
           nn.LeakyReLU(0.2, True),
           nn.Conv2d(512, 512,kernel_size=4, stride=2,padding=1)

       )

    def forward(self, input):
        return self.model(input)
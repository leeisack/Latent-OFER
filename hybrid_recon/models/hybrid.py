#-*-coding:utf-8-*-

import torch
from collections import OrderedDict
from torch.autograd import Variable

import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
from .vgg16 import Vgg16

class SelfAssembly(BaseModel):
    def name(self):
        return 'SelfAssembly'

    def initialize(self, opt,model_fer=None):
        BaseModel.initialize(self, opt)
        self.device = torch.device('cuda')
        self.opt = opt
        self.isTrain = opt.isTrain
        self.coarse_exist = 0
        self.FER = model_fer
        self.label_gt = -1
        self.label_pred = -1
        self.vgg=Vgg16(requires_grad=False)
        self.vgg=self.vgg.cuda()
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_coarse = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)                           

        self.mask_global = torch.ByteTensor(1, 1, opt.fineSize, opt.fineSize)

        self.mask_global.zero_()
        self.mask_global[:, :, int(self.opt.fineSize/4) + self.opt.overlap : int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap,\
                                int(self.opt.fineSize/4) + self.opt.overlap: int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap] = 1

        self.mask_type = opt.mask_type
        self.gMask_opts = {}

        if len(opt.gpu_ids) > 0:
            self.use_gpu = True
            self.mask_global = self.mask_global.cuda()

        self.netG,self.Cosis_list,self.Cosis_list2, self.h_model= networks.define_G(opt.input_nc_g, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt, self.mask_global, opt.norm, opt.use_dropout, opt.init_type, self.gpu_ids, opt.init_gain)

        if self.isTrain:
            use_sigmoid = False
            if opt.gan_type == 'vanilla':
                use_sigmoid = True 

            self.netD = networks.define_D(opt.input_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, opt.init_gain)
            self.netF = networks.define_D(opt.input_nc, opt.ndf,
                                          opt.which_model_netF,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids,
                                          opt.init_gain)            
        if not self.isTrain or opt.continue_train:
            print('Loading pre-trained network!')
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)
                self.load_network(self.netF, 'F', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(gan_type=opt.gan_type, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionSC = torch.nn.CrossEntropyLoss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(),
                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_F)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG)
            if self.isTrain:
               networks.print_network(self.netD)
               networks.print_network(self.netF)
            print('-----------------------------------------------')

    def set_input(self,input,input2,mask,label=None):

        input_A = input
        input_B = input.clone()
        input_coarse = input2
        
        input_mask=mask
        self.label_gt = label

        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.input_coarse.resize_(input_coarse.size()).copy_(input_coarse)

        self.image_paths = 0

        if self.opt.mask_type == 'random':
            self.mask_global.zero_()
            self.mask_global=input_mask
        else:
            raise ValueError("Mask_type [%s] not recognized." % self.opt.mask_type)

        self.ex_mask = self.mask_global.expand(1, 3, self.mask_global.size(2), self.mask_global.size(3))

        self.inv_ex_mask = torch.add(torch.neg(self.ex_mask.float()), 1).byte()
        self.input_A.narrow(1,0,1).masked_fill_(self.mask_global, 2*123.0/255.0 - 1.0)
        self.input_A.narrow(1,1,1).masked_fill_(self.mask_global, 2*104.0/255.0 - 1.0)
        self.input_A.narrow(1,2,1).masked_fill_(self.mask_global, 2*117.0/255.0 - 1.0)

        self.set_latent_mask(self.mask_global, 3, self.opt.threshold)
        
    def set_latent_mask(self, mask_global, layer_to_last, threshold):
        self.h_model[0].set_mask(mask_global, layer_to_last, threshold)
        self.Cosis_list[0].set_mask(mask_global, self.opt)
        self.Cosis_list2[0].set_mask(mask_global, self.opt)
        
    def forward(self):
        self.real_A =self.input_A.to(self.device)
        self.fake_P= self.input_coarse.to(self.device) 
        self.un=self.fake_P.clone()
        self.Unknowregion=self.un.data.masked_fill_(self.inv_ex_mask, 0)
        self.knownregion=self.real_A.data.masked_fill_(self.ex_mask, 0)
        self.Syn=self.Unknowregion+self.knownregion
        self.Middle=torch.cat((self.Syn,self.input_A),1)
        self.fake_B = self.netG(self.Middle)
        self.FER.eval()
        self.label_pred = self.FER(self.fake_B.data)
        self.real_B = self.input_B.to(self.device)

    def set_gt_latent(self):
        gt_latent=self.vgg(Variable(self.input_B,requires_grad=False))
        self.Cosis_list[0].set_target(gt_latent.relu4_3)
        self.Cosis_list2[0].set_target(gt_latent.relu4_3)


    def test(self):
        self.real_A = self.input_A.to(self.device)
        self.fake_P = self.input_coarse.to(self.device) #
        self.un=self.fake_P.clone()
        self.Unknowregion=self.un.data.masked_fill_(self.inv_ex_mask, 0)
        self.knownregion=self.real_A.data.masked_fill_(self.ex_mask, 0)
        self.Syn=self.Unknowregion+self.knownregion
        self.Middle=torch.cat((self.Syn,self.input_A),1)
        self.fake_B = self.netG(self.Middle)
        self.real_B = self.input_B.to(self.device)


    def backward_D(self):
        fake_AB = self.fake_B
        self.gt_latent_fake = self.vgg(Variable(self.fake_B.data, requires_grad=False))
        self.gt_latent_real = self.vgg(Variable(self.input_B, requires_grad=False))
        real_AB = self.real_B

        self.pred_fake = self.netD(fake_AB.detach())
        self.pred_real = self.netD(real_AB)
        self.loss_D_fake = self.criterionGAN(self.pred_fake, self.pred_real, True)

        self.pred_fake_F = self.netF(self.gt_latent_fake.relu3_3.detach())
        self.pred_real_F = self.netF(self.gt_latent_real.relu3_3)
        self.loss_F_fake = self.criterionGAN(self.pred_fake_F,self.pred_real_F, True)

        self.loss_D =self.loss_D_fake * 0.5 + self.loss_F_fake  * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_AB = self.fake_B
        fake_f = self.gt_latent_fake
        
        pred_fake = self.netD(fake_AB)
        pred_fake_f = self.netF(fake_f.relu3_3)
        
        pred_real=self.netD(self.real_B)
        pred_real_F=self.netF(self.gt_latent_real.relu3_3)

        self.loss_G_GAN = self.criterionGAN(pred_fake,pred_real, False)+self.criterionGAN(pred_fake_f, pred_real_F,False)
        self.loss_G_L1 =( self.criterionL1(self.fake_B, self.real_B) +self.criterionL1(self.fake_P, self.real_B) )* self.opt.lambda_A


        self.loss_G = self.loss_G_L1 + self.loss_G_GAN * self.opt.gan_weight

        self.ng_loss_value = 0
        self.ng_loss_value2 = 0
        if self.opt.cosis:
            for gl in self.Cosis_list:
                self.ng_loss_value += Variable(gl.loss.data, requires_grad=True)
            self.loss_G += self.ng_loss_value
            for gl in self.Cosis_list2:
                self.ng_loss_value2 += Variable(gl.loss.data, requires_grad=True)
            self.loss_G += self.ng_loss_value2

        self.loss_G_SC = self.criterionSC(self.label_pred[0],self.label_gt)
        self.loss_G = self.loss_G+self.opt.lambda_sc*self.loss_G_SC
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_D.zero_grad()
        self.optimizer_F.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.optimizer_F.step()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data.item()),
                            ('G_L1', self.loss_G_L1.data.item()),
                            ('G_SC', self.loss_G_SC.data.item()),
                            ('G_C1', self.ng_loss_value.item()),
                            ('G_C2', self.ng_loss_value2.item()),
                            ('D', self.loss_D_fake.data.item()),
                            ('F', self.loss_F_fake.data.item())
                            ])

    def get_current_visuals(self):

        real_A =self.real_A.data
        fake_B = self.fake_B.data
        real_B =self.real_B.data
        coarse = self.fake_P.data
        return real_A,real_B,coarse,fake_B


    def save(self, epoch):
        self.save_network(self.netG, 'G', epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', epoch, self.gpu_ids)
        self.save_network(self.netF, 'F', epoch, self.gpu_ids)

    def load(self, epoch):
        self.load_network(self.netG, 'G', epoch)



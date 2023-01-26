import itertools

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from Bias_GAN.code.src.data import CycleDataset, ProjectionDataset


"""
The CycleGAN model implementation has been adopted from
https://github.com/Adi-iitd/AI-Art/blob/master/src/CycleGAN/CycleGAN-PL.py
"""


class DataModule(pl.LightningDataModule):

    def __init__(self,
                 config,
                 training_batch_size: int = 4,
                 test_batch_size: int = 64):


        super().__init__()

        self.config = config
        self.training_batch_size = training_batch_size
        self.test_batch_size = test_batch_size

    def setup(self, stage: str = None):

        if stage == 'fit' or stage is None:
            self.train = CycleDataset('train', self.config)
            self.valid = CycleDataset('valid', self.config)

        if stage == 'test':
            self.test = CycleDataset('test', self.config)

        if stage == 'predict':
            self.test = ProjectionDataset(self.config)


    def train_dataloader(self):
        return DataLoader(self.train,
                         batch_size=self.training_batch_size,
                         shuffle=True,
                         num_workers=0,
                         pin_memory=True)


    def val_dataloader  (self):
        return DataLoader(self.valid,
                          batch_size=self.test_batch_size,
                          shuffle=False,
                          num_workers=0,
                          pin_memory=True)


    def test_dataloader (self):
        return DataLoader(self.test,
                          batch_size=self.test_batch_size,
                          shuffle=False,
                          num_workers=0,
                          pin_memory=True)


def show_image(image):
    plt.imshow((image.squeeze()))


def get_random_sample(dataset):
    return dataset[np.random.randint(0, len(dataset))]


class ConstrainedGenerator(torch.nn.Module):
    def __init__(self, generator_model: torch.nn.Module, constrain=True):
        super(ConstrainedGenerator, self).__init__()
        self.generator =  generator_model
        self.constrained_layer = ConstraintLayer(constrain=constrain)
        
    def forward(self, x):
        out = self.generator(x)
        out = self.constrained_layer(x, out)
        return out


class ConstraintLayer(nn.Module):
    """ Layer constraining the Generator output to conserve the sum of the input feature values.
        Before rescaling the output, inverse transforms are applied to convert the input and output
        to precipitation units.
    """


    def __init__(self, constrain=True):
        super().__init__()
        self.epsilon = torch.tensor(0.0001)
        self.constrain = constrain


    def model_inv_log_transform(self, x):
        return torch.exp(x + torch.log(self.epsilon)) - self.epsilon


    def model_inv_norm_minus1_to_plus1_transform(self, x, is_input=True):

        if is_input:
            historical_reference_min = torch.tensor(0)
            historical_reference_max = torch.tensor(2.895)

        else:
            historical_reference_min = torch.tensor(0)
            historical_reference_max = torch.tensor(4.152)

        x = (x + 1)/2
        results = x * (historical_reference_max - historical_reference_min) + historical_reference_min

        return results


    def forward(self, input, output): 

        output = self.model_inv_norm_minus1_to_plus1_transform(output, is_input=False)
        output = self.model_inv_log_transform(output)
        output = torch.nn.functional.relu(output)
            
        if self.constrain is True:
            input = self.model_inv_norm_minus1_to_plus1_transform(input, is_input=True)
            input = self.model_inv_log_transform(input)
            
            norm_fraction = input.sum(dim=(2,3))/output.sum(dim=(2,3))
            norm_fraction = norm_fraction.unsqueeze(-1)
            norm_fraction = norm_fraction.unsqueeze(-1)
            output = output*norm_fraction

        return output


class ResBlock(nn.Module):

    def __init__(self, in_channels: int, apply_dropout: bool = True):

        """
                            Defines a ResBlock
        X ------------------------identity------------------------
        |-- Convolution -- Norm -- ReLU -- Convolution -- Norm --|
        """

        super().__init__()

        conv = nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, stride = 1)
        layers =  [nn.ReflectionPad2d(1), conv, nn.InstanceNorm2d(in_channels), nn.ReLU(True)]

        if apply_dropout:
            layers += [nn.Dropout(0.5)]

        conv = nn.Conv2d(in_channels = in_channels, out_channels = in_channels, kernel_size = 3, stride = 1)
        layers += [nn.ReflectionPad2d(1), conv, nn.InstanceNorm2d(in_channels)]

        self.net = nn.Sequential(*layers)


    def forward(self, x): return x + self.net(x)



class Generator(nn.Module):

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 64,
                 apply_dropout: bool = True,
                 image_size: int = 128,
                 num_resnet_blocks=6,
                 default_nbr_resnet=True
                 ):

        super().__init__()

        f = 1
        num_downsampling = 0
        
        if default_nbr_resnet==True:
            num_resnet_blocks = 6 if image_size == 128 else 9
        
        if default_nbr_resnet==False:
            num_resnet_blocks=num_resnet_blocks
            print("num_resnet_blocks:",num_resnet_blocks)
            

        conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 7, stride = 1)
        self.layers = [nn.ReflectionPad2d(3), conv, nn.InstanceNorm2d(out_channels), nn.ReLU(True)]

        for i in range(num_downsampling):
            conv = nn.Conv2d(out_channels * f, out_channels * 2 * f, kernel_size = 3, stride = 2, padding = 1)
            self.layers += [conv, nn.InstanceNorm2d(out_channels * 2 * f), nn.ReLU(True)]
            f *= 2

        for i in range(num_resnet_blocks):
            resnet_block = ResBlock(in_channels = out_channels * f, apply_dropout = apply_dropout)
            self.layers += [resnet_block]

        for i in range(num_downsampling):
            conv = nn.ConvTranspose2d(out_channels * f, out_channels * (f//2), 3, 2, padding = 1, output_padding = 1)
            self.layers += [conv, nn.InstanceNorm2d(out_channels * (f//2)), nn.ReLU(True)]
            f = f // 2

        conv = nn.Conv2d(in_channels = out_channels, out_channels = in_channels, kernel_size = 7, stride = 1)
        self.layers += [nn.ReflectionPad2d(3), conv, nn.Tanh()]

        self.net = nn.Sequential(*self.layers)


    def forward(self, x): 
       
        x = self.net(x)

        return x 


class Discriminator(nn.Module):

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 64,
                 num_layers: int = 3):

        """
                                    Discriminator Architecture!
        C64 - C128 - C256 - C512, where Ck denote a Convolution-InstanceNorm-LeakyReLU layer with k filters
        """

        super().__init__()
        in_f  = 1
        out_f = 2

        conv = nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1)
        self.layers = [conv, nn.LeakyReLU(0.2, True)]

        for idx in range(1, num_layers):
            conv = nn.Conv2d(out_channels * in_f, out_channels * out_f, kernel_size = 4, stride = 2, padding = 1)
            self.layers += [conv, nn.InstanceNorm2d(out_channels * out_f), nn.LeakyReLU(0.2, True)]
            in_f   = out_f
            out_f *= 2

        out_f = min(2 ** num_layers, 8)
        conv = nn.Conv2d(out_channels * in_f,  out_channels * out_f, kernel_size = 4, stride = 1, padding = 1)
        self.layers += [conv, nn.InstanceNorm2d(out_channels * out_f), nn.LeakyReLU(0.2, True)]

        conv = nn.Conv2d(out_channels * out_f, out_channels = 1, kernel_size = 4, stride = 1, padding = 1)
        self.layers += [conv]

        self.net = nn.Sequential(*self.layers)


    def forward(self, x): return self.net(x)


class Initializer:

    def __init__(self, init_type: str = 'normal', init_gain: float = 0.02):

        """
        Parameters:
            init_type: Initializer type - 'kaiming' or 'xavier' or 'normal'
            init_gain: Standard deviation of the normal distribution
        """
        
        self.init_type = init_type
        self.init_gain = init_gain


    def init_module(self, m):

        cls_name = m.__class__.__name__;
        if hasattr(m, 'weight') and (cls_name.find('Conv') != -1 or cls_name.find('Linear') != -1):

            if   self.init_type == 'kaiming': nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif self.init_type == 'xavier' : nn.init.xavier_normal_ (m.weight.data,  gain = self.init_gain)
            elif self.init_type == 'normal' : nn.init.normal_(m.weight.data, mean = 0, std = self.init_gain)
            else: raise ValueError('Initialization not found!!')

            if m.bias is not None: nn.init.constant_(m.bias.data, val = 0);

        if hasattr(m, 'weight') and cls_name.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, mean = 1.0, std = self.init_gain)
            nn.init.constant_(m.bias.data, val = 0)


    def __call__(self, net):

        """
        Parameters:
            net: Network
        """

        net.apply(self.init_module)

        return net


class ImagePool:

    """
    This class implements an image buffer that stores previously generated images! This buffer enables to update
    discriminators using a history of generated image rather than the latest ones produced by generator.
    """

    def __init__(self, pool_sz: int = 50):

        """
        Parameters:
            pool_sz: Size of the image buffer
        """

        self.nb_images = 0
        self.image_pool = []
        self.pool_sz = pool_sz


    def push_and_pop(self, images):

        """
        Parameters:
            images: latest images generated by the generator
        Returns a batch of images from pool!
        """

        images_to_return = []
        for image in images:
            image = torch.unsqueeze(image, 0)

            if  self.nb_images < self.pool_sz:
                self.image_pool.append (image)
                images_to_return.append(image)
                self.nb_images += 1
            else:
                if np.random.uniform(0, 1) > 0.5:

                    rand_int = np.random.randint(0, self.pool_sz)
                    temp_img = self.image_pool[rand_int].clone()
                    self.image_pool[rand_int] = image
                    images_to_return.append(temp_img)
                else:
                    images_to_return.append(image)

        return torch.cat(images_to_return, 0)


class Loss:

    """
    This class implements different losses required to train the generators and discriminators of CycleGAN
    """

    def __init__(self, loss_type: str = 'MSE', lambda_: int = 10):

        """
        Parameters:
            loss_type: Loss Function to train CycleGAN
            lambda_:   Weightage of Cycle-consistency loss
        """

        self.loss = nn.MSELoss() if loss_type == 'MSE' else nn.BCEWithLogitsLoss()
        self.lambda_ = lambda_


    def get_dis_loss(self, dis_pred_real_data, dis_pred_fake_data):

        """
        Parameters:
            dis_pred_real_data: Discriminator's prediction on real data
            dis_pred_fake_data: Discriminator's prediction on fake data
        """

        dis_tar_real_data = torch.ones_like (dis_pred_real_data, requires_grad = False)
        dis_tar_fake_data = torch.zeros_like(dis_pred_fake_data, requires_grad = False)

        loss_real_data = self.loss(dis_pred_real_data, dis_tar_real_data)
        loss_fake_data = self.loss(dis_pred_fake_data, dis_tar_fake_data)

        dis_tot_loss = (loss_real_data + loss_fake_data) * 0.5

        return dis_tot_loss


    def get_gen_gan_loss(self, dis_pred_fake_data):

        """
        Parameters:
            dis_pred_fake_data: Discriminator's prediction on fake data
        """

        gen_tar_fake_data = torch.ones_like(dis_pred_fake_data, requires_grad = False)
        gen_tot_loss = self.loss(dis_pred_fake_data, gen_tar_fake_data)

        return gen_tot_loss


    def get_gen_cyc_loss(self, real_data, cyc_data):

        """
        Parameters:
            real_data: Real images sampled from the dataloaders
            cyc_data:  Image reconstructed after passing the real image through both the generators
                       X_recons = F * G (X_real), where F and G are the two generators
        """

        gen_cyc_loss = torch.nn.L1Loss()(real_data, cyc_data)
        gen_tot_loss = gen_cyc_loss * self.lambda_

        return gen_tot_loss


    def get_gen_idt_loss(self, real_data, idt_data):

        """
        Implements the identity loss:
            nn.L1Loss(LG_B2A(real_A), real_A)
            nn.L1Loss(LG_A2B(real_B), real_B)
        """

        gen_idt_loss = torch.nn.L1Loss()(real_data, idt_data)
        gen_tot_loss = gen_idt_loss * self.lambda_ * 0.5

        return gen_tot_loss


    def get_gen_loss(self, real_A, real_B, cyc_A, cyc_B, idt_A, idt_B, d_A_pred_fake_data,
                     d_B_pred_fake_data):

        """
        Implements the total Generator loss
        Sum of Cycle loss, Identity loss, and GAN loss
        """

        #Cycle loss
        cyc_loss_A = self.get_gen_cyc_loss(real_A, cyc_A)
        cyc_loss_B = self.get_gen_cyc_loss(real_B, cyc_B)
        tot_cyc_loss = cyc_loss_A + cyc_loss_B

        # GAN loss
        g_A2B_gan_loss = self.get_gen_gan_loss(d_B_pred_fake_data)
        g_B2A_gan_loss = self.get_gen_gan_loss(d_A_pred_fake_data)

        # Identity loss
        g_B2A_idt_loss = self.get_gen_idt_loss(real_A, idt_A)
        g_A2B_idt_loss = self.get_gen_idt_loss(real_B, idt_B)

        # Total individual losses
        g_A2B_loss = g_A2B_gan_loss + g_A2B_idt_loss + tot_cyc_loss
        g_B2A_loss = g_B2A_gan_loss + g_B2A_idt_loss + tot_cyc_loss
        g_tot_loss = g_A2B_loss + g_B2A_loss - tot_cyc_loss

        return g_A2B_loss, g_B2A_loss, g_tot_loss


class CycleGAN(pl.LightningModule):

    def __init__(self, d_lr: float = 2e-4, g_lr: float = 2e-4, beta_1: float = 0.5, beta_2: float = 0.999, 
                 epoch_decay: int = 200, running_bias=False,num_resnet_blocks=6, default_nbr_resnet=True):

        super().__init__()

        self.d_lr = d_lr
        self.g_lr = g_lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epoch_decay = epoch_decay

        self.fake_pool_A = ImagePool(pool_sz = 50)
        self.fake_pool_B = ImagePool(pool_sz = 50)

        self.loss = Loss(loss_type = 'MSE', lambda_ = 10)
        init = Initializer(init_type = 'normal', init_gain = 0.02)

        self.d_A = init(Discriminator(in_channels = 1, out_channels = 64, num_layers = 3))
        self.d_B = init(Discriminator(in_channels = 1, out_channels = 64, num_layers = 3))
        self.g_A2B = init(Generator(in_channels = 1, out_channels = 64, apply_dropout = False,num_resnet_blocks=num_resnet_blocks,default_nbr_resnet=default_nbr_resnet))
        self.g_B2A = init(Generator(in_channels = 1, out_channels = 64, apply_dropout = False,num_resnet_blocks=num_resnet_blocks,default_nbr_resnet=default_nbr_resnet))

        self.d_A_params = self.d_A.parameters()
        self.d_B_params = self.d_B.parameters()
        self.g_params   = itertools.chain([*self.g_A2B.parameters(), *self.g_B2A.parameters()])

        #self.example_input_array = [torch.rand(1, 3, image_size, image_size, device = self.device),
        #                            torch.rand(2, 3, image_size, image_size, device = self.device)]
        self.running_bias = running_bias 
        self.train_bias_buffer = []
        self.val_bias_buffer = []


    @staticmethod
    def set_requires_grad(nets, requires_grad = False):

        """
        Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """

        if not isinstance(nets, list): nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad


    def forward(self, real_A, real_B):
        
        """
        This is different from the training step. You should treat this as the final inference code 
        (final outputs that you are looking for!), but you can definitely use it in the training_step 
        to make some code reusable.
        Parameters:
            real_A -- real image of A
            real_B -- real image of B
        """
        
        fake_B = self.g_A2B(real_A)
        fake_A = self.g_B2A(real_B)

        return fake_B, fake_A
    
    
    def forward_gen(self, real_A, real_B, fake_A, fake_B):
        
        """
        Gets the remaining output of both the generators for the training/validation step
        Parameters:
            real_A -- real image of A
            real_B -- real image of B
            fake_A -- fake image of A
            fake_B -- fake image of B
        """
        
        cyc_A = self.g_B2A(fake_B)
        idt_A = self.g_B2A(real_A)
        
        cyc_B = self.g_A2B(fake_A)
        idt_B = self.g_A2B(real_B)
        
        return cyc_A, idt_A, cyc_B, idt_B
    
    
    @staticmethod
    def forward_dis(dis, real_data, fake_data):
        
        """
        Gets the Discriminator output
        Parameters:
            dis       -- Discriminator
            real_data -- real image
            fake_data -- fake image
        """
        
        pred_real_data = dis(real_data)
        pred_fake_data = dis(fake_data)
        
        return pred_real_data, pred_fake_data


    def running_bias(self, real_A, cyc_A, stage: str = 'train', buffer_length=10_000):

        tmp_real_A = real_A.squeeze()
        tmp_real_A = ConstraintLayer().model_inv_norm_minus1_to_plus1_transform(tmp_real_A, is_input=False)
        tmp_real_A = ConstraintLayer().model_inv_log_transform(tmp_real_A)*24*3600

        tmp_cyc_A = cyc_A.squeeze()
        tmp_cyc_A = ConstraintLayer().model_inv_norm_minus1_to_plus1_transform(tmp_cyc_A, is_input=False)
        tmp_cyc_A = ConstraintLayer().model_inv_log_transform(tmp_cyc_A)*24*3600

        diff = tmp_real_A - tmp_cyc_A

        if stage == 'train':
            self.train_bias_buffer.append(diff.unsqueeze(0))
            if len(self.train_bias_buffer) == buffer_length:
                self.train_bias_buffer.pop(0) 
            bias = (torch.abs(torch.cat(self.train_bias_buffer).mean(dim=0))).mean()

        if stage == 'val':
            self.val_bias_buffer.append(diff)
            if len(self.val_bias_buffer) == buffer_length/diff.shape[0]:
                self.val_bias_buffer.pop(0) 
            bias = (torch.abs(torch.cat(self.val_bias_buffer).mean(dim=0))).mean()

        return bias


    def training_step(self, batch, batch_idx, optimizer_idx):

        real_A, real_B = batch['A'], batch['B']
        fake_B, fake_A = self(real_A, real_B)
        
        
        if optimizer_idx == 0:
            
            cyc_A, idt_A, cyc_B, idt_B = self.forward_gen(real_A, real_B, fake_A, fake_B)

            if self.running_bias is True:
                g_B2A_bias = self.running_bias(real_A, cyc_B, stage='train')
            
            # No need to calculate the gradients for Discriminators' parameters
            self.set_requires_grad([self.d_A, self.d_B], requires_grad = False)
            d_A_pred_fake_data = self.d_A(fake_A)
            d_B_pred_fake_data = self.d_B(fake_B)

            g_A2B_loss, g_B2A_loss, g_tot_loss = self.loss.get_gen_loss(real_A, real_B, cyc_A, cyc_B, idt_A, idt_B, 
                                                                        d_A_pred_fake_data, d_B_pred_fake_data)

            dict_ = {'g_tot_train_loss': g_tot_loss,
                     'g_A2B_train_loss': g_A2B_loss,
                     'g_B2A_train_loss': g_B2A_loss,
                    }

            if self.running_bias is True:
                dict_['g_B2A_train_bias'] =  g_B2A_bias

            self.log_dict(dict_, on_step = True,
                                 on_epoch = True,
                                 prog_bar = True,
                                 logger = True)
            
            self.logger.experiment.add_scalar("Total Generator Loss/Train",g_tot_loss ,self.current_epoch)

            return g_tot_loss
        

        if optimizer_idx == 1:
            
            self.set_requires_grad([self.d_A], requires_grad = True)
            fake_A = self.fake_pool_A.push_and_pop(fake_A)
            d_A_pred_real_data, d_A_pred_fake_data = self.forward_dis(self.d_A, real_A, fake_A.detach())

            # GAN loss
            d_A_loss = self.loss.get_dis_loss(d_A_pred_real_data, d_A_pred_fake_data)
            self.log("d_A_train_loss", d_A_loss, on_step = True, on_epoch = True, prog_bar = True, logger = True)
            
            self.logger.experiment.add_scalar("Discriminator A Loss/Train",d_A_loss ,self.current_epoch)

            return d_A_loss
        

        if optimizer_idx == 2:
            
            self.set_requires_grad([self.d_B], requires_grad = True)
            fake_B = self.fake_pool_B.push_and_pop(fake_B)
            d_B_pred_real_data, d_B_pred_fake_data = self.forward_dis(self.d_B, real_B, fake_B.detach())

            # GAN loss
            d_B_loss = self.loss.get_dis_loss(d_B_pred_real_data, d_B_pred_fake_data)
            self.log("d_B_train_loss", d_B_loss, on_step = True, on_epoch = True, prog_bar = True, logger = True)
            
            self.logger.experiment.add_scalar("Discriminator B Loss/Train",d_B_loss ,self.current_epoch)

            return d_B_loss


    def shared_step(self, batch, stage: str = 'val'):

        grid_A = []
        grid_B = []

        real_A, real_B = batch['A'], batch['B']
        
        fake_B, fake_A = self(real_A, real_B)
        cyc_A , idt_A , cyc_B, idt_B = self.forward_gen(real_A, real_B, fake_A, fake_B)

        if self.running_bias is True:
            g_B2A_bias = self.running_bias(real_A, cyc_B, stage='val')
        
        d_A_pred_real_data, d_A_pred_fake_data = self.forward_dis(self.d_A, real_A, fake_A)
        d_B_pred_real_data, d_B_pred_fake_data = self.forward_dis(self.d_B, real_B, fake_B)
        
        # G_A2B loss, G_B2A loss, G loss
        g_A2B_loss, g_B2A_loss, g_tot_loss = self.loss.get_gen_loss(real_A, real_B, cyc_A, cyc_B, idt_A, idt_B, 
                                                                    d_A_pred_fake_data, d_B_pred_fake_data)

        # D_A loss, D_B loss
        d_A_loss = self.loss.get_dis_loss(d_A_pred_real_data, d_A_pred_fake_data)
        d_B_loss = self.loss.get_dis_loss(d_B_pred_real_data, d_B_pred_fake_data)

        dict_ = {f'g_tot_{stage}_loss': g_tot_loss,
                 f'g_A2B_{stage}_loss': g_A2B_loss,
                 f'g_B2A_{stage}_loss': g_B2A_loss, 
                 f'd_A_{stage}_loss'  : d_A_loss  ,
                 f'd_B_{stage}_loss'  : d_B_loss}

        if self.running_bias is True:
            dict_[f'g_B2A_{stage}_bias'] =  g_B2A_bias

        self.log_dict(dict_, on_step = False, on_epoch = True, prog_bar = True, logger = True)

        for i in range(12):
            rand_int = np.random.randint(0, len(real_A))
            tensor = torch.stack([real_A[rand_int], fake_B[rand_int], cyc_A[rand_int],
                                  real_B[rand_int], fake_A[rand_int], cyc_B[rand_int]])
            tensor = (tensor + 1) / 2
            grid_A.append(tensor[:3])
            grid_B.append(tensor[3:])
        
        # log the results on tensorboard
        grid_A = torchvision.utils.make_grid(torch.cat(grid_A, 0), nrow = 6)
        grid_B = torchvision.utils.make_grid(torch.cat(grid_B, 0), nrow = 6)
        self.logger.experiment.add_image('Grid_A', grid_A, self.current_epoch, dataformats = "CHW")
        self.logger.experiment.add_image('Grid_B', grid_B, self.current_epoch, dataformats = "CHW")


    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')


    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test')


    def lr_lambda(self, epoch):

        fraction = (epoch - self.epoch_decay) / self.epoch_decay
        return 1 if epoch < self.epoch_decay else 1 - fraction


    def configure_optimizers(self):
        
        # define the optimizers here
        g_opt   = torch.optim.Adam(self.g_params  , lr = self.g_lr, betas = (self.beta_1, self.beta_2))
        d_A_opt = torch.optim.Adam(self.d_A_params, lr = self.d_lr, betas = (self.beta_1, self.beta_2))
        d_B_opt = torch.optim.Adam(self.d_B_params, lr = self.d_lr, betas = (self.beta_1, self.beta_2))
        
        # define the lr_schedulers here
        g_sch   = optim.lr_scheduler.LambdaLR(g_opt  , lr_lambda = self.lr_lambda)
        d_A_sch = optim.lr_scheduler.LambdaLR(d_A_opt, lr_lambda = self.lr_lambda)
        d_B_sch = optim.lr_scheduler.LambdaLR(d_B_opt, lr_lambda = self.lr_lambda)
        
        # first return value is a list of optimizers and second is a list of lr_schedulers 
        # (you can return empty list also)
        return [g_opt, d_A_opt, d_B_opt], [g_sch, d_A_sch, d_B_sch]




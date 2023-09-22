import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
# from efficientnet_pytorch import EfficientNet
import torchvision.models as models
import torch.nn.functional as F
###############################################################################
# Helper Functions
###############################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
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


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_G(netG, flag = None, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'netg':
        net = NetG(flag = None,norm_layer=norm_layer)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(netD, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'netd':  # default PatchGAN classifier
        net = NetD(norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)





def define_C(netC,gpu_ids=[]):
    net = None
    if netC == 'netc':
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            classification = models.mobilenet_v2(num_classes=2)
            classification.to(gpu_ids[0])
            net = torch.nn.DataParallel(classification, gpu_ids)
    else:
        raise NotImplementedError('Classifier model name [%s] is not recognized' % netC)
    return net

##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None





class Decoder(nn.Module):
    def __init__(self, in_channel = 160, in_width = 3,flag = None, norm_layer=nn.BatchNorm2d):
        super(Decoder,self).__init__()
        self.channel = in_channel
        self.width = in_width

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        main = nn.Sequential()
        # if flag == True:
        main.add_module('decoder convt {}-{}'.format(self.channel+1, self.channel - 80), nn.ConvTranspose2d(self.channel+1, self.channel - 80, 4, 2,1, bias=use_bias))
        # else :
        #     main.add_module('decoder convt {}-{}'.format(self.channel, self.channel - 80),nn.ConvTranspose2d(self.channel, self.channel - 80, 4, 2, 1, bias=use_bias))
        main.add_module('batchnorm {}'.format(self.channel - 80),nn.BatchNorm2d(self.channel - 80))
        self.channel -=80 #80
        self.width *= 2 #14

        main.add_module('decoder convt {}-{}'.format(self.channel, self.channel - 40), nn.ConvTranspose2d(self.channel, self.channel - 40, 4,2,1, bias=use_bias))
        main.add_module('batchnorm {}'.format(self.channel - 40), nn.BatchNorm2d(self.channel - 40))
        main.add_module('relu2', nn.ReLU(inplace=False))
        self.channel -= 40 # 40
        self.width *= 7 #28

        main.add_module('decoder convt {}-{}'.format(self.channel, self.channel - 20),
                        nn.ConvTranspose2d(self.channel, self.channel - 20, 4, 2, 1, bias=use_bias))
        main.add_module('batchnorm {}'.format(self.channel - 20), nn.BatchNorm2d(self.channel - 20))
        main.add_module('relu3', nn.ReLU(inplace=False))
        self.channel -= 20 #20
        self.width *= 2 #56

        main.add_module('decoder convt {}-{}'.format(self.channel, self.channel - 10),
                        nn.ConvTranspose2d(self.channel, self.channel - 10, 4, 2, 1, bias=use_bias))
        main.add_module('batchnorm {}'.format(self.channel - 10), nn.BatchNorm2d(self.channel - 10))
        main.add_module('relu4', nn.ReLU(inplace=False))
        self.channel -= 10 #10
        self.width *= 2 #112

        main.add_module('decoder convt {}-{}'.format(self.channel, self.channel - 5),
                        nn.ConvTranspose2d(self.channel, self.channel - 5, 4, 2, 1, bias=use_bias))
        main.add_module('batchnorm {}'.format(self.channel - 5), nn.BatchNorm2d(self.channel - 5))
        main.add_module('relu5', nn.ReLU(inplace=False))
        self.channel -= 5 #5
        self.width *= 2 #224


        main.add_module('decoder convt {}-{}'.format(self.channel, self.channel - 2),
                        nn.ConvTranspose2d(self.channel,  3, 3, 1, 1, bias=use_bias))
        main.add_module('relu7', nn.ReLU(inplace=False))

        self.main = main
    def forward(self, input, target):
        if target == None:
            output = self.main(input)
        else:
            size = input.shape[-1]
            c_labels = torch.zeros(target.shape[0], 1, size, size).cuda()
            # print('zeros =', zeros)
            # print('zeros shape = ', zeros.shape)
            for i in range(target.shape[0]):
                c_labels[i:] = target[i]
            # print('c_labels shape = ', c_labels.shape)
            # input.cuda()
            # c_labels.cuda()
            input = torch.cat((input, c_labels), 1)
            output = self.main(input)
        return output



class mobilenet_v2_layer_output(nn.Module):
    def __init__(self):
        super().__init__()
        self.mobilenet_v2_feat = models.mobilenet_v2(pretrained=True).features
        self.layer_name_mapping = {
            '16': "InvertedResidual-144"
        }

    def forward(self, x):
        output = {}
        for name, module in self.mobilenet_v2_feat._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return output


class Encoder(nn.Module):
    def __init__(self,in_channel = 3, in_width = 224,norm_layer=nn.BatchNorm2d):
        super(Encoder,self).__init__()
        self.channel = 160
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.new_model = mobilenet_v2_layer_output()
        conv_tmp = nn.Sequential()
        conv_tmp.add_module('conv', nn.Conv2d(self.channel, self.channel, 3, 1, 1, bias=use_bias))
        self.conv_tmp = conv_tmp
        self.fc1 = nn.Linear(8 * 8* 160, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, input):
        output = self.new_model(input)
        for key, value in output.items():
            output02 = value
        # print('output02 shape = ', output02.shape)
        output03 = self.conv_tmp(output02)
        out = output03.view(output03.size(0), -1)
        # print('out shape =', out.shape)
        out = self.fc1(out)
        out = self.bn1(out)
        # out = F.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        # out = F.relu(out)
        prob = torch.sigmoid(self.fc3(out))
        return output03,prob




class NetG(nn.Module):
    @staticmethod
    def name():
        return 'netg'
    def __init__(self,flag = None, norm_layer=nn.BatchNorm2d):
        super(NetG, self).__init__()
        self.encoder = Encoder(in_channel=3, in_width=224,norm_layer=norm_layer)
        self.decoder = Decoder( in_width=8,flag = None, norm_layer=norm_layer)

    def forward(self,input, target):
        latent_i,prob= self.encoder(input)
        gen_image = self.decoder(latent_i,target)
        return gen_image, latent_i,prob

class NetD(nn.Module):
    @staticmethod
    def name():
        return 'netd'

    def __init__(self,norm_layer=nn.BatchNorm2d):
        super(NetD,self).__init__()
        self.feature = Encoder(in_channel= 3, in_width=224,norm_layer=norm_layer)

    def forward(self,input):
            feature,prob = self.feature(input)
            return feature, prob



class CE_loss_fn(nn.Module):
    def __init__(self):
        super(CE_loss_fn, self).__init__()
        self.loss = nn.CrossEntropyLoss()
    def __call__(self, outputs , labels):
        loss = self.loss(outputs, labels)
        return loss


def l2_loss(input, target, size_average = True):
    """
    l2 loss without reduce flag
    :param input: (FloatTensor) Input tensor
    :param target: (FloatTensor) Output tensor
    :param size_average:
    :return: L2 distance between input and output
    """

    if size_average:
        return torch.mean(torch.pow((input - target), 2))
    else:
        return torch.pow((input - target), 2)
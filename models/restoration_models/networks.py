import yaml
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Partial') != -1 or classname.find('PConv') != -1:
        pass
    elif classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
             n_blocks_local=3, norm='instance', gpu_ids=[], ymlPath=""):    
    norm_layer = get_norm_layer(norm_type=norm)     
    if netG == 'global':    
        netG = GlobalGenerator(3, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)       
    elif netG == 'ffc':
        with open(ymlPath, 'r', encoding='utf-8') as f:
            config = f.read() 
        generater_cfg = yaml.load(config,Loader=yaml.FullLoader)
        # netG = FFCResNetGeneratorwithHistogramAttention(input_nc, output_nc, ngf, n_downsample_global, 9, norm_layer,init_conv_kwargs=generater_cfg['init_conv_kwargs'], downsample_conv_kwargs=generater_cfg['downsample_conv_kwargs'],resnet_conv_kwargs=generater_cfg['resnet_conv_kwargs'])
        netG = FFCResNetGenerator(input_nc, output_nc, ngf, n_downsample_global, 9, norm_layer,init_conv_kwargs=generater_cfg['init_conv_kwargs'], downsample_conv_kwargs=generater_cfg['downsample_conv_kwargs'],resnet_conv_kwargs=generater_cfg['resnet_conv_kwargs'])
    elif netG == 'local':        
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
                                  n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    else:
        raise('generator not implemented!')
    #print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())   
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm) 
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat) 
    # netD = FFCNLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer) 
    print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################

def GP(critic,real,fake,device='cpu'):
    BATCH_SIZE,C,H,W = real.shape
    alpha = torch.randn(size=(BATCH_SIZE,1,1,1)).repeat(1,C,H,W).to(device)
    interpolated_images=real*alpha + fake*(1-alpha)

    #计算判别器输出
    mixed_scores = critic(interpolated_images)
    #求导
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True
    )[0]
    gradient = gradient.view(gradient.shape[0],-1)
    gradient_norm = gradient.norm(2,dim = 1)
    gradient_penality = torch.mean((gradient_norm - 1)**2)
    return gradient_penality


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

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
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss

##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model        
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)                

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]      

            ### final convolution
            if n == n_local_enhancers:                
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]                       
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])        
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input)             


class FFCResNetGeneratorwithHistogramAttention(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act='sigmoid', max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### histgram
        self.hist_layer_local = HistogramLayerLocal()
        
        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        ### upsample
        model_ups = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model_ups += [nn.ConvTranspose2d(min(max_features, ngf * mult)+768,
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]
        
        model_head = []
        if out_ffc:
            model_head += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model_head += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model_head.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)
        self.model_ups = nn.Sequential(*model_ups)
        self.model_head = nn.Sequential(*model_head)

    def forward(self, input, ref=None):
        hist = self.hist_layer_local(input)
        
        feat_i = self.model(input)
        i = 0
        while i<12:
            hist_i = F.avg_pool2d(hist, 2**(4-(i//3)))
            feat_i = torch.cat([feat_i, hist_i],1)
            for j in range(3):
                feat_i = self.model_ups[i+j](feat_i)
            i+=3
        
        out = self.model_head(feat_i)
        return out

class FFCResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act='sigmoid', max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        model = [nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, kernel_size=7, padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, **init_conv_kwargs)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                cur_conv_kwargs = dict(downsample_conv_kwargs)
                cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            else:
                cur_conv_kwargs = downsample_conv_kwargs
            model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 kernel_size=3, stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 **cur_conv_kwargs)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)
        
        ### resnet blocks
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            model += [cur_resblock]

        model += [ConcatTupleLayer()]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            model += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

class PConvFFCResNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act='sigmoid', max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        dslayers = [nn.ReflectionPad2d(3),
                 PConv_BN_ACT(input_nc, ngf, kernel_size=7, padding=0)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            # if i == n_downsampling - 1:
            #     cur_conv_kwargs = dict(downsample_conv_kwargs)
            #     cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            # else:
            #     cur_conv_kwargs = downsample_conv_kwargs
            dslayers += [PConv_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 stride=2)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        # ### histgram
        # self.hist_layer_local = HistogramLayerLocal()
        
        ### resnet blocks
        residualblocks = []
        for i in range(n_blocks):
            cur_resblock = FFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, inline=True,  **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            residualblocks += [cur_resblock]

        ### upsample
        uplayers = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            uplayers += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]

        if out_ffc:
            uplayers += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        uplayers += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            uplayers.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.dslayers = nn.Sequential(*dslayers)
        self.residualblocks = nn.Sequential(*residualblocks)
        self.uplayers = nn.Sequential(*uplayers)

    def forward(self, input, mask):
        # downsampling
        for i in range(len(self.dslayers)):
            if i==0:
                feat = self.dslayers[0](input)
                new_mask = self.dslayers[0](mask)
            else:
                feat, new_mask = self.dslayers[i](feat, new_mask)

        # residual blocks

        feat = self.residualblocks(feat)
        # for i in range(len(self.residualblocks)):
        #     # feat, new_mask = self.residualblocks[i](feat, new_mask)
        #     feat = self.residualblocks[i](feat)
        del new_mask
        
        out = self.uplayers(feat)
        return out


class PConvFFCResNetGeneratorwithHistogramAttention(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 up_norm_layer=nn.BatchNorm2d, up_activation=nn.ReLU(True),
                 init_conv_kwargs={}, downsample_conv_kwargs={}, resnet_conv_kwargs={},
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 add_out_act='sigmoid', max_features=1024, out_ffc=False, out_ffc_kwargs={}):
        assert (n_blocks >= 0)
        super().__init__()

        dslayers = [nn.ReflectionPad2d(3),
                 PConv_BN_ACT(input_nc, ngf, kernel_size=7, padding=0)]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            # if i == n_downsampling - 1:
            #     cur_conv_kwargs = dict(downsample_conv_kwargs)
            #     cur_conv_kwargs['ratio_gout'] = resnet_conv_kwargs.get('ratio_gin', 0)
            # else:
            #     cur_conv_kwargs = downsample_conv_kwargs
            dslayers += [PConv_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),
                                 stride=2)]

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)
        
        ### histgram
        self.hist_layer_local = HistogramLayerLocal()
        
        ### resnet blocks
        residualblocks = []
        for i in range(n_blocks):
            cur_resblock = PConvFFCResnetBlock(feats_num_bottleneck, padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer, **resnet_conv_kwargs)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
            residualblocks += [cur_resblock]

        ### upsample
        uplayers = []
        fusion_modules = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            fusion_modules += [HistFusionModule(min(max_features, ngf * mult)+768, min(max_features, ngf * mult))]
            uplayers += [nn.ConvTranspose2d(min(max_features, ngf * mult),
                                         min(max_features, int(ngf * mult / 2)),
                                         kernel_size=3, stride=2, padding=1, output_padding=1),
                      up_norm_layer(min(max_features, int(ngf * mult / 2))),
                      up_activation]


        model_head = []
        if out_ffc:
            model_head += [FFCResnetBlock(ngf, padding_type=padding_type, activation_layer=activation_layer,
                                     norm_layer=norm_layer, inline=True, **out_ffc_kwargs)]

        model_head += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if add_out_act:
            model_head.append(get_activation('tanh' if add_out_act is True else add_out_act))
        self.dslayers = nn.Sequential(*dslayers)
        self.residualblocks = nn.Sequential(*residualblocks)
        self.fusion_modules = nn.Sequential(*fusion_modules)
        self.uplayers = nn.Sequential(*uplayers)
        self.model_head = nn.Sequential(*model_head)

    def forward(self, input, mask, ref=None):
        # downsampling
        for i in range(len(self.dslayers)):
            if i==0:
                feat = self.dslayers[0](input)
                new_mask = self.dslayers[0](mask)
            else:
                feat, new_mask = self.dslayers[i](feat, new_mask)

        # residual blocks
        for i in range(len(self.residualblocks)):
            feat, new_mask = self.residualblocks[i](feat, new_mask)
        del new_mask
        
        if ref is not None:
            hist = self.hist_layer_local(ref)#, mask)
        else:
            # ref=input
            # hist = self.hist_layer_local(ref)#, mask)
            hist = torch.ones((input.shape[0],768,256,256)).to(input.device)
            
        i,k = 0,0
        while i<12:
            hist_i = F.avg_pool2d(hist, 2**(4-(i//3)))
            feat = torch.cat([feat, hist_i],1)
            feat = self.fusion_modules[k](feat)
            for j in range(3):
                feat = self.uplayers[i+j](feat)
            i+=3
            k+=1
    
        out = self.model_head(feat)
        
        return out


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()        
        self.output_nc = output_nc        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                 norm_layer(ngf), nn.ReLU(True)]             
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]        

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model) 

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))        
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i)).nonzero() # n x 4            
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]                    
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)                                        
                    outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat                       
        return outputs_mean

def get_activation(kind='tanh'):
    if kind == 'tanh':
        return nn.Tanh()
    if kind == 'sigmoid':
        return nn.Sigmoid()
    if kind is False:
        return nn.Identity()
    raise ValueError(f'Unknown activation kind {kind}')

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        res = x * y.expand_as(x)
        return res

class LearnableSpatialTransformWrapper(nn.Module):
    def __init__(self, impl, pad_coef=0.5, angle_init_range=80, train_angle=True):
        super().__init__()
        self.impl = impl
        self.angle = torch.rand(1) * angle_init_range
        if train_angle:
            self.angle = nn.Parameter(self.angle, requires_grad=True)
        self.pad_coef = pad_coef

    def forward(self, x):
        if torch.is_tensor(x):
            return self.inverse_transform(self.impl(self.transform(x)), x)
        elif isinstance(x, tuple):
            x_trans = tuple(self.transform(elem) for elem in x)
            y_trans = self.impl(x_trans)
            return tuple(self.inverse_transform(elem, orig_x) for elem, orig_x in zip(y_trans, x))
        else:
            raise ValueError(f'Unexpected input type {type(x)}')

    def transform(self, x):
        height, width = x.shape[2:]
        pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)
        x_padded = F.pad(x, [pad_w, pad_w, pad_h, pad_h], mode='reflect')
        x_padded_rotated = rotate(x_padded, angle=self.angle.to(x_padded))
        return x_padded_rotated

    def inverse_transform(self, y_padded_rotated, orig_x):
        height, width = orig_x.shape[2:]
        pad_h, pad_w = int(height * self.pad_coef), int(width * self.pad_coef)

        y_padded = rotate(y_padded_rotated, angle=-self.angle.to(y_padded_rotated))
        y_height, y_width = y_padded.shape[2:]
        y = y_padded[:, :, pad_h : y_height - pad_h, pad_w : y_width - pad_w]
        return y

class FFCSE_block(nn.Module):

    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r,
                               kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * \
            self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * \
            self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs) 

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output

class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g

class FFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out

class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        return out_xl, out_xg

##      pconv - ffc 

class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        # self.input_conv.apply(weights_init('kaiming'))
        torch.nn.init.kaiming_normal_(self.input_conv.weight)
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask

class PConv_BN_ACT(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilated=1, bn=True, activ='relu',
                 conv_bias=False):
        super().__init__()
        
        self.pconv = PartialConv(in_ch, out_ch, kernel_size, stride, padding, dilated, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, input_mask):
        h, h_mask = self.pconv(input, input_mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask

class PConvFFCResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False, **conv_kwargs):
        super().__init__()
        self.pconv = PConv_BN_ACT(dim, dim, activ='leaky')
        self.conv1 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        self.conv2 = FFC_BN_ACT(dim, dim, kernel_size=3, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                **conv_kwargs)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x, mask):
        # if self.inline:
        x, mask = self.pconv(x, mask)
        x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        # else:
        #     x, mask = self.pconv(x, mask)
        #     x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        # if self.inline:
        out = torch.cat(out, dim=1)
        return out, mask

class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output
    
    
class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)
    
class GaussianHistogram(nn.Module):
    """
    Use gaussian distribution
    Args:
        bins: number of bins to seperate values
        min: minium vale of the data
        max: maximum value of the data
        sigma: a learable paramerter, init=0.01
    """

    def __init__(self, bins, min, max, sigma, require_grad=False):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max

        self.sigma = torch.tensor([sigma])
        self.sigma = Variable(self.sigma, requires_grad=require_grad)

        self.delta = float(max - min) / float(bins)
        self.centers = nn.Parameter(float(min) + self.delta * (torch.arange(bins).float() + 0.5), requires_grad=False)

    def forward(self, x, attention_mask=None):
        device = x.device
        self.sigma = self.sigma.to(device)
        self.centers = self.centers.to(device)

        x = torch.unsqueeze(x, dim=1) - torch.unsqueeze(self.centers, 1)
        hist_dist = torch.exp(-0.5 * (x / self.sigma) ** 2) / (self.sigma * np.sqrt(np.pi * 2)) * self.delta
        # multiply with attention mask
        if not type(attention_mask) == type(None):
            hist_dist *= torch.unsqueeze(attention_mask, 1)

        hist = hist_dist.sum(dim=-1)
        hist = hist / torch.sum(hist, dim=1, keepdim=True)

        return hist, hist_dist

class HistogramLayerLocal(nn.Module):
    def __init__(self):
        super().__init__()
        self.hist_layer = GaussianHistogram(bins=256, min=-1., max=1., sigma=0.01, require_grad=False)
        self.channels = 3

    def forward(self, ref, attention_mask=None):
        if not type(attention_mask) == type(None):
            attention_mask = torch.flatten(attention_mask[:, 0, :, :], start_dim=1, end_dim=-1)

        layers = []
        for i in range(self.channels):
            input_channel = torch.flatten(ref[:, i, :, :], start_dim=1, end_dim=-1)
            input_hist, hist_dist = self.hist_layer(input_channel, attention_mask)
            hist_dist = hist_dist.view(-1, 256, ref.shape[2], ref.shape[3])
            layers.append(hist_dist)

        return torch.cat(layers, 1)


class HistFusionModule(nn.Module):
    """
    Global pooling fused with histogram
    """

    def __init__(self, in_features, out_features, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)
        self.normalization = norm_layer(out_features)
        self.activation = nn.ReLU()

    def forward(self, feature):
        feature = self.conv(feature)
        feature = self.normalization(feature)
        return self.activation(feature)
    
    
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

class FFCNLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, max_features=512,
                 init_conv_kwargs={}, conv_kwargs={}):
        super(FFCNLayerDiscriminator, self).__init__()
        self.n_layers = n_layers

        def _act_ctor(inplace=True):
            return nn.LeakyReLU(negative_slope=0.2, inplace=inplace)

        kw = 3
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[FFC_BN_ACT(input_nc, ndf, kernel_size=kw, padding=padw, norm_layer=norm_layer,
                                activation_layer=_act_ctor, ratio_gin=0, ratio_gout=0, **init_conv_kwargs)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, max_features)

            cur_model = [
                FFC_BN_ACT(nf_prev, nf,
                           kernel_size=kw, stride=2, padding=padw,
                           norm_layer=norm_layer,
                           activation_layer=_act_ctor,
                           ratio_gin=0, 
                           ratio_gout=0, 
                           **conv_kwargs)
            ]
            sequence.append(cur_model)

        nf_prev = nf
        nf = min(nf * 2, 512)

        cur_model = [
            FFC_BN_ACT(nf_prev, nf,
                       kernel_size=kw, stride=1, padding=padw,
                       norm_layer=norm_layer,
                       ratio_gin=0, 
                       ratio_gout=0, 
                       activation_layer=lambda *args, **kwargs: nn.LeakyReLU(*args, negative_slope=0.2, **kwargs),
                       **conv_kwargs),
            ConcatTupleLayer()
        ]
        sequence.append(cur_model)

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        for n in range(len(sequence)):
            setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

    def get_all_activations(self, x):
        res = [x]
        for n in range(self.n_layers + 2):
            model = getattr(self, 'model' + str(n))
            res.append(model(res[-1]))
        return res[1:]

    def forward(self, x):
        act = self.get_all_activations(x)
        feats = []
        for out in act[:-1]:
            if isinstance(out, tuple):
                if torch.is_tensor(out[1]):
                    out = torch.cat(out, dim=1)
                else:
                    out = out[0]
            feats.append(out)
        return [feats]



from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

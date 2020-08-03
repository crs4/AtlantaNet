import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import functools


class Resnet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super(Resnet, self).__init__()
        ##assert backbone in ENCODER_RESNET
        self.encoder = getattr(models, backbone)(pretrained=pretrained)
        del self.encoder.fc, self.encoder.avgpool

    def forward(self, x):
        features = []
        x = self.encoder.conv1(x) ###block0
        x = self.encoder.bn1(x) ###block0
        x = self.encoder.relu(x) ###block0
        x = self.encoder.maxpool(x) ###block0
                
        x = self.encoder.layer1(x);  features.append(x)  #
        x = self.encoder.layer2(x);  features.append(x)  #
        x = self.encoder.layer3(x);  features.append(x)  #
        x = self.encoder.layer4(x);  features.append(x)  #
        return features

    def list_blocks(self):
        lst = [m for m in self.encoder.children()]
        block0 = lst[:4]
        block1 = lst[4:5]
        block2 = lst[5:6]
        block3 = lst[6:7]
        block4 = lst[7:8]
        return block0, block1, block2, block3, block4

class ConvCompressWH(nn.Module):
    ''' Reduce feature height and width by factor of two '''
    def __init__(self, in_c, out_c, ks=3, st=2):
        super(ConvCompressWH, self).__init__()
        assert ks % 2 == 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=ks, stride=(st, st), padding=ks//2), ###NEW modifing stride
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        ##print('compWH shape', x.shape)
        return self.layers(x)

class ReshapeConv(nn.Module):
    def __init__(self, in_c, out_c, f1, f2 ,f3):
        ###reduce by 2^f
        super(ReshapeConv, self).__init__()
        self.layer = nn.Sequential(
            ConvCompressWH(in_c, in_c, 3, 1+f1),
            ConvCompressWH(in_c, out_c, 3, 1+f2),
            ConvCompressWH(out_c, out_c,3, 1+f3) )

    def forward(self, x):
        x = self.layer(x)
        return x

class ReshapeConv3(nn.Module):
    def __init__(self, in_c, out_c):
        ###reduce by 2^0
        super(ReshapeConv3, self).__init__()
        self.layer = nn.Sequential(
            ConvCompressWH(in_c, out_c, 3, 1) )

    def forward(self, x):
        ##print('ReshapeConv3 x ',x.shape)
        x = self.layer(x)        
        ##print('ReshapeConv3 shape', x.shape)
        return x 
   
class MergeE2PFeatures2Seq(nn.Module):
    def __init__(self, out_fets): 
        ''' Process 4 blocks from encoder to single multiscale features '''
        super(MergeE2PFeatures2Seq, self).__init__()
        
        self.out_fets = out_fets
        
        self.ghc_lst = nn.ModuleList([
            ##################################
            ReshapeConv(256, self.out_fets//4,1,1,1),
            ReshapeConv(512, self.out_fets//4,0,1,1),
            ReshapeConv(1024, self.out_fets//4,0,0,1),
            ReshapeConv(2048, self.out_fets//4,0,0,0)
            ])

        
    def forward(self, conv_list, out_wh):
        assert len(conv_list) == 4
        
        bs = conv_list[0].shape[0]
                
        feature = torch.cat([
            f(x).reshape(bs, self.out_fets//4, out_wh)
            for f, x in zip(self.ghc_lst, conv_list)
        ], dim=1)
                
        return feature

class AtlantaNet(nn.Module):
    x_mean = torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None])
    x_std = torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None])

    def __init__(self, backbone, use_gpu = True):
        super(AtlantaNet, self).__init__()
        self.backbone = backbone
        self.use_gpu = use_gpu
        self.fp_size = 1024
        self.rnn_hidden_size = 512
        self.feature_count = 1024        
        self.out_size = 32  
        self.up_mode = 'nearest'
               
        self.feature_extractor = Resnet(backbone, pretrained=True)                
                                      
        ###merge features at different level of detail -  feature_count to reduce_wh_module init  
        self.reduce_wh_module = MergeE2PFeatures2Seq(out_fets = self.feature_count)
        
        self.bi_rnn = nn.LSTM(input_size=self.feature_count,
                              hidden_size=self.rnn_hidden_size,
                              num_layers=2,
                              dropout=0.5,
                              batch_first=False,
                              bidirectional=True)

        self.drop_out = nn.Dropout(0.5)         
        
        self.decoder = nn.ModuleList([
            ReshapeConv3(2 * self.rnn_hidden_size, 256),
            ReshapeConv3(256, 128),
            ReshapeConv3(128, 64),
            ReshapeConv3(64, 32),
            ReshapeConv3(32, 1),
            ])    

        self.x_mean.requires_grad = False
        self.x_std.requires_grad = False
        wrap_lr_pad(self)

    def _prepare_x(self, x):
        if self.x_mean.device != x.device:
            self.x_mean = self.x_mean.to(x.device)
            self.x_std = self.x_std.to(x.device)
        return (x[:, :3] - self.x_mean) / self.x_std
        
    def forward(self, x):  
        ####prepare data using mean and std
        x = self._prepare_x(x)
        
        #### encode features
        conv_list = self.feature_extractor(x)
                          
        seq_count = (x.shape[3]//32)**2 ####to reduce_wh_module forward
       
        ##merge features and covert to 1D sequence
        feature = self.reduce_wh_module(conv_list, seq_count) ### 1X1024x1024: wxh 

        feature = feature.permute(2, 0, 1)  # [w*h, b, layers] ### eg. 256 x b x 1024
        ##print('feature in to rnn',feature.shape)
        output, hidden = self.bi_rnn(feature)  # [seq_len, b, num_directions * hidden_size] -> 256x b x (2*512)
                    
        output = self.drop_out(output)                      
                              
        output = output.permute(1, 2, 0)  # [b, 1, seq_len]
                                                                  
        mask = output.reshape(output.shape[0], output.shape[1], self.out_size, self.out_size)
                                
        ###DECODER last step
        for i, conv in enumerate(self.decoder):
            mask = F.interpolate(mask, scale_factor=(2,2), mode=self.up_mode)
            mask = conv(mask)
                            
        mask = mask.squeeze(1)
                                          
        return mask

################model end###############################################

############utilities and testing


def lr_pad(x, padding=1):
    ''' Pad left/right-most to each other instead of zero padding '''
    return torch.cat([x[..., -padding:], x, x[..., :padding]], dim=3)

class LR_PAD(nn.Module):
    ''' Pad left/right-most to each other instead of zero padding '''
    def __init__(self, padding=1):
        super(LR_PAD, self).__init__()
        self.padding = padding

    def forward(self, x):
        return lr_pad(x, self.padding)

def wrap_lr_pad(net):
    for name, m in net.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if m.padding[1] == 0:
            continue
        w_pad = int(m.padding[1])
        m.padding = (m.padding[0], 0)
        names = name.split('.')
        root = functools.reduce(lambda o, i: getattr(o, i), [net] + names[:-1])
        setattr(
            root, names[-1],
            nn.Sequential(LR_PAD(w_pad), m)
        )

if __name__ == '__main__':
    print('testing 2D map AtlantaNet')

    device = torch.device('cuda')

    net = AtlantaNet('resnet50', use_gpu = True).to(device)

    fp_size = net.fp_size

    print('transform size',fp_size)

    pytorch_total_params = sum(p.numel() for p in net.parameters())

    print('pytorch_total_params', pytorch_total_params)

    pytorch_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print('pytorch_trainable_params', pytorch_trainable_params)

    batch = torch.ones(2, 3, fp_size, fp_size).to(device)

    mask = net(batch)
               
    print('mask shape', mask.shape)

    print('test done')
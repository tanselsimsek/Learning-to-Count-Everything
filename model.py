import torch, torchvision
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Wide_Resnet50_2(nn.Module):
    def __init__(self):
        super(Wide_Resnet50_2, self).__init__()
        #Loading pre-trained Resnet 50
        self.resnet = torchvision.models.wide_resnet50_2(pretrained= True)
        #list of all the children layer of resnet 50
        children = list(self.resnet.children())
        #creating a new sequential model with children 4 as the first layer
        self.conv1 = nn.Sequential(*children[:4])
        self.conv2 = children[4]
        self.conv3 = children[5]
        self.conv4 = children[6]
    def forward(self, im_data):
        feat = OrderedDict()
        #generating different features maps according to the convolutional layer
        feat_map = self.conv1(im_data)
        feat_map = self.conv2(feat_map)
        feat_map3 = self.conv3(feat_map)
        feat_map4 = self.conv4(feat_map3)
        #storing activation maps into an ordered dict
        feat['map3'] = feat_map3
        feat['map4'] = feat_map4
        return feat

class VGG16FPN(nn.Module):
    def __init__(self):
        super(VGG16FPN, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        children = list(vgg16.children())[0]
        self.conv1 = nn.Sequential(children[:5])
        self.conv2 = nn.Sequential(children[5:10])
        self.conv3 = nn.Sequential(children[10:14])
        self.conv4 = nn.Sequential(children[14:19])
        self.conv5 = nn.Sequential(children[19:24])
        self.conv6 = nn.Sequential(children[24:])
    def forward(self, im_data):
        feat = OrderedDict()
        #generating different features maps according to the convolutional layer
        feat_map = self.conv1(im_data)
        feat_map = self.conv2(feat_map)
        feat_map = self.conv3(feat_map)
        feat_map = self.conv4(feat_map)
        feat_map3 = self.conv5(feat_map)
        feat_map4 = self.conv6(feat_map3)
        #storing activation maps into an ordered dict
        feat['map3'] = feat_map3
        feat['map4'] = feat_map4
        return feat
        
class Resnet50FPN(nn.Module):
    def _init_(self):
        super(Resnet50FPN, self)._init_()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        children = list(self.resnet.children())
        self.conv1 = nn.Sequential(*children[:4])
        self.conv2 = children[4]
        self.conv3 = children[5]
        self.conv4 = children[6]
    def forward(self, im_data):
        feat = OrderedDict()
        feat_map = self.conv1(im_data)
        feat_map = self.conv2(feat_map)
        feat_map3 = self.conv3(feat_map)
        feat_map4 = self.conv4(feat_map3)
        feat['map3'] = feat_map3
        feat['map4'] = feat_map4
        return feat

class CountRegressor(nn.Module):
    def __init__(self, input_channels,pool='mean'):
        super(CountRegressor, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, 7, padding=3),
            nn.ReLU(),
            #Applies a 2D bilinear upsampling to an input signal composed of several input channels
            #given a tensor of size 2, UpsamplingBilinear2d(scale_factor=2) return an array of size 4
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )

    def forward(self, im):
        num_sample =  im.shape[0]
        if num_sample == 1:
            output = self.regressor(im.squeeze(0))
            #Average Pooling is a pooling operation that calculates the average value for patches of a feature map
            if self.pool == 'mean':
                output = torch.mean(output, dim=(0),keepdim=True)  
                return output
            #Usual max-pooling operation over the features map
            elif self.pool == 'max':
                output, _ = torch.max(output, 0,keepdim=True)
                return output
        else:
            for i in range(0,num_sample):
                output = self.regressor(im[i])
                if self.pool == 'mean':
                    output = torch.mean(output, dim=(0),keepdim=True)
                elif self.pool == 'max':
                    output, _ = torch.max(output, 0,keepdim=True)
                if i == 0:
                    Output = output
                else:
                    Output = torch.cat((Output,output),dim=0)
            return Output

#The following two function are used to weight init
def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):                
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def weights_xavier_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
            
            
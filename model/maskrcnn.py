from turtle import back, forward
import types
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.ops import RoIAlign
import config
import torch
from fpn_rpn import my_unet_backbone,RPN


def backbone_forward(self,x):
    x,x1 = self.down1(x)
    x,x2 = self.down2(x)
    x,x3 = self.down3(x)
    x =self.mid(x)
    x3 = self.ag1(x,x3)
    x,p2 = self.up1(x,x3)
    x2 = self.ag2(x,x2)
    x = self.up2(x,x2)
    x = self.out(x,x1)
    return x,p2

def up1_forward(self,x,path):
    x = torch.cat([x,path],dim=1)
    x = self.conv1(x)
    x = self.conv2(x)+x
    return self.up(x),x

def out_forward(self,x,path):
    x = torch.cat([x,path],dim=1)
    x = self.conv1(x)
    x = self.conv2(x)+x
    return x

class mask_rcnn_unet_backbone(nn.Module):
    def __init__(self,in_c,out_c,pre_trained:bool=False) -> None:
        super().__init__()
        self.rpn = RPN(256,9)
        self.roialign =RoIAlign((1,out_c,5,5),spatial_scale=4)
        self.mask_layer = 
        self.reg_layer = 
    
    def forward(self,x):


    
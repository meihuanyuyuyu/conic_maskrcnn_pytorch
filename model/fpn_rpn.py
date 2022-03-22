from typing import Optional
import torch
from torchvision.models import resnet
import types
import torch.nn as nn

def init_weight(m):
    if  isinstance(m,nn.Conv2d):
        nn.init.kaiming_normal_(m.weight,mode='fan_in')
    if isinstance(m,nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight,mode='fan_in')

class bn_active_conv(nn.Module):
    def __init__(self,in_c,out_c,kernel_size,stride,padding,active_f:nn.Module=nn.LeakyReLU) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_c),
            active_f(0.1),
            nn.Conv2d(in_c,out_c,kernel_size,stride=stride,padding=padding)
        )
        
    def forward(self,x):
        return self.conv(x)

class Down_block(nn.Module):
    def __init__(self,in_c,out_c,active_f:nn.Module=nn.LeakyReLU) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c,out_c,3,1,1)
        self.conv2 = nn.Sequential(
            bn_active_conv(out_c,out_c,3,1,1,active_f),
            bn_active_conv(out_c,out_c,3,1,1,active_f)
        )
        self.pool = nn.MaxPool2d(2)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x) +x
        return self.pool(x),x

class Up_block(nn.Module):
    def __init__(self,in_c,out_c,active_f:nn.Module=nn.LeakyReLU) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c,out_c,3,1,1)
        self.conv2 = nn.Sequential(
            bn_active_conv(out_c,out_c,3,1,1,active_f),
            bn_active_conv(out_c,out_c,3,1,1,active_f)
        )
        self.up = nn.ConvTranspose2d(out_c,out_c//2,2,2)
        
    def forward(self,x,path):
        x = torch.cat([x,path],dim=1)
        x = self.conv1(x)
        x = self.conv2(x)+x
        return self.up(x)

class attention_gate(nn.Module):
    def __init__(self,in_c,active_f) -> None:
        super().__init__()
        self.w_g = bn_active_conv(in_c,in_c,1,1,0,active_f=active_f)
        self.w_path = bn_active_conv(in_c,in_c,1,1,0,active_f=active_f)
        self.att = nn.Sequential(
            bn_active_conv(in_c,in_c,1,1,0,active_f),
            nn.Sigmoid()
        )
        
    def forward(self,gate,path):
        gate = self.w_g(gate)
        path = self.w_path(path)
        path = self.att(gate+path)
        return path

class mid_bridge(Up_block):
    def __init__(self, in_c, out_c, active_f: nn.Module = nn.LeakyReLU) -> None:
        super().__init__(in_c, out_c, active_f)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)+x
        return self.up(x)

class out_layer(Up_block):
    def __init__(self, in_c, out_c, num_class, active_f: nn.Module = nn.LeakyReLU) -> None:
        super().__init__(in_c, out_c, active_f)
        self.conv3 = nn.Conv2d(out_c,num_class,1,1)
        
    def forward(self,x,path):
        x = torch.cat([x,path],dim=1)
        x = self.conv1(x)
        x = self.conv2(x)+x
        return self.conv3(x)


class my_unet_backbone(nn.Module):
    def __init__(self,in_c,num_class,pre_train=False) -> None:
        super().__init__()
        self.pretrain =pre_train
        self.feature = [64,128,256,512]
        self.down1 = Down_block(in_c,64)
        self.down2 =Down_block(64,128)
        self.down3 =Down_block(128,256)
        self.mid = mid_bridge(256,512)
        self.up1 = Up_block(512,256)
        self.ag1 = attention_gate(256,nn.LeakyReLU)
        self.up2 = Up_block(256,128)
        self.ag2 = attention_gate(128,nn.LeakyReLU)
        self.out = out_layer(128,64,num_class=num_class)
        
    def forward(self,x):
        x,x1 = self.down1(x)
        x,x2 = self.down2(x)
        x,x3 = self.down3(x)
        x =self.mid(x)
        x3 = self.ag1(x,x3)
        x = self.up1(x,x3)
        x2 = self.ag2(x,x2)
        x = self.up2(x,x2)
        x = self.out(x,x1)
        return x

class RPN(nn.Module):
    def __init__(self,in_c,k) -> None:
        super().__init__()
        self.k = k
        self.conv = bn_active_conv(in_c,256,3,1,1)
        self.conv_cls = nn.Conv2d(256,k,1,1)
        self.conc_reg = nn.Conv2d(256,4*k,1,1)

    def forward(self,p2):
        p2 = self.conv(p2)
        cls = self.conv_cls(p2)
        reg = self.conc_reg(p2)
        return cls,reg

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

class fpn_rpn(nn.Module):
    def __init__(self,in_c,pre_trained) -> None:
        super().__init__()
        self.backbone = my_unet_backbone(in_c,7,pre_trained)
        if pre_trained:
            self.backbone.load_state_dict(torch.load('model_parameters/7_EM_model.pt'))
            self.backbone.requires_grad_(False)
        self.backbone.up1.forward =types.MethodType(up1_forward,self.backbone.up1)
        self.backbone.out.forward =types.MethodType(out_forward,self.backbone.out)
        self.backbone.forward = types.MethodType(backbone_forward,self.backbone)
        self.rpn = RPN(256,28)
    
    def forward(self,x):
        x,p2 = self.backbone(x)
        cls,reg = self.rpn(p2)
        cls = torch.sigmoid(cls)
        return x,cls,reg
        
import torch
import numpy as np
import utils
import config
from torchvision.utils import draw_bounding_boxes,save_image
from tqdm import tqdm
from utils import general_anchors,apply_box_delt,nms
from augmentation import Normalize_
import torch.nn.functional as nnf
from model.fpn_rpn import  fpn_rpn
from torch.utils.data import DataLoader,Subset
from conic_dataset import conic_stage1_data
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

labels = (torch.from_numpy(np.load('project_conic/CoNIC_Challenge/labels.npy')[:,:,:,0].astype(np.float64))).float() # 4981,h,w
g_anchors,height,width,k = utils.general_anchors(labels[0],config.anchors,4) 
g_anchors = g_anchors.to('cuda')
data_set = conic_stage1_data([Normalize_([0.5,0.5,0.5],[0.5,0.5,0.5])])
train_index = torch.load('splitted_indexes.pt')['train']
test_index = torch.load('splitted_indexes.pt')['test']
train_set,test_set = DataLoader(Subset(data_set,train_index),batch_size=8,shuffle=1,num_workers=0),DataLoader(Subset(data_set,test_index),batch_size=8,shuffle=1,num_workers=0)
net = fpn_rpn(3,True).to('cuda')
opt = SGD(net.parameters(),lr=7e-3,momentum=0.9,weight_decay=1e-4)
lr_c = MultiStepLR(opt,[15,25,35],0.5)
for _ in range(50):
    bar = tqdm(train_set)
    for data in bar:
        opt.zero_grad()
        imgs,positive,negative,reg = data
        imgs= imgs.to('cuda')
        x,cls,reg_p = net(imgs)
        reg_p = reg_p.reshape(-1,4,28,64,64)
        loss_cls = -((positive*torch.log(cls.clip(0.001,0.99))).mean()+(negative*torch.log((1-cls).clip(0.01,0.99))).mean())
        loss_reg = nnf.smooth_l1_loss(reg_p,reg)
        loss = loss_cls+loss_reg
        loss.backward()
        opt.step()
        bar.set_description(f'loss:{loss.item()}')
    
    with torch.no_grad():
        bar =tqdm(enumerate(test_set))
        for _,data in bar:
            imgs,positive,negative,reg = data
            imgs = imgs.to('cuda')
            x,cls,reg_p = net(imgs)
            acc_t=(cls>=0.5)[positive].sum()/positive.sum().clip(0.001)
            acc_f=(cls<0.5)[negative].sum()/negative.sum().clip(0.001)
            bar.set_description(f' acc_t:{acc_t},acc_f:{acc_f}')
            if _ % 40 ==39:
                score =cls[0]
                cls = cls[0].flatten()>=0.5
                anchors = g_anchors[cls]
                delta = reg_p[0].view(4,-1).T[cls]
                anchors = apply_box_delt(anchors,delta)
                index = nms(anchors,score[score>=0.5],0.15)
                anchors=anchors[index]
                imgs = (imgs+1)/2
                save_image(imgs,'rpn_image.png')
                imgs = torch.tensor(imgs*255,dtype=torch.uint8)
                imgs = (draw_bounding_boxes(imgs[0],anchors)/255).float()
                save_image(imgs,'rpn_roi.png')
                torch.save(net.state_dict(),'model_parameters/stage_1/fpn_rpn.pt')
    

    

'''
cls:(n,k,h,w)
reg:(n,4,k,h,w)

'''
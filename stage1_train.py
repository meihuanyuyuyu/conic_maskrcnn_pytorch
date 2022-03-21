import torch
import json
from tqdm import tqdm
from augmentation import Normalize_
import torch.nn.functional as nnf
from model.fpn_rpn import  fpn_rpn
from torch.utils.data import DataLoader,Subset
from conic_dataset import conic_stage1_data
from torch.optim import SGD

data_set = conic_stage1_data([Normalize_([0.5,0.5,0.5],[0.5,0.5,0.5])])
train_index = torch.load('splitted_indexes.pt')['train']
test_index = torch.load('splitted_indexes.pt')['test']
train_set,test_set = DataLoader(Subset(data_set,train_index),batch_size=8,shuffle=1,num_workers=0),DataLoader(Subset(data_set,test_index),batch_size=8,shuffle=1,num_workers=0)
net = fpn_rpn(3,True).to('cuda')
opt = SGD(net.parameters(),lr=1e-2,momentum=0.9,weight_decay=1e-4)
for _ in range(50):
    bar = tqdm(train_set)
    for data in bar:
        opt.zero_grad()
        imgs,positive,negative,reg = data
        imgs= imgs.to('cuda')
        x,cls,reg_p = net(imgs)
        cls = cls.squeeze(1)
        reg_p = reg_p.reshape(-1,4,9,64,64)
        loss_cls = -(positive*torch.log(cls.clip(0.01,0.99))+negative*torch.log((1-cls).clip(0.01,0.99))).mean()
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
            cls = cls.squeeze(1)
            acc_p = (cls>=0.5)[positive].sum() /positive.sum()
            acc_f = (cls<0.5)[negative].sum() / negative.sum()
            bar.set_description(f' acc_p:{acc_p.item()},acc_f:{acc_p.item()}')
            if _ % 20 ==1:
                torch.save(net.state_dict(),'model_parameters/stage_1/fpn_rpn.pt')
    

    

'''
cls:(n,k,h,w)
reg:(n,4,k,h,w)

'''
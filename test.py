from cProfile import label
import torch
import os
from tqdm import tqdm
import numpy as np
from generate_labels import masks_to_box
from torchvision.ops import box_convert,remove_small_boxes

def random_splite_train_test_index():
    r'划分数据集'
    a = os.listdir('project_conic/CoNIC_Challenge/stage1_labels')
    length = len(a)
    train_index,test_index= torch.randperm(length).split((4*length)//5) # 随机划分索引
    dic = {'train':train_index,'test':test_index}
    torch.save(dic,'splitted_indexes.pt')


def print_label_quality(i:int):
    r'检查生成锚框正例标签数量与实际标签数量'
    b = (torch.from_numpy(np.load('project_conic/CoNIC_Challenge/labels.npy').astype(np.float64)))[i,:,:,0]
    a = os.listdir('project_conic/CoNIC_Challenge/stage1_labels')
    path = os.path.join('project_conic/CoNIC_Challenge/stage1_labels',a[i])
    dic = torch.load(path)
    label = b.max()-1
    total = dic['false'].numel()
    true = dic['true'].sum()
    false = dic['false'].sum()
    print(total,true,label,false)

def cluster():
    r'聚类生成锚框30个'
    labels = (torch.from_numpy(np.load('project_conic/CoNIC_Challenge/labels.npy').astype(np.float64)))[...,0].to('cuda')
    a = torch.zeros((1,2),device='cuda')
    for _ in labels:
        if _.max() == 0:
            continue
        _ = masks_to_box(_)
        index = remove_small_boxes(_,2)
        _ = _[index]
        _=box_convert(_,'xyxy','xywh')[:,2:]
        a=torch.cat([a,_],dim=0)
    a = a[1:]
    print(a.max(),a.min())
    wh = torch.linspace(4,27,steps=30,device='cuda').unsqueeze(1)
    wh = torch.cat([wh,wh],dim=1)
    bar = tqdm(range(100))
    for _ in bar:
        dic = (a.unsqueeze(1)-wh).norm(p=2,dim=2)
        dic = torch.argmin(dic,dim=-1)
        for i in range(30):
            wh[i] = a[dic==i].mean(dim=0)
    print(wh)


print_label_quality(50)



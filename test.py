import torch
import os
import time


'''a = os.listdir('project_conic/CoNIC_Challenge/stage1_labels')
length = len(a)
train_index,test_index= torch.randperm(length).split((4*length)//5) # 随机划分索引
dic = {'train':train_index,'test':test_index}
torch.save(dic,'splitted_indexes.pt')'''


a = os.listdir('project_conic/CoNIC_Challenge/stage1_labels')
for _ in a:
    path = os.path.join('project_conic/CoNIC_Challenge/stage1_labels',_)
    dic = torch.load(path)
    total = dic['false'].numel()
    true = dic['false'].sum()
    print(total,true)
    time.sleep(1)

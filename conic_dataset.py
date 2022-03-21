from matplotlib import transforms
from torch.utils.data import Dataset,DataLoader
import torch
import os
import numpy as np

class conic_unet_data(Dataset):
    def __init__(self,transforms:list) -> None:
        super().__init__()
        self.transfroms = transforms
        self.imgs = (torch.from_numpy(np.load('project_conic/CoNIC_Challenge/images.npy'))//255).permute(0,3,1,2).float()
        self.labels = (torch.from_numpy(np.load('project_conic/CoNIC_Challenge/labels.npy').astype(np.float32))).permute(0,3,1,2)[:,1:]
        
    def __len__(self):
        return self.imgs.size(0)

        
    def _transfrom(self,imgs,labels):
        if self.transfroms is not None:
            for _ in self.transfroms:
                imgs,labels = _(imgs,labels)
            return imgs,labels
        else:
            return imgs,labels

    def __getitem__(self, index):
        return self._transfrom(self.imgs[index],self.labels[index])

class conic_stage1_data(Dataset):
    def __init__(self,transf:list=[]) -> None:
        super().__init__()
        self.transf = transf
        self.imgs = (torch.from_numpy(np.load('project_conic/CoNIC_Challenge/images.npy').astype(np.float64))//255).permute(0,3,1,2).float()
        self.labels = os.listdir('project_conic/CoNIC_Challenge/stage1_labels')
        self.path = 'project_conic/CoNIC_Challenge/stage1_labels'

    def _transfrom(self,imgs,t,f,d):
        if self.transf is not None:
            for _ in self.transf:
                imgs,t,f,d = _(imgs,t,f,d)
            return imgs,t,f,d
        else:
            return imgs,t,f,d

    def __getitem__(self, index):
        label =torch.load(os.path.join(self.path,self.labels[index]))
        index = int(self.labels[index][:-3])
        return self._transfrom(self.imgs[index],label['true'],label['false'],label['t'])
    


        
        
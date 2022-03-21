from os import access
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import torch.nn as nn
from conic_dataset import conic_unet_data
from model.fpn_rpn import my_unet_backbone
from augmentation import Randomrotation_vh,Normalize_
import torch.nn.functional as nnf
import json
from torch.utils.data import Subset,DataLoader



def val_train(model: nn.Module, sampled_data: DataLoader, device: str = 'cuda',metric=metric.multi_label_accuracy):
	scores = []
	model.eval()
	bar = tqdm(enumerate(sampled_data))
	for _,data in bar:
		imgs, labels = data
		labels = labels.squeeze(1).to(device,dtype=torch.long)
		#labels_2 = labels_2.squeeze(1).to(device,dtype=torch.long)
		outputs = model(imgs.to(device))
		binarymap = torch.argmax(outputs, dim=1)
		#binarymap2 = torch.argmax(outputs[:,7:,...],dim=1)
		score = metric(binarymap, labels)
		scores.append(score)
		bar.set_description(f'val 1 epoch accuracy:{score.item()}')
		if _%20 ==0:
			visualization(binarymap, labels, imgs, 'val.png')


indexes = json.load(open('project_conic/CoNIC_Challenge/indexes.json', 'r'))
train_indexes = indexes['train_indexes']
test_indexes = [x for x in range(4981) if x not in train_indexes]
train_data = conic_unet_data([Randomrotation_vh(),Normalize_([0.5,0.5,0.5],[0.5,0.5,0.5])])
test_data = conic_unet_data(Normalize_([0.5,0.5,0.5],[0.5,0.5,0.5]))
train_data,test_data = Subset(train_data,train_indexes),Subset(test_data,test_indexes)
train_s,test_s = DataLoader(train_data,8,True,num_workers=8),DataLoader(test_data,8,True,num_workers=8)
net = my_unet_backbone(3,7).to('cuda')
opt = SGD(net.parameters(),lr=8e-2,momentum=0.9,weight_decay=1e-4)
criterion = nnf.cross_entropy
lr_schedule = MultiStepLR(opt,[15,20],gamma=0.5)
loss_all=[]
acc_t =[]
acc_val =[]


	






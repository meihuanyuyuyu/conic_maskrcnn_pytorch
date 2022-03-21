import torch
import utils
from torch.nn.functional import one_hot
import numpy as np
import config
import time
from tqdm import tqdm
from torchvision.ops import remove_small_boxes



def masks_to_box(input:torch.Tensor):
    'input:(h,w)'
    num_classes = int((input.max()+1).item())
    input = one_hot(input.to(dtype=torch.int64),num_classes).permute(2,0,1)[1:,...] # (n,h,w)
    boxes = torch.zeros((num_classes-1,4),device=input.device)
    for _,mask in enumerate(input):
        xy =torch.nonzero(mask)
        boxes[_,:2]=xy.argmin(0)
        boxes[_,2:]=xy.argmax(0)
    return boxes




labels = (torch.from_numpy(np.load('project_conic/CoNIC_Challenge/labels.npy')[:,:,:,0].astype(np.float64))).float() # 4981,h,w
g_anchors,height,width,k = utils.general_anchors(labels[0],config.anchors,4) 
labels = labels.to('cuda') #256,256
bar = tqdm(enumerate(labels))
for _,label in bar:
    if label.max() == 0:
        continue
    label = masks_to_box(label)
    index = remove_small_boxes(label,2)
    if index.shape[0] ==0:
        continue
    label = label[index]
    dict=utils.generate_box_labels(g_anchors,label,height//4,width//4,k)
    torch.save(dict,f'project_conic/CoNIC_Challenge/stage1_labels/{_}.pt')
    break




import torch
import utils
from torch.nn.functional import one_hot
import numpy as np
import config
import time
from tqdm import tqdm
from torchvision.utils import draw_bounding_boxes,save_image
from torchvision.ops import remove_small_boxes


def masks_to_box(input:torch.Tensor):
    'input:(h,w)'
    num_classes = int((input.max()+1).item())
    input = one_hot(input.to(dtype=torch.int64),num_classes).permute(2,0,1)[1:,...] # (n,h,w)
    boxes = torch.zeros((num_classes-1,4),device=input.device)
    for _,mask in enumerate(input):
        xy =torch.nonzero(mask)
        boxes[_,:2],a=xy.min(dim=0)
        boxes[_,2:],a=xy.max(0)
    return boxes


#img = (torch.from_numpy(np.load('project_conic/CoNIC_Challenge/images.npy').astype(np.float64))/255).permute(0,3,1,2)[0]
#img =  torch.tensor(img,dtype=torch.uint8)
labels = (torch.from_numpy(np.load('project_conic/CoNIC_Challenge/labels.npy')[:,:,:,0].astype(np.float64))).float() # 4981,h,w
g_anchors,height,width,k = utils.general_anchors(labels[0],config.anchors,4) 
#print(g_anchors)
#pic = (draw_bounding_boxes(img,g_anchors[999:1000])/255).float()
#save_image(pic,'anchor_boxes_in_img.png')
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
    #print(label.size(0))
    dict=utils.generate_box_labels(g_anchors,label,height//4,width//4,k)
    torch.save(dict,f'project_conic/CoNIC_Challenge/stage1_labels/{_}.pt')




import torch
from torchvision.ops import clip_boxes_to_image,box_convert,box_iou,remove_small_boxes,nms




def general_anchors(label:torch.Tensor,anchor_wh:torch.Tensor,rate=4)->torch.Tensor:
    r'输入图片chw，anchor_wh:(k,2),return:(num,4)锚框坐标,hw为输入特征图的倍率大小，用作源图片需要+0.5放大4倍'
    '''    
    h = imgs.size(2)/rate
    w = imgs.size(3)/rate
    k = anchor_wh[0]
    for i in range(h):
        for j in range(w):
            one_position = torch.tensor([[i,j,i,j]for _ in range(k)])
            one_position[:,:2] = i - anchor_wh*0.5
            one_position[:,2:] = one_position + anchor_wh*0.5
            anchors = torch.stack(one_position,dim=0)
    anchors = clip_boxes_to_image(anchors,[h,w])'''
    anchor_wh = anchor_wh/rate
    h = label.size(0)
    w = label.size(1)
    k = anchor_wh.size(0)
    location = torch.ones((h//rate,w//rate),device=anchor_wh.device)
    location = torch.nonzero(location) # [(0,0),(0,1)...] (hw,2)
    xy1 = location- anchor_wh.unsqueeze(1)*0.5 # ->k,hw,2
    xy2 = location + anchor_wh.unsqueeze(1)*0.5
    anchors = torch.cat([xy1,xy2],dim=-1).view(-1,4)
    anchors = (anchors+0.5) * 4
    return  anchors,h,w,k



def apply_box_delt(boxes:torch.Tensor,deltas:torch.Tensor)->torch.Tensor:
    r'boxes(rois,x1,y1,x2,y2),deltas(rois,dx,dy,log(dw),log(dh))  return:(rois,4)'
    boxes = box_convert(boxes,'xyxy','xywh')
    boxes[:,2]= boxes[:,2:]*deltas[:,:2] + boxes[:,:2]
    boxes[:,2:] = boxes[:,2:]*torch.exp(deltas[:,2:])
    return boxes
    


def generate_box_labels(g_anchors:torch.Tensor,ground_truth:torch.Tensor,h,w,k,threshold:float=0.7)->dict:
    r'g_a:(hwk,4),gt:(n,4)'
    #print('gt.size()',ground_truth.size())
    ious = box_iou(g_anchors,ground_truth).view(k,h,w,-1)# (khw,num)iou
    positive = (ious>threshold).sum(-1).bool() # (h,w,k) 在生成框里有正例
    negative = (ious<(1-threshold)).sum(-1)==ground_truth.size(0) # (h,w,k)
    g_anchors = box_convert(g_anchors,'xyxy','xywh').view(k,h,w,-1)
    ground_truth = box_convert(ground_truth,'xyxy','xywh')
    #print(g_anchors.size(),g_anchors[...,2:].max(),ground_truth[...,2:].min())
    max_iou_index = torch.argmax(ious,-1)
    max_iou = ground_truth[max_iou_index]# (k,h,w,,4)
    dxdy = (max_iou[...,:2] -g_anchors[...,:2])/g_anchors[...,2:]
    dwdh = torch.log(max_iou[...,2:] /g_anchors[...,2:])
    regression = torch.cat([dxdy,dwdh],dim=-1).permute(3,0,1,2)
    dict = {'true':positive,'false':negative,'t':regression}
    return dict


def proposal_layer(g_anchors:torch.Tensor,cls_score:torch.Tensor,regression:torch.Tensor,h=256,w=256):
    r'g_anchors:(h,w,k,xyxy),cls_score:(2,h,w,k),regression:(4,h,w,k)'

    positive_anchors = torch.argmax(cls_score,dim=0).bool()
    cls_score = cls_score[1][positive_anchors]
    positive_reg = regression.permute(1,2,3,0)[positive_anchors] 
    positive_xy = g_anchors[positive_anchors]
    revised_positive = apply_box_delt(positive_xy,positive_reg)
    revised_positive = clip_boxes_to_image(revised_positive,[256,256])
    remaining_index = remove_small_boxes(revised_positive,1.0)
    remaining_index = nms(revised_positive[remaining_index],cls_score[remaining_index],0.7)
    revised_positive = revised_positive[remaining_index]
    return revised_positive




    

    '''  
    positive_anchors = torch.argmax(cls_score,dim=1).bool()     # 1通道为正 (n,h,w,k)
    cls_score = cls_score[:,1][positive_anchors]  #(num,scores)
    positive_anchors_reg = regression.permute(0,2,3,4,1)[positive_anchors].contiguous() # (num,4)
    positive_anchors_xyxy =  g_anchors.unsqueeze(0).expand(batch,-1,-1,-1,-1) [positive_anchors] # (rois,4)
    positive_index = remove_small_boxes(positive_anchors_xyxy,1.2) #()
    positive_index = batched_nms(positive_anchors_xyxy[positive_index],cls_score[positive_index],)

    positive_anchors_xyxy = positive_anchors_xyxy[positive_index] 
    positive_anchors_reg = positive_anchors_reg[positive_index]
    proposed_boxes = apply_box_delt(positive_anchors_xyxy,positive_anchors_reg)
    '''
def detction_targets_graph(proposals,gt_boxes,gt_masks):
    r'pro:(rois,c,h,w),'
    
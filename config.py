from matplotlib.colors import same_color
import torch
import numpy as np
from torchvision.utils import save_image

anchors =torch.tensor([[15.3525, 22.7983],
        [ 8.7042,  4.8223],
        [ 6.0075,  7.1270],
        [ 8.2470, 14.6097],
        [13.7141, 13.6732],
        [14.4428,  8.4806],
        [23.2141, 13.4517],
        [ 2.9848,  3.2845],
        [ 9.1061,  9.1358]],device='cuda')

lr = 8e-2
labels = torch.from_numpy(np.load('project_conic/CoNIC_Challenge/labels.npy').astype(np.float64)[...,1])
label = labels[0]
a = torch.ones_like(label)
label = torch.where(label!=0,1.,0.0)
print(label.size())
save_image(label,'label0.png')

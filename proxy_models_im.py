import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

import tqdm
import pc_encoder
import gzip

device = torch.device('cuda')

class LossProxy(nn.Module):
    def __init__(self,mean=0.,std=1.):
        super(LossProxy, self).__init__()

        reshape_sz = 256

        hsz0 = reshape_sz // 2
        hsz1 = reshape_sz // 4
        hsz2 = reshape_sz // 8

        dropout = 0.2

        self.encoder = pc_encoder.PCEncoder()

        self.linear1 = nn.Linear(reshape_sz,1)

        self.loss_network = nn.Sequential(
            self.linear1,
            nn.Flatten(0)
        )

        self.mean = mean
        self.val = std

    def predict(self,inputs,betas):

        encoding = self.encoder(inputs,betas)
        return self.loss_network(encoding)

    def forward(self,chairs,betas):

        chairs = chairs.view(1,-1,3)
        return F.softplus(self.predict(chairs,betas)).flatten()

    def loss(self,pred,labels):

        labels = (labels - self.mean) / self.val
        loss = 0.
        loss += (pred - labels).abs().sum()

        return loss

    def contrastive(self,pred,labels):

        labels = (labels - self.mean) / self.val
        num_choices = max(len(pred),5000)
        choices = np.random.choice(len(pred),(2*num_choices))

        p1,p2 = pred[choices].view(2,-1)
        t1,t2 = labels[choices].view(2,-1)

        truth = t1 < t2
        p_comp = (p1 < p2) == truth
            
        return p_comp.float().mean()
    
class PoseProxy(nn.Module):
    def __init__(self,mean=0.,std=1.):
        super(PoseProxy, self).__init__()

        reshape_sz = 256
        
        hsz0 = reshape_sz // 2
        hsz1 = reshape_sz // 4
        hsz2 = reshape_sz // 8

        dropout = 0.4
        
        self.encoder = pc_encoder.PCEncoderPose()
        
        self.linear1 = nn.Linear(reshape_sz,1)

        self.loss_network = nn.Sequential(
            self.linear1,
            nn.Flatten(0)
        )

        self.mean = mean
        self.std = std

    def predict(self,inputs):

        encoding = self.encoder(inputs)
        return self.loss_network(encoding)

    def forward(self,chairs,pose):

        pts = torch.cat((
            torch.cat((chairs,
                       torch.ones(len(chairs),1).to(device),
                       torch.zeros(len(chairs),1).to(device)
            ),dim=-1),
            torch.cat((pose,
                       torch.zeros(len(pose),1).to(device),
                       torch.ones(len(pose),1).to(device)
            ),dim=-1)
        ),dim=0)

        return F.softplus(self.predict(pts.unsqueeze(0)).flatten())

    def loss(self,pred,labels):

        labels = (labels - self.mean) / self.std

        loss = 0.
        loss += (pred - labels).abs().sum()

        return loss

    def recognition(self,pred):

        p_truth,p_random = pred.view(-1,2).T

        return (p_truth < p_random).float().mean()

    def contrastive(self,pred,labels):

        labels = (labels - self.mean) / self.std
        num_choices = max(len(pred),5000)
        choices = np.random.choice(len(pred),(2*num_choices))

        p1,p2 = pred[choices].view(2,-1)
        t1,t2 = labels[choices].view(2,-1)

        truth = t1 < t2
        p_comp = (p1 < p2) == truth

        return p_comp.float().mean()

shapeAssembly_dir = '..'
files = list(torch.load(f"{shapeAssembly_dir}/lens_info_im.pt").keys())

def get_pose_batch(batch_size=1):
    
    fs = [files[x] for x in np.random.choice(len(files),size=batch_size)]
        
    pose_vecs = []
    pose_pts = []
    for f in fs:
        
        name = f.replace("/","-")[:-3]
        
        pose_vec = torch.tensor(torch.load(f'{shapeAssembly_dir}/body_data_im/{name}/0.pt')['pose'])

        n = f'{shapeAssembly_dir}/bd_samples_im/{name}.gz'
        f = gzip.GzipFile(n, "r")
        pose_points = torch.tensor(np.load(f))
        
        pose_vecs.append(pose_vec)
        pose_pts.append(pose_points)
        
    fn = lambda x: torch.stack(x) if batch_size != 1 else x[0]
    return fn(pose_vecs).to(device),fn(pose_pts).to(device)


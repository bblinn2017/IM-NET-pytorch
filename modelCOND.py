import open3d as o3d

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.measure import marching_cubes_lewiner
import numpy as np

import trimesh

import proxy_models_im as proxy_models
import pc_encoder_im as pc_encoder

import tqdm
import os
import shutil

import matplotlib.pyplot as plt

device = torch.device('cuda')

def newdir(name):

    if os.path.isdir(name):
        shutil.rmtree(name)
    os.mkdir(name)

class Generator(nn.Module):

    def __init__(self,gf_dim,latent_dim,z_dim):
        super(Generator,self).__init__()

        self.network = nn.Sequential(
            nn.Linear(z_dim,gf_dim),
            nn.LeakyReLU(),

            nn.Linear(gf_dim,latent_dim),
        )

    def forward(self,inputs):
        return self.network(inputs)

class Decoder(nn.Module):
    def __init__(self, z_dim, point_dim, gf_dim):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.point_dim = point_dim
        self.gf_dim = gf_dim
        self.linear_1 = nn.Linear(self.z_dim+self.point_dim, self.gf_dim*8, bias=True)
        self.linear_2 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
        self.linear_3 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
        self.linear_4 = nn.Linear(self.gf_dim*8, self.gf_dim*4, bias=True)
        self.linear_5 = nn.Linear(self.gf_dim*4, self.gf_dim*2, bias=True)
        self.linear_6 = nn.Linear(self.gf_dim*2, self.gf_dim*1, bias=True)
        self.linear_7 = nn.Linear(self.gf_dim*1, 1, bias=True)
        nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_1.bias,0)
        nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_2.bias,0)
        nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_3.bias,0)
        nn.init.normal_(self.linear_4.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_4.bias,0)
        nn.init.normal_(self.linear_5.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_5.bias,0)
        nn.init.normal_(self.linear_6.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.linear_6.bias,0)
        nn.init.normal_(self.linear_7.weight, mean=1e-5, std=0.02)
        nn.init.constant_(self.linear_7.bias,0)

    def forward(self, points, z):

        pts = points.repeat(len(z),1,1)
        zs = z.view(-1,1,self.z_dim).repeat(1,pts.shape[1],1)
        pointz = torch.cat([pts,zs],2)

        l1 = self.linear_1(pointz)
        l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

        l4 = self.linear_4(l3)
        l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

        l5 = self.linear_5(l4)
        l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

        l6 = self.linear_6(l5)
        l6 = F.leaky_relu(l6, negative_slope=0.02, inplace=True)

        l7 = self.linear_7(l6)

        #l7 = torch.clamp(l7, min=0, max=1)
        l7 = torch.max(torch.min(l7, l7*0.01+0.99), l7*0.01)

        return l7

class Model(nn.Module):

    def __init__(self):
        super(Model,self).__init__()

        self.bsz = 1

        self.z_dim = 256
        self.latent_dim = 256

        self.g_hdim = 2048
        self.d_hdim = 128

        self.b_mean,self.b_std = torch.load('betas_params.pt')

        self.frame_coords = torch.tensor(torch.load('frame_coords.pt')).to(device).unsqueeze(0)
        self.N = 64
        self.isosurface = 0.4

        self.z_mean = 0.
        self.z_std = 1.2

        self.M = pc_encoder.FiLMNetwork(16,self.latent_dim)

        self.G = Generator(self.g_hdim,self.latent_dim,self.z_dim)
        self.G.load_state_dict(torch.load('modelG_dict.pt'))
        for param in self.G.parameters():
            param.requires_grad = False
        self.G.eval()

        self.D = Decoder(self.latent_dim,3,self.d_hdim)
        self.D.load_state_dict(torch.load('modelD_dict.pt'))
        for param in self.D.parameters():
            param.requires_grad = False
        self.D.eval()

        self.proxy = proxy_models.LossProxy()
        self.proxy.load_state_dict(torch.load('comf_dict_im.pt'))
        for param in self.proxy.parameters():
            param.requires_grad = False
        self.proxy.eval()

        self.optimizer = torch.optim.Adam(self.M.parameters(),1e-5)

    def forward(self,cond=True):

        # Random Inputs
        z_curr = self.z_mean + torch.randn(size=import mesh_to_sdf(self.bsz,self.z_dim)).to(device) * self.z_std
        betas = (self.b_mean + torch.randn(size=(self.bsz,len(self.b_std))) * self.b_std).float().to(device)

        if is_test:
            z_curr = glb_zs[[idx]]
            betas = glb_betas[[idx]]

        # Condition
        if cond:
            z_curr = self.M(betas,z_curr)

        # Generator
        z_curr = self.G(z_curr)

        # Decoder
        net_out = self.D(self.frame_coords, z_curr)
        model_out = - (net_out - self.isosurface).view(self.N,self.N,self.N).detach().cpu().numpy()

        verts,faces,_,_ = marching_cubes_lewiner(
            model_out, level = 0., spacing = [1./self.N]*3
        )
        faces = np.concatenate([faces])

        return z_curr,verts,faces,betas

    def loss(self,z_curr,verts,faces,betas):

        # Calculate dL/dp
        xyz_upstream = torch.tensor(verts.astype(float), dtype=torch.float, device=device)
        faces_upstream = torch.tensor(faces.astype(np.long), dtype=torch.long, device=device)

        # Sample points
        pts = self.sample_mesh_surface(xyz_upstream,faces_upstream)

        # Points upstream
        points_upstream = pts.clone().detach().requires_grad_(True)
        comf = self.proxy(points_upstream.unsqueeze(0),betas,True)

        comf.backward()
        dL_dp = points_upstream.grad

        # Calculate dL/ds
        self.optimizer.zero_grad()
        points = pts.clone().detach().requires_grad_(True)

        pred_sdf = self.D(points, z_curr)
        pred_sdf = - (pred_sdf - self.isosurface).squeeze(0)

        loss_normals = torch.sum(pred_sdf)
        loss_normals.backward(retain_graph = True)

        normals = points.grad/torch.norm(points.grad, 2, 1).unsqueeze(-1)

        self.optimizer.zero_grad()

        dL_ds = -torch.matmul(dL_dp.unsqueeze(1), normals.unsqueeze(-1)).squeeze(-1)

        # Calculate dL/ds(s)
        loss_backward = (dL_ds * pred_sdf).mean()

        return loss_backward

    def optimize(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def transform_verts(self,verts):
        if torch.is_tensor(verts):
            mini = verts.min(dim=0)[0]
            maxi = verts.max(dim=0)[0]
            center = (mini + maxi)/2.
            scale_factor = torch.prod(maxi - mini).pow(1./3.)
            fn = lambda x: torch.stack(x,dim=-1)
        else:
            mini = verts.min(axis=0)
            maxi = verts.max(axis=0)
            center = (mini + maxi)/2.
            scale_factor = np.prod(maxi - mini) ** (1./3.)
            fn = lambda x: np.column_stack(x)
        verts = (verts - center) / scale_factor
        return fn((-verts[:,0],verts[:,1],-verts[:,2]))

    def sample_mesh_surface(self,v,f,n_s=10000):

        a,b,c = v[f].permute(1,0,2)
        areas = torch.cross(a - b, b - c).norm(dim=-1)
        weights = (areas / areas.sum()).detach().cpu().numpy()

        choices = np.random.choice(a=len(weights),size=n_s,p=weights)
        u,v = torch.rand(size=(2,n_s)).to(device)

        pts = (1 - u**.5).view(-1,1) * a[choices]
        pts += (u**.5 * (1 - v)).view(-1,1) * b[choices]
        pts += (v * u ** .5).view(-1,1) * c[choices]

        return pts

    def writeObj(self, verts, faces, outfile):
        faces += 1
        with open(outfile, 'w') as f:
            for a, b, c in verts.tolist():
                f.write(f'v {a} {b} {c}\n')

            for a, b, c in faces.tolist():
                f.write(f"f {a} {b} {c}\n")

    def export(self,cond):

        z,v,f,betas = self.forward(cond)
        v = torch.tensor(v).float()
        f = torch.tensor(f).long()

        epsilon = 0.

        u_pts = torch.tensor(np.mgrid[0:64,0:64,0:64].T).view(-1,3) / 63.
        a = v.min(dim=0)[0]
        b = v.max(dim=0)[0]

        a = a - .5
        b = b - .5

        pts = a + u_pts * (b - a)

        sdf = self.D(pts.to(device),z).view(1,self.N,self.N,self.N).cpu().detach()
        sdf = -(sdf - self.isosurface).flip((1,3))

        v = self.transform_verts(v)
        grid_min = v.min(dim=0)[0]
        grid_max = v.max(dim=0)[0]

        d = {
            'sdf':sdf.detach().cpu().numpy(),
            'grid_min':grid_min.detach().cpu().numpy(),
            'grid_max':grid_max.detach().cpu().numpy(),
            'vertices':v.detach().cpu().numpy(),
            'faces':f.detach().cpu().numpy()
            'betas':betas.detach().cpu().numpy()
        }
        
        return d

def train(model):

    global idx
    idx = 0

    iters = 100
    for i in range(0,iters,model.bsz):
        outputs = model()
        loss = model.loss(*outputs)
        model.optimize(loss)
        idx += 1

def test(model,name,cond=True):

    global idx
    idx = 0

    num_exp = 100
    for i in tqdm.tqdm(range(num_exp)):
        item = model.export(cond)
        write_gzip(item,f'model_output_cond/{name}/{i}.gz')
        idx += 1

def test_all():

    model = Model().to(device)

    newdir('model_output_cond')
    names = [e for e in range(step,step*num_exp+1,step)]
    for name in names:
        exp_dir = f'model_output_cond/{name}'
        newdir(exp_dir)
        model.load_state_dict(torch.load(f'comf_model_{name}.pt'))
        test(model,name)

    newdir('model_output_cond/reg')
    test(model,'reg',False)

def main():

    global glb_betas
    glb_betas = torch.load('gen_betas.pt').to(device)
    global glb_zs
    glb_zs = torch.load('gen_zs.pt').to(device)

    global is_test
    is_test = False
    is_train = True

    if is_train:
        model = Model().to(device)
        epochs = step * num_exp

        for epoch in tqdm.tqdm(range(epochs)):
            train(model)

            if (epoch+1) % step == 0:
                torch.save(model.state_dict(),f'comf_model_{epoch+1}.pt')

    is_test = True
    test_all()


if __name__ == "__main__":

    global step,num_exp
    step = 1
    num_exp = 5

    main()

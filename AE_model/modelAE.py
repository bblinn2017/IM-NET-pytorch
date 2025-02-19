import os
import time
import math
import random
import numpy as np
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F

# import mcubes
import skimage
from skimage import measure

from utils import *
device = torch.device('cuda')

#from proxy_models import get_comfort_loss,get_pose_loss
import pc_encoder
import modelGAN

#pytorch 1.2.0 implementation


class generator(nn.Module):
    def __init__(self, z_dim, point_dim, gf_dim):
        super(generator, self).__init__()
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

    def forward(self, points, z, is_training=False):
        zs = z.view(-1,1,self.z_dim).repeat(1,points.size()[1],1)
        pointz = torch.cat([points,zs],2)

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

class encoder(nn.Module):
    def __init__(self, ef_dim, z_dim):
        super(encoder, self).__init__()
        self.ef_dim = ef_dim
        self.z_dim = z_dim
        self.conv_1 = nn.Conv3d(1, self.ef_dim, 4, stride=2, padding=1, bias=False)
        self.in_1 = nn.InstanceNorm3d(self.ef_dim)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim*2, 4, stride=2, padding=1, bias=False)
        self.in_2 = nn.InstanceNorm3d(self.ef_dim*2)
        self.conv_3 = nn.Conv3d(self.ef_dim*2, self.ef_dim*4, 4, stride=2, padding=1, bias=False)
        self.in_3 = nn.InstanceNorm3d(self.ef_dim*4)
        self.conv_4 = nn.Conv3d(self.ef_dim*4, self.ef_dim*8, 4, stride=2, padding=1, bias=False)
        self.in_4 = nn.InstanceNorm3d(self.ef_dim*8)
        self.conv_5 = nn.Conv3d(self.ef_dim*8, self.z_dim, 4, stride=1, padding=0, bias=True)
        #self.conv_5 = nn.Conv3d(self.ef_dim*8, self.z_dim*2, 4, stride=1, padding=0, bias=True)
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.xavier_uniform_(self.conv_3.weight)
        nn.init.xavier_uniform_(self.conv_4.weight)
        nn.init.xavier_uniform_(self.conv_5.weight)
        nn.init.constant_(self.conv_5.bias,0)

    def forward(self, inputs, is_training=False):
        d_1 = self.in_1(self.conv_1(inputs))
        d_1 = F.leaky_relu(d_1, negative_slope=0.02, inplace=True)

        d_2 = self.in_2(self.conv_2(d_1))
        d_2 = F.leaky_relu(d_2, negative_slope=0.02, inplace=True)

        d_3 = self.in_3(self.conv_3(d_2))
        d_3 = F.leaky_relu(d_3, negative_slope=0.02, inplace=True)

        d_4 = self.in_4(self.conv_4(d_3))
        d_4 = F.leaky_relu(d_4, negative_slope=0.02, inplace=True)

        d_5 = self.conv_5(d_4)
        d_5 = F.leaky_relu(d_5, negative_slope=0.02, inplace=True)
        d_5 = d_5.view(-1, self.z_dim)
        d_5 = torch.sigmoid(d_5)
        return d_5
        """
        # VAE
        d_5 = d_5.view(-1, 2, self.z_dim)
        mu = d_5[:,0]
        log_var = d_5[:,1]

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        sample = torch.sigmoid(sample)
        
        return sample, mu, log_var
        """

class im_network(nn.Module):
    def __init__(self, ef_dim, gf_dim, z_dim, point_dim):
        super(im_network, self).__init__()
        self.ef_dim = ef_dim
        self.gf_dim = gf_dim
        self.z_dim = z_dim
        self.point_dim = point_dim
        self.encoder = encoder(self.ef_dim, self.z_dim)
        self.generator = generator(self.z_dim, self.point_dim, self.gf_dim)

    def forward(self, inputs, z_vector, point_coord, is_training=False):
        if is_training:
            z_vector = self.encoder(inputs, is_training=is_training)
            net_out = self.generator(point_coord, z_vector, is_training=is_training)
            return z_vector,net_out
        
        if inputs is not None:
            z_vector = self.encoder(inputs, is_training=is_training)
        if z_vector is not None and point_coord is not None:
            net_out = self.generator(point_coord, z_vector, is_training=is_training)
        else:
            net_out = None

        return z_vector, net_out


class IM_AE(object):
    def __init__(self, config):
        #progressive training
        #1-- (16, 16*16*16)
        #2-- (32, 16*16*16)
        #3-- (64, 16*16*16*4)
        self.config = config
        self.sample_vox_size = config.sample_vox_size
        if self.sample_vox_size==16:
            self.load_point_batch_size = 16*16*16
            self.point_batch_size = 16*16*16
            self.shape_batch_size = 32
        elif self.sample_vox_size==32:
            self.load_point_batch_size = 16*16*16
            self.point_batch_size = 16*16*16
            self.shape_batch_size = 32
        elif self.sample_vox_size==64:
            self.load_point_batch_size = 16*16*16*4
            self.point_batch_size = 16*16*16
            self.shape_batch_size = 64
        self.input_size = 64 #input voxel grid size

        self.ef_dim = 32
        self.gf_dim = 128
        self.z_dim = 256
        self.point_dim = 3

        self.dataset_name = config.dataset
        self.dataset_load = self.dataset_name + '_train'
        if not (config.train or config.getz):
            self.dataset_load = self.dataset_name + '_test'
        self.checkpoint_dir = config.checkpoint_dir
        self.data_dir = config.data_dir

        if config.load_data:
            data_hdf5_name = self.data_dir+'/'+self.dataset_load+'.hdf5'
            if os.path.exists(data_hdf5_name):
                data_dict = h5py.File(data_hdf5_name, 'r')
                self.data_points = (data_dict['points_'+str(self.sample_vox_size)][:].astype(np.float32)+0.5)/256-0.5
                self.data_values = data_dict['values_'+str(self.sample_vox_size)][:].astype(np.float32)
                self.data_voxels = data_dict['voxels'][:]
                #reshape to NCHW
                self.data_voxels = np.reshape(self.data_voxels, [-1,1,self.input_size,self.input_size,self.input_size])
            else:
                print("error: cannot load "+data_hdf5_name)
                exit(0)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        #build model
        self.im_network = im_network(self.ef_dim, self.gf_dim, self.z_dim, self.point_dim)
        self.im_network.to(self.device)
        #print params
        #for param_tensor in self.im_network.state_dict():
        #       print(param_tensor, "\t", self.im_network.state_dict()[param_tensor].size())
        self.optimizer = torch.optim.Adam(self.im_network.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
        #pytorch does not have a checkpoint manager
        #have to define it myself to manage max num of checkpoints to keep
        self.max_to_keep = 2
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
        self.checkpoint_name='IM_AE.model'
        self.checkpoint_manager_list = [None] * self.max_to_keep
        self.checkpoint_manager_pointer = 0
        #loss
        def network_loss(G,point_value):
            R = torch.mean((G-point_value)**2)
            #KL = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum()
            return R#,KL
        self.loss = network_loss


        #keep everything a power of 2
        self.cell_grid_size = 4
        self.frame_grid_size = 64
        self.real_size = self.cell_grid_size*self.frame_grid_size #=256, output point-value voxel grid size in testing
        self.test_size = 32 #related to testing batch_size, adjust according to gpu memory size
        self.test_point_batch_size = self.test_size*self.test_size*self.test_size #do not change

        #get coords for training
        dima = self.test_size
        dim = self.frame_grid_size
        self.aux_x = np.zeros([dima,dima,dima],np.uint8)
        self.aux_y = np.zeros([dima,dima,dima],np.uint8)
        self.aux_z = np.zeros([dima,dima,dima],np.uint8)
        multiplier = int(dim/dima)
        multiplier2 = multiplier*multiplier
        multiplier3 = multiplier*multiplier*multiplier
        for i in range(dima):
            for j in range(dima):
                for k in range(dima):
                    self.aux_x[i,j,k] = i*multiplier
                    self.aux_y[i,j,k] = j*multiplier
                    self.aux_z[i,j,k] = k*multiplier
        self.coords = np.zeros([multiplier3,dima,dima,dima,3],np.float32)
        for i in range(multiplier):
            for j in range(multiplier):
                for k in range(multiplier):
                    self.coords[i*multiplier2+j*multiplier+k,:,:,:,0] = self.aux_x+i
                    self.coords[i*multiplier2+j*multiplier+k,:,:,:,1] = self.aux_y+j
                    self.coords[i*multiplier2+j*multiplier+k,:,:,:,2] = self.aux_z+k
        self.coords = (self.coords.astype(np.float32)+0.5)/dim-0.5
        self.coords = np.reshape(self.coords,[multiplier3,self.test_point_batch_size,3])
        self.coords = torch.from_numpy(self.coords)
        self.coords = self.coords.to(self.device)


        #get coords for testing
        dimc = self.cell_grid_size
        dimf = self.frame_grid_size
        self.cell_x = np.zeros([dimc,dimc,dimc],np.int32)
        self.cell_y = np.zeros([dimc,dimc,dimc],np.int32)
        self.cell_z = np.zeros([dimc,dimc,dimc],np.int32)
        self.cell_coords = np.zeros([dimf,dimf,dimf,dimc,dimc,dimc,3],np.float32)
        self.frame_coords = np.zeros([dimf,dimf,dimf,3],np.float32)
        self.frame_x = np.zeros([dimf,dimf,dimf],np.int32)
        self.frame_y = np.zeros([dimf,dimf,dimf],np.int32)
        self.frame_z = np.zeros([dimf,dimf,dimf],np.int32)
        for i in range(dimc):
            for j in range(dimc):
                for k in range(dimc):
                    self.cell_x[i,j,k] = i
                    self.cell_y[i,j,k] = j
                    self.cell_z[i,j,k] = k
        for i in range(dimf):
            for j in range(dimf):
                for k in range(dimf):
                    self.cell_coords[i,j,k,:,:,:,0] = self.cell_x+i*dimc
                    self.cell_coords[i,j,k,:,:,:,1] = self.cell_y+j*dimc
                    self.cell_coords[i,j,k,:,:,:,2] = self.cell_z+k*dimc
                    self.frame_coords[i,j,k,0] = i
                    self.frame_coords[i,j,k,1] = j
                    self.frame_coords[i,j,k,2] = k
                    self.frame_x[i,j,k] = i
                    self.frame_y[i,j,k] = j
                    self.frame_z[i,j,k] = k
        self.cell_coords = (self.cell_coords.astype(np.float32)+0.5)/self.real_size-0.5
        self.cell_coords = np.reshape(self.cell_coords,[dimf,dimf,dimf,dimc*dimc*dimc,3])
        self.cell_x = np.reshape(self.cell_x,[dimc*dimc*dimc])
        self.cell_y = np.reshape(self.cell_y,[dimc*dimc*dimc])
        self.cell_z = np.reshape(self.cell_z,[dimc*dimc*dimc])
        self.frame_x = np.reshape(self.frame_x,[dimf*dimf*dimf])
        self.frame_y = np.reshape(self.frame_y,[dimf*dimf*dimf])
        self.frame_z = np.reshape(self.frame_z,[dimf*dimf*dimf])
        self.frame_coords = (self.frame_coords.astype(np.float32)+0.5)/dimf-0.5
        self.frame_coords = np.reshape(self.frame_coords,[dimf*dimf*dimf,3])

        self.sampling_threshold = 0.5 #final marching cubes threshold

        self.film_network = pc_encoder.FiLMNetwork(16,256).to(device)
    @property
    def model_dir(self):
        return "{}_ae_{}".format(self.dataset_name, self.input_size)

    def transform_verts(self,verts):

        tup = (-verts[:,0],verts[:,1],-verts[:,2])
        if torch.is_tensor(verts):
            fn = torch.stack
            params = {'dim':-1}
        else:
            fn = np.column_stack
            params = {}
        return fn(tup,**params) * 2.
    
    def conditioned_train(self, config):
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            self.im_network.load_state_dict(torch.load(model_dir))
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        training_epoch = config.epoch
        optimizer = torch.optim.Adam(self.film_network.parameters(),1e-5)
        optimizer.zero_grad()

        iters = 1000
        self.im_network.train()
        
        import tqdm
        for epoch in tqdm.tqdm(range(training_epoch)):

            for i in range(iters):
                self.train_iteration(optimizer)
                
            if (epoch+1)%20==0 or (epoch+1) == training_epoch:
                if not os.path.exists(self.checkpoint_path):
                    os.makedirs(self.checkpoint_path)
                save_dir = os.path.join(self.checkpoint_path,self.checkpoint_name+str(self.sample_vox_size)+"-cond-"+str(epoch)+".pth")
                self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer+1)%self.max_to_keep
                #delete checkpoint
                if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
                    if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
                        os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])
                #save checkpoint
                torch.save({'im_network':self.im_network.state_dict(),
                            'film_network':self.film_network.state_dict()}, save_dir)
                #update checkpoint manager
                self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
                #write file
                checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint_cond")
                fout = open(checkpoint_txt, 'w')
                for i in range(self.max_to_keep):
                    pointer = (self.checkpoint_manager_pointer+self.max_to_keep-i)%self.max_to_keep
                    if self.checkpoint_manager_list[pointer] is not None:
                        fout.write(self.checkpoint_manager_list[pointer]+"\n")
                fout.close()

    def train_iteration(self,optimizer,bsz=1):

        # Calculate dL/dx
        z, verts, faces, betas = self.conditioned_z2voxel()
        
        xyz_upstream = torch.tensor(verts.astype(float), requires_grad=True, dtype=torch.float, device=self.device)
        faces_upstream = torch.tensor(faces.astype(int), requires_grad=False, dtype=torch.long, device=self.device)

        points = self.sample_mesh_surface(xyz_upstream,faces_upstream)
        comf = get_comfort_loss(
            self.transform_verts(points),
        betas)

        comf.backward()
        dL_dx = xyz_upstream.grad

        # Calculate dL/ds
        optimizer.zero_grad()
        xyz = torch.tensor(verts.astype(float), requires_grad = True, dtype=torch.float, device=device)

        _, pred_sdf = self.im_network(None, z, xyz.unsqueeze(0), is_training=False)
        pred_sdf = - (pred_sdf - 0.5).squeeze(0)

        loss_normals = torch.sum(pred_sdf)
        loss_normals.backward(retain_graph = True)

        normals = xyz.grad/torch.norm(xyz.grad, 2, 1).unsqueeze(-1)

        optimizer.zero_grad()

        dL_ds = -torch.matmul(dL_dx.unsqueeze(1), normals.unsqueeze(-1)).squeeze(-1)

        # Calculate dL/ds(s)
        loss_backward = torch.sum(dL_ds * pred_sdf)
        loss_backward.backward()

        optimizer.step()

    def conditioned_generate(self,config):

        num = 5000
        outdir = config.sample_dir 

        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint_cond")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            d = torch.load(model_dir)
            self.im_network.load_state_dict(d['im_network'])
            self.film_network.load_state_dict(d['film_network'])
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        #dic = {'im_network':self.im_network.state_dict(),
        #       'config':self.config}
        #torch.save(dic,'modelAE_dict.pt')
            
        b = []
        import tqdm
        for i in tqdm.tqdm(range(num)):
            z, verts, faces, betas = self.conditioned_z2voxel(False)
            verts = self.transform_verts(verts)
            
            verts = torch.tensor(verts).float()
            faces = torch.tensor(faces.astype(np.long))

            bbox = verts.min(dim=0)[0],verts.max(dim=0)[0]
            
            params = {'z':z,'scale':.5,'bbox':bbox}
            
            torch.save(
                (
                    params,
                    verts,faces,
                    betas.cpu()
                ),
            f'{outdir}/{i}.pt')
            self.writeObj(verts,faces,f'{outdir}/{i}.obj')

    def writeObj(self, verts, faces, outfile):
        faces += 1
        with open(outfile, 'w') as f:
            for a, b, c in verts.tolist():
                f.write(f'v {a} {b} {c}\n')
                
            for a, b, c in faces.tolist():
                f.write(f"f {a} {b} {c}\n")
        
    def conditioned_z2voxel(self,iscond=True,betas=None):

        if betas == None:
            betas = (self.betas_mean + torch.randn(size=self.betas_std.shape) * self.betas_std).float\
().to(device)

        z0 = torch.randn(size=(1,256)).to(device)
        if iscond:
            z0 = self.film_network(betas,z0).to(device)
        
        z = self.latent_mean.float().to(device) + z0 * self.latent_std.float().to(device)

        point_coord = torch.tensor(self.frame_coords).unsqueeze(0).to(self.device)
        _, model_out_ = self.im_network(None, z, point_coord, is_training=False)

        N = 64
        model_out = - (model_out_ - 0.5).view(N,N,N).detach().cpu().numpy()

        verts,faces,_,_ = skimage.measure.marching_cubes_lewiner(
            model_out, level = 0., spacing = [1./self.frame_grid_size]*3
        )
        verts -= 0.5

        return z,verts,faces,betas
        
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

    def train(self, config):
        #load previous checkpoint
        """
        if config.load_checkpoint:
            checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
            if os.path.exists(checkpoint_txt):
                fin = open(checkpoint_txt)
                model_dir = fin.readline().strip()
                fin.close()
                self.im_network.load_state_dict(torch.load(model_dir))
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        """
        #self.im_network.load_state_dict(torch.load('modelAE_dict.pt')['im_network'])
        shape_num = len(self.data_voxels)
        batch_index_list = np.arange(shape_num)
        

        print("\n\n----------net summary----------")
        print("training samples   ", shape_num)
        print("-------------------------------\n\n")

        start_time = time.time()
        assert config.epoch==0 or config.iteration==0
        training_epoch = config.epoch + int(config.iteration/shape_num)
        batch_num = int(shape_num/self.shape_batch_size)
        point_batch_num = int(self.load_point_batch_size/self.point_batch_size)

        import tqdm
        for epoch in range(training_epoch):
            self.im_network.train()
            np.random.shuffle(batch_index_list)
            avg_loss_sp = 0
            avg_num = 0
            print(f'{epoch+1} / {training_epoch}')
            for idx in tqdm.tqdm(range(batch_num)):
                dxb = batch_index_list[idx*self.shape_batch_size:(idx+1)*self.shape_batch_size]
                batch_voxels = self.data_voxels[dxb].astype(np.float32)
                if point_batch_num==1:
                    point_coord = self.data_points[dxb]
                    point_value = self.data_values[dxb]
                else:
                    which_batch = np.random.randint(point_batch_num)
                    point_coord = self.data_points[dxb,which_batch*self.point_batch_size:(which_batch+1)*self.point_batch_size]
                    point_value = self.data_values[dxb,which_batch*self.point_batch_size:(which_batch+1)*self.point_batch_size]

                batch_voxels = torch.from_numpy(batch_voxels)
                point_coord = torch.from_numpy(point_coord)
                point_value = torch.from_numpy(point_value)

                batch_voxels = batch_voxels.to(self.device)
                point_coord = point_coord.to(self.device)
                point_value = point_value.to(self.device)

                self.im_network.zero_grad()
                _, net_out = self.im_network(batch_voxels, None, point_coord, is_training=True)
                
                R = self.loss(net_out, point_value)
                errSP = R
                
                errSP.backward()
                self.optimizer.step()

                avg_loss_sp += R.detach().item()
                avg_num += 1

                del net_out, R, errSP
                torch.cuda.empty_cache()

            print(f'Loss: {avg_loss_sp/avg_num}')
            if (epoch%20==19) or (epoch+1) == training_epoch:
                
                if not os.path.exists(self.checkpoint_path):
                    os.makedirs(self.checkpoint_path)
                save_dir = os.path.join(self.checkpoint_path,self.checkpoint_name+str(self.sample_vox_size)+"-"+str(epoch)+".pth")
                self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer+1)%self.max_to_keep
                #delete checkpoint
                if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
                    if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
                        os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])
                #save checkpoint
                torch.save(self.im_network.state_dict(), 'checkpoint/im_dict.pt')
                #update checkpoint manager
                self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
                #write file
                checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
                fout = open(checkpoint_txt, 'w')
                for i in range(self.max_to_keep):
                    pointer = (self.checkpoint_manager_pointer+self.max_to_keep-i)%self.max_to_keep
                    if self.checkpoint_manager_list[pointer] is not None:
                        fout.write(self.checkpoint_manager_list[pointer]+"\n")
                fout.close()

    def test_1(self, config, name):
        multiplier = int(self.frame_grid_size/self.test_size)
        multiplier2 = multiplier*multiplier
        self.im_network.eval()
        t = np.random.randint(len(self.data_voxels))
        model_float = np.zeros([self.frame_grid_size+2,self.frame_grid_size+2,self.frame_grid_size+2],np.float32)
        batch_voxels = self.data_voxels[t:t+1].astype(np.float32)
        batch_voxels = torch.from_numpy(batch_voxels)
        batch_voxels = batch_voxels.to(self.device)
        z_vector, _ = self.im_network(batch_voxels, None, None, is_training=False)
        for i in range(multiplier):
            for j in range(multiplier):
                for k in range(multiplier):
                    minib = i*multiplier2+j*multiplier+k
                    point_coord = self.coords[minib:minib+1]
                    _, net_out = self.im_network(None, z_vector, point_coord, is_training=False)
                    #net_out = torch.clamp(net_out, min=0, max=1)
                    model_float[self.aux_x+i+1,self.aux_y+j+1,self.aux_z+k+1] = np.reshape(net_out.detach().cpu().numpy(), [self.test_size,self.test_size,self.test_size])

        vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
        vertices = (vertices.astype(np.float32)-0.5)/self.frame_grid_size-0.5
        #output ply sum
        write_ply_triangle(config.sample_dir+"/"+name+".ply", vertices, triangles)
        print("[sample]")



    def z2voxel(self, z):
        model_float = np.zeros([self.real_size+2,self.real_size+2,self.real_size+2],np.float32)
        dimc = self.cell_grid_size
        dimf = self.frame_grid_size

        frame_flag = np.zeros([dimf+2,dimf+2,dimf+2],np.uint8)
        queue = []

        frame_batch_num = int(dimf**3/self.test_point_batch_size)
        assert frame_batch_num>0

        #get frame grid values
        for i in range(frame_batch_num):
            point_coord = self.frame_coords[i*self.test_point_batch_size:(i+1)*self.test_point_batch_size]
            point_coord = np.expand_dims(point_coord, axis=0)
            point_coord = torch.from_numpy(point_coord)
            point_coord = point_coord.to(self.device)
            _, model_out_ = self.im_network(None, z, point_coord, is_training=False)
            model_out = model_out_.detach().cpu().numpy()[0]
            x_coords = self.frame_x[i*self.test_point_batch_size:(i+1)*self.test_point_batch_size]
            y_coords = self.frame_y[i*self.test_point_batch_size:(i+1)*self.test_point_batch_size]
            z_coords = self.frame_z[i*self.test_point_batch_size:(i+1)*self.test_point_batch_size]
            frame_flag[x_coords+1,y_coords+1,z_coords+1] = np.reshape((model_out>self.sampling_threshold).astype(np.uint8), [self.test_point_batch_size])

        #get queue and fill up ones
        for i in range(1,dimf+1):
            for j in range(1,dimf+1):
                for k in range(1,dimf+1):
                    maxv = np.max(frame_flag[i-1:i+2,j-1:j+2,k-1:k+2])
                    minv = np.min(frame_flag[i-1:i+2,j-1:j+2,k-1:k+2])
                    if maxv!=minv:
                        queue.append((i,j,k))
                    elif maxv==1:
                        x_coords = self.cell_x+(i-1)*dimc
                        y_coords = self.cell_y+(j-1)*dimc
                        z_coords = self.cell_z+(k-1)*dimc
                        model_float[x_coords+1,y_coords+1,z_coords+1] = 1.0

        print("running queue:",len(queue))
        cell_batch_size = dimc**3
        cell_batch_num = int(self.test_point_batch_size/cell_batch_size)
        assert cell_batch_num>0
        #run queue
        while len(queue)>0:
            batch_num = min(len(queue),cell_batch_num)
            point_list = []
            cell_coords = []
            for i in range(batch_num):
                point = queue.pop(0)
                point_list.append(point)
                cell_coords.append(self.cell_coords[point[0]-1,point[1]-1,point[2]-1])
            cell_coords = np.concatenate(cell_coords, axis=0)
            cell_coords = np.expand_dims(cell_coords, axis=0)
            cell_coords = torch.from_numpy(cell_coords)
            cell_coords = cell_coords.to(self.device)
            _, model_out_batch_ = self.im_network(None, z, cell_coords, is_training=False)
            model_out_batch = model_out_batch_.detach().cpu().numpy()[0]
            for i in range(batch_num):
                point = point_list[i]
                model_out = model_out_batch[i*cell_batch_size:(i+1)*cell_batch_size,0]
                x_coords = self.cell_x+(point[0]-1)*dimc
                y_coords = self.cell_y+(point[1]-1)*dimc
                z_coords = self.cell_z+(point[2]-1)*dimc
                model_float[x_coords+1,y_coords+1,z_coords+1] = model_out

                if np.max(model_out)>self.sampling_threshold:
                    for i in range(-1,2):
                        pi = point[0]+i
                        if pi<=0 or pi>dimf: continue
                        for j in range(-1,2):
                            pj = point[1]+j
                            if pj<=0 or pj>dimf: continue
                            for k in range(-1,2):
                                pk = point[2]+k
                                if pk<=0 or pk>dimf: continue
                                if (frame_flag[pi,pj,pk] == 0):
                                    frame_flag[pi,pj,pk] = 1
                                    queue.append((pi,pj,pk))
        return model_float

    #may introduce foldovers
    def optimize_mesh(self, vertices, z, iteration = 3):
        new_vertices = np.copy(vertices)

        new_vertices_ = np.expand_dims(new_vertices, axis=0)
        new_vertices_ = torch.from_numpy(new_vertices_)
        new_vertices_ = new_vertices_.to(self.device)
        _, new_v_out_ = self.im_network(None, z, new_vertices_, is_training=False)
        new_v_out = new_v_out_.detach().cpu().numpy()[0]

        for iter in range(iteration):
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    for k in [-1,0,1]:
                        if i==0 and j==0 and k==0: continue
                        offset = np.array([[i,j,k]],np.float32)/(self.real_size*6*2**iter)
                        current_vertices = vertices+offset
                        current_vertices_ = np.expand_dims(current_vertices, axis=0)
                        current_vertices_ = torch.from_numpy(current_vertices_)
                        current_vertices_ = current_vertices_.to(self.device)
                        _, current_v_out_ = self.im_network(None, z, current_vertices_, is_training=False)
                        current_v_out = current_v_out_.detach().cpu().numpy()[0]
                        keep_flag = abs(current_v_out-self.sampling_threshold)<abs(new_v_out-self.sampling_threshold)
                        keep_flag = keep_flag.astype(np.float32)
                        new_vertices = current_vertices*keep_flag+new_vertices*(1-keep_flag)
                        new_v_out = current_v_out*keep_flag+new_v_out*(1-keep_flag)
            vertices = new_vertices

        return vertices

    def test_mesh_gan(self, config):
        self.im_network.load_state_dict(torch.load('checkpoint/im_dict.pt'))
        
        gan = modelGAN.ZGAN(config)
        gan.load_state_dict(torch.load(f'{config.checkpoint_dir}/zgan_dict_curr2.pt'))
        G = gan.G
        
        data_hdf5_name = 'checkpoint/03001627_vox256_img_ae_64/03001627_vox256_img_train_z.hdf5'
        data_dict = h5py.File(data_hdf5_name, 'r')
        data_zs = data_dict['zs']
        data_zs,data_zs_test = data_zs[:-100],data_zs[-100:]
        

        betas = torch.load('default_betas.pt').float()
        outdir = 'meshes'

        isosurface = .4
        import tqdm

        for t in tqdm.tqdm(range(config.end)):
            point_coord = torch.tensor(self.frame_coords).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                z = torch.randn(size=(1,gan.z_dim))
                model_z = G(z).to(device)
                #model_z = torch.tensor(data_zs_test[[t]]).to(device)
                
                _, model_out_ = self.im_network(None, model_z, point_coord, is_training=False)

                N = 64
                model_out = - (model_out_ - isosurface).view(N,N,N).detach().cpu().numpy()

            verts,faces,_,_ = skimage.measure.marching_cubes_lewiner(
                model_out, level = 0., spacing = [1./self.frame_grid_size]*3
            )
            verts -= 0.5

            verts = self.transform_verts(verts)

            verts = torch.tensor(verts).float()
            faces = torch.tensor(faces.astype(np.long))

            bbox = verts.min(dim=0)[0],verts.max(dim=0)[0]

            params = {'z':model_z.detach().cpu(),'scale':.5,'bbox':bbox}

            torch.save(
                (
                    params,
                    verts,faces,
                    betas
                ),
                f'{config.sample_dir}/{t}.pt')

            self.writeObj(verts,faces,f'{outdir}/{t}.obj')

    #output shape as ply
    def test_mesh(self, config):
        self.im_network.load_state_dict(torch.load('checkpoint/im_dict.pt'))

        betas = torch.load('default_betas.pt')
        isosurface = .4
        import tqdm
        #import mcubes
        begin = 0#1355
        
        for t in tqdm.tqdm(range(min(len(self.data_voxels),config.end))):
            batch_voxels_ = self.data_voxels[t:t+1].astype(np.float32)
            batch_voxels = torch.from_numpy(batch_voxels_)
            batch_voxels = batch_voxels.to(self.device)
            model_z,_ = self.im_network(batch_voxels, None, None, is_training=False)

            with torch.no_grad():
                point_coord = torch.tensor(self.frame_coords).unsqueeze(0).to(self.device)
                _, model_out_ = self.im_network(None, model_z, point_coord, is_training=False)
            
                N = 64
                model_out = - (model_out_ - isosurface).view(N,N,N).detach().cpu().numpy()
            import open3d as o3d
        
            verts,faces,_,_ = skimage.measure.marching_cubes_lewiner(
                model_out, level = 0., spacing = [1./self.frame_grid_size]*3
            )
            faces = np.concatenate([faces])

            v = torch.tensor(verts).float()
            f = torch.tensor(faces).long()
        
            u_pts = torch.tensor(np.mgrid[0:64,0:64,0:64].T).view(-1,3) / 63.
            a = v.min(dim=0)[0] - .5
            b = v.max(dim=0)[0] - .5

            u_pts = a + u_pts * (b - a)
            u_pts = u_pts.view(1,-1,3).to(device)

            _,sdf = self.im_network(None, model_z, u_pts, is_training=False)
            sdf = sdf.view(1,64,64,64).cpu().detach()
            sdf = -(sdf - isosurface).flip((1,3))

            mini = v.min(dim=0)[0]
            maxi = v.max(dim=0)[0]
            center = (mini + maxi)/2.
            scale_factor = torch.prod(maxi - mini).pow(1./3.)
            fn = lambda x: torch.stack(x,dim=-1)
      
            v = (v - center) / scale_factor
            v = fn((-v[:,0],v[:,1],-v[:,2]))

            grid_min = v.min(dim=0)[0]
            grid_max = v.max(dim=0)[0]

            dic = {'sdf':sdf,'grid_min':grid_min,'grid_max':grid_max}

            item = dic,v,f,None
            torch.save(item,f'{config.sample_dir}/{t+begin}.pt')
            
            self.writeObj(v,f,f'meshes/{t+begin}.obj')


    #output shape as ply and point cloud as ply
    def test_mesh_point(self, config):
        #load previous checkpoint
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            self.im_network.load_state_dict(torch.load(model_dir))
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return

        self.im_network.eval()
        for t in range(config.start, min(len(self.data_voxels),config.end)):
            batch_voxels_ = self.data_voxels[t:t+1].astype(np.float32)
            batch_voxels = torch.from_numpy(batch_voxels_)
            batch_voxels = batch_voxels.to(self.device)
            model_z,_ = self.im_network(batch_voxels, None, None, is_training=False)
            model_float = self.z2voxel(model_z)

            vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
            vertices = (vertices.astype(np.float32)-0.5)/self.real_size-0.5
            #vertices = self.optimize_mesh(vertices,model_z)
            write_ply_triangle(config.sample_dir+"/"+str(t)+"_vox.ply", vertices, triangles)

            print("[sample]")

            #sample surface points
            sampled_points_normals = sample_points_triangle(vertices, triangles, 4096)
            np.random.shuffle(sampled_points_normals)
            write_ply_point_normal(config.sample_dir+"/"+str(t)+"_pc.ply", sampled_points_normals)

            print("[sample]")


    def get_z(self, config):
        #load previous checkpoint
        self.im_network.load_state_dict(torch.load('checkpoint/im_dict.pt'))

        hdf5_path = self.checkpoint_dir+'/'+self.model_dir+'/'+self.dataset_name+'_train_z.hdf5'
        print(hdf5_path)
        shape_num = len(self.data_voxels)
        hdf5_file = h5py.File(hdf5_path, mode='w')
        hdf5_file.create_dataset("zs", [shape_num,self.z_dim], np.float32)

        self.im_network.eval()
        print(shape_num)
        for t in range(shape_num):
            batch_voxels = self.data_voxels[t:t+1].astype(np.float32)
            batch_voxels = torch.from_numpy(batch_voxels)
            batch_voxels = batch_voxels.to(self.device)
            out_z,_ = self.im_network(batch_voxels, None, None, is_training=False)
            hdf5_file["zs"][t:t+1,:] = out_z.detach().cpu().numpy()

        hdf5_file.close()
        print("[z]")


    def test_z(self, config, batch_z, dim):
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return

        for t in range(batch_z.shape[0]):
            model_z = batch_z[t:t+1]
            model_z = torch.from_numpy(model_z)
            model_z = model_z.to(self.device)
            model_float = self.z2voxel(model_z)
            #img1 = np.clip(np.amax(model_float, axis=0)*256, 0,255).astype(np.uint8)
            #img2 = np.clip(np.amax(model_float, axis=1)*256, 0,255).astype(np.uint8)
            #img3 = np.clip(np.amax(model_float, axis=2)*256, 0,255).astype(np.uint8)
            #cv2.imwrite(config.sample_dir+"/"+str(t)+"_1t.png",img1)
            #cv2.imwrite(config.sample_dir+"/"+str(t)+"_2t.png",img2)
            #cv2.imwrite(config.sample_dir+"/"+str(t)+"_3t.png",img3)

            vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
            vertices = (vertices.astype(np.float32)-0.5)/self.real_size-0.5
            #vertices = self.optimize_mesh(vertices,model_z)
            write_ply(config.sample_dir+"/"+"out"+str(t)+".ply", vertices, triangles)

            print("[sample Z]")

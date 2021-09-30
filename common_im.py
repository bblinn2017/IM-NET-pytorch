import os
from os import listdir
import shutil
import zipfile
import open3d as o3d
import torch

device = torch.device('cuda')

b_mean,b_std = torch.load('betas_params.pt')

def sample_betas(batch_size=1,is_default=False):
    if is_default:
        return default.repeat(batch_size,1)
    return (b_mean + torch.randn((batch_size,b_std.shape[-1])) * b_std).float()

def visualize_pcd(points,colors = None,save=None):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if save is None:
        o3d.visualization.draw_geometries([pcd])
    else:
        o3d.io.write_point_cloud(save,pcd)

def newdir(name,remove=True):

    if os.path.isdir(name) and remove:
        shutil.rmtree(name)
    os.mkdir(name)

def zipdir(name):

    zipf = zipfile.ZipFile(f'{name}.zip','w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(name):
        for file in files:
            zipf.write(os.path.join(root,file))
    zipf.close()

def itemsdir(name,ends=None):

    if ends is None:
        return [f'{name}/{f}' for f in listdir(name)]
    return [f'{name}/{f}' for f in listdir(name) if f.endswith(ends)]


def cdf(vals,titles=None,filename='plot_cdf.png'):
    plt.clf()

    if isinstance(vals[0],float):
        vals = np.array(vals).reshape(1,-1)
        v = vals
    else:
        v = np.concatenate(vals)

    p = np.linspace(0,1,len(vals[0]))
    handles = []
    for i in range(len(vals)):
        h, = plt.plot(np.sort(vals[i]),p)
        handles.append(h)

    if titles is not None:
        plt.legend(handles,titles)

    plt.savefig(filename)

def histogram(vals,rows=1,cols=1,rang=None,titles=None,ymax=None,filename='plot_pdf.png'):

    plt.clf()
    fig = plt.figure(figsize=(10,10))
    gs = fig.add_gridspec(nrows=rows,ncols=cols,figure=fig)
    axs = gs.subplots()

    if isinstance(vals[0],float):
        vals = np.array(vals).reshape(1,-1)
        v = vals
    elif not isinstance(vals,np.ndarray):
        v = np.array(vals)
    else:
        v = vals

    if rang is None:
        mini,maxi = np.min(v),np.max(v)
    else:
        mini,maxi = rang

    hists = []
    hist_max = 0.

    for i in range(len(v)):
        hist,edges = np.histogram(v[i],range=(mini,maxi))
        hist = hist / len(v[i])

        hist_max = max(max(hist),hist_max)
        hists.append(hist)

    if ymax is None:
        lim = hist_max * 1.01
    else:
        lim = ymax
    edges = edges[:-1]
    width = edges[1] - edges[0]

    for i in range(len(vals)):
        if rows == 1 and cols == 1:
            ax = axs
        elif rows == 1 or cols == 1:
            ax = axs[i]
        else:
            ax = axs[i//cols,i%cols]

        ax.bar(edges,hists[i],width,align='edge')
        if titles is not None:
            ax.title.set_text(titles[i])
        ax.set_ylim([0,lim])

    plt.show()
    plt.savefig(filename)

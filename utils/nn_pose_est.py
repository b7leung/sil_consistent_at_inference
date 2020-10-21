
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from pytorch3d.renderer import look_at_view_transform
import pandas as pd
from tqdm.autonotebook import tqdm

from utils import general_utils, network_utils

class PoseEstimationNetwork(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.input_dim = 3
        hidden_dim = 3
        act = nn.LeakyReLU()
        self.pose_est_net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, 3)
        )

        network_utils.weights_init_normal(self.pose_est_net, var=0.000001)


    # outputs pred_azim, pred_elev, pred_dist
    def forward(self, init_azim, init_elev, init_dist):
        if init_azim is not None and init_elev is not None and init_dist is not None:
            input = torch.tensor([[init_azim, init_elev, init_dist]]).to(self.device)
        else:
            input = torch.ones((1,self.input_dim)).to(self.device)
        est_pose_logits = self.pose_est_net(input)
        #est_pose = torch.sigmoid(est_pose_logits)
        #pred_azim = est_pose[0][0] * 360
        #pred_elev = est_pose[0][1] * 180 # 360 instead?
        #pred_dist = est_pose[0][2] * 3 # not sure if 3 is the best number
        pred_azim = est_pose_logits[0][0]
        pred_elev = est_pose_logits[0][1]
        pred_dist = est_pose_logits[0][2]
        return pred_azim, pred_elev, pred_dist


def nn_optimization_estimate_pose(mesh, rgba_image, iterations, lr, device, init_azim=None, init_elev=None, init_dist=None):

    mesh = mesh.to(device)    

    mask = rgba_image[:,:,3] > 0
    mask = torch.unsqueeze(torch.tensor(mask, dtype=torch.float), 0).to(device)

    pose_est_net = PoseEstimationNetwork(device)
    pose_est_net.to(device)
    optimizer = optim.Adam(pose_est_net.parameters(), lr=lr, weight_decay=0.01)
    loss_info = pd.DataFrame()

    best_pred_azim = init_azim
    best_pred_elev = init_elev
    best_pred_dist = init_dist
    R, T = look_at_view_transform(init_dist, init_elev, init_azim)
    init_render = general_utils.render_mesh(mesh, R, T, device, img_size=224, silhouette=True)
    init_silhouette = init_render[:, :, :, 3]
    best_sil_loss = F.binary_cross_entropy(init_silhouette, mask).item()
    loss_info = loss_info.append({"iteration":-1, "sil_loss": best_sil_loss}, ignore_index=True)

    for i in tqdm(range(iterations)):

        pred_azim, pred_elev, pred_dist = pose_est_net.forward(init_azim, init_elev, init_dist)
        R, T = look_at_view_transform(pred_dist, pred_elev, pred_azim) 
        render = general_utils.render_mesh(mesh, R, T, device, img_size=224, silhouette=True)
        silhouette = render[:, :, :, 3]

        sil_loss = F.binary_cross_entropy(silhouette, mask)
        sil_loss.backward()
        optimizer.step()

        loss_info = loss_info.append({"iteration":i, "sil_loss": sil_loss.item()}, ignore_index=True)
        if sil_loss.item() < best_sil_loss:
            best_sil_loss = sil_loss.item()
            best_pred_azim = pred_azim.item()
            best_pred_elev = pred_elev.item()
            best_pred_dist = pred_dist.item()
            print("updated best pose to {}".format([best_pred_azim, best_pred_elev, best_pred_dist]))
            
    
    return best_pred_azim, best_pred_elev, best_pred_dist, loss_info

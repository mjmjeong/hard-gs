import warnings

warnings.filterwarnings("ignore")

import json
import os
import random

import numpy as np
import torch
from PIL import Image
import math
from tqdm import tqdm
from scene.utils import Camera
from typing import NamedTuple
from torch.utils.data import Dataset
from utils.general_utils import PILtoTorch
# from scene.dataset_readers import 
import torch.nn.functional as F
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.pose_utils import smooth_camera_poses

from scene.colmap_loader import read_points3D_binary
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    mask: np.array = None
    depth: np.array = None
    flow_cams: dict = None

class Load_hyper_data(Dataset):
    def __init__(self, 
                 datadir, 
                 ratio=1.0,
                 use_bg_points=False,
                 split="train",
                 args=None
                 ):
        
        from .utils import Camera
        datadir = os.path.expanduser(datadir)
        self.datadir = datadir
        with open(f'{datadir}/scene.json', 'r') as f:
            scene_json = json.load(f)
        with open(f'{datadir}/metadata.json', 'r') as f:
            meta_json = json.load(f)
        with open(f'{datadir}/dataset.json', 'r') as f:
            dataset_json = json.load(f)

        self.flow_intervals = args.flow_intervals
        self.near = scene_json['near']
        self.far = scene_json['far']
        self.coord_scale = scene_json['scale']
        self.scene_center = scene_json['center']
        self.flow_cam_dict = {}
        self.all_img = dataset_json['ids']
        self.val_id = dataset_json['val_ids']
        self.split = split
        if len(self.val_id) == 0:
            self.i_train = np.array([i for i in np.arange(len(self.all_img)) if
                            (i%4 == 0)])
            self.i_test = self.i_train+2
            self.i_test = self.i_test[:-1,]
        else:
            self.train_id = dataset_json['train_ids']
            self.i_test = []
            self.i_train = []
            for i in range(len(self.all_img)):
                id = self.all_img[i]
                if id in self.val_id:
                    self.i_test.append(i)
                if id in self.train_id:
                    self.i_train.append(i)

        self.all_cam = [meta_json[i]['camera_id'] for i in self.all_img]
        self.all_time = [meta_json[i]['warp_id'] for i in self.all_img]
        max_time = max(self.all_time)
        self.all_time = [meta_json[i]['warp_id']/max_time for i in self.all_img]
        self.selected_time = set(self.all_time)
        self.ratio = ratio
        self.max_time = max(self.all_time)
        self.min_time = min(self.all_time)
        self.i_video = [i for i in range(len(self.all_img))]
        self.i_video.sort()
        self.all_cam_params = []
        for im in self.all_img:
            camera = Camera.from_json(f'{datadir}/camera/{im}.json')

            self.all_cam_params.append(camera)
        self.all_img_origin = self.all_img
        self.depth_type = args.depth_type

        # if refined depth is exists use that
        if self.split == 'train':
            if self.depth_type == 'calibrate':
                self.all_depth = [f'{datadir}/calibrated_depth/{int(1/ratio)}x_sparse/{i}.npy' for i in self.all_img]
                #self.all_depth = [f'{datadir}/processed_depth/1x/.npy' for i in self.all_img]
                self.depth_is_calibrated = True

                for d_idx in self.i_train:
                    depth_file_name = self.all_depth[d_idx]
                    if not os.path.isfile(depth_file_name):
                        print(depth_file_name, 'is not exists!')
                        self.all_depth = [f'{datadir}/midas_depth/{int(1/ratio)}x/{i}-dpt_beit_large_512.png' for i in self.all_img]
                        self.depth_is_calibrated = False
                        break
            elif self.depth_type == 'preprocess':
                self.all_depth = [f'{datadir}/depth/1x/{i}.npy' for i in self.all_img] # TODO
                self.depth_is_calibrated = True

        else:
            self.all_depth = [f'{datadir}/midas_depth/{int(1/ratio)}x/{i}-dpt_beit_large_512.png' for i in self.all_img]
            self.depth_is_calibrated = False

        self.all_img = [f'{datadir}/rgb/{int(1/ratio)}x/{i}.png' for i in self.all_img]

        self.h, self.w = self.all_cam_params[0].image_shape
        self.map = {}
        self.image_one = Image.open(self.all_img[0])
        self.image_one_torch = PILtoTorch(self.image_one,None).to(torch.float32)
        if os.path.exists(os.path.join(datadir,"covisible")):
            self.image_mask = [f'{datadir}/covisible/{int(2)}x/val/{i}.png' for i in self.all_img_origin]
        else:
            self.image_mask = None
#        self.generate_video_path()

        # load_prompt
        with open(f'{datadir}/text_prompt/prompt_blip.txt', "r")  as file:
            string = file.readlines()
            file.close()
        self.prompt = string[0].replace('\n','')        

    def load_flow_cams(self,idx):
        image_name = self.all_img[idx].split("/")[-1]
        if self.split == 'train':
            # existing dict
            if image_name in self.flow_cam_dict.keys():
                return self.flow_cam_dict[image_name]

            # load
            curr_id = image_name.split('_')[1].split('.')[0]
            prefix = image_name.split('_')[0] + '_'
            flow_path = os.path.join(self.datadir, 'optical_flow')
            # get flows
            flow_dict = {}
            for i in self.flow_intervals:
                # forwward
                flow_cam_fwd = f'{(int(curr_id) + i):05d}'
                flow_path_fwd = os.path.join(flow_path, f'i_{i}', curr_id  + '_' + flow_cam_fwd +'.npz')
                image_name_fwd = prefix + flow_cam_fwd + '.png'
                if os.path.isfile(flow_path_fwd):
                    flow_dict[image_name_fwd] = flow_path_fwd
                # backwward
                flow_cam_bwd = f'{(int(curr_id) - i):05d}'
                flow_path_bwd = os.path.join(flow_path, f'i_{i}', curr_id  + '_' + flow_cam_bwd +'.npz')
                image_name_bwd = prefix + flow_cam_bwd + '.png'
                if os.path.isfile(flow_path_bwd):
                    flow_dict[image_name_bwd] = flow_path_bwd
            self.flow_cam_dict[image_name] = flow_dict
            return flow_dict
        else:
            return None
    def generate_video_path(self):
        
        self.select_video_cams = [item for i, item in enumerate(self.all_cam_params) if i % 1 == 0 ]
        self.video_path, self.video_time = smooth_camera_poses(self.select_video_cams,10)
        # breakpoint()
        self.video_path = self.video_path[:500]
        self.video_time = self.video_time[:500]
        # breakpoint()
    def __getitem__(self, index):
        if self.split == "train":
            return self.load_raw(self.i_train[index])
 
        elif self.split == "test":
            return self.load_raw(self.i_test[index])
        elif self.split == "video":
            return self.load_video(index)
    def __len__(self):
        if self.split == "train":
            return len(self.i_train)
        elif self.split == "test":
            return len(self.i_test)
        elif self.split == "video":
            return len(self.video_path)
            # return len(self.video_v2)
    def load_video(self, idx):
        if idx in self.map.keys():
            return self.map[idx]
        camera = self.all_cam_params[idx]
        # camera = self.video_path[idx]
        w = self.image_one.size[0]
        h = self.image_one.size[1]
        # image = PILtoTorch(image,None)
        # image = image.to(torch.float32)
        time = self.video_time[idx]
        # .astype(np.float32)
        R = camera.orientation.T
        T = - camera.position @ R
        FovY = focal2fov(camera.focal_length, self.h)
        FovX = focal2fov(camera.focal_length, self.w)
        image_path = "/".join(self.all_img[idx].split("/")[:-1])
        image_name = self.all_img[idx].split("/")[-1]
        caminfo = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=self.image_one_torch,
                              image_path=image_path, image_name=image_name, width=w, height=h, fid=time, mask=None
                              )
        self.map[idx] = caminfo
        return caminfo  
    def load_raw(self, idx):
        if idx in self.map.keys():
            return self.map[idx]
        camera = self.all_cam_params[idx]
        image = Image.open(self.all_img[idx])
        w = image.size[0]
        h = image.size[1]
        image = PILtoTorch(image,None)
        image = image.to(torch.float32)[:3,:,:]
        time = self.all_time[idx]
        R = camera.orientation.T
        T = - camera.position @ R
        FovY = focal2fov(camera.focal_length, self.h)
        FovX = focal2fov(camera.focal_length, self.w)
        image_path = "/".join(self.all_img[idx].split("/")[:-1])
        image_name = self.all_img[idx].split("/")[-1]
        
        if os.path.exists(self.all_depth[idx]):
            if self.all_depth[idx].endswith('.npy'):
                if self.depth_type == 'calibrate':
                    depth = np.load(self.all_depth[idx])
                    depth = torch.tensor(depth).to(torch.float32).unsqueeze(0)
                elif  self.depth_type == 'preprocess' or 'default':
                    depth = np.load(self.all_depth[idx])
                    depth = torch.tensor(depth).to(torch.float32)
                    depth = depth.permute(2,0,1).unsqueeze(0) #1,1,H,W
                    depth = F.interpolate(depth, (h,w))[0]
            else:
                depth = Image.open(self.all_depth[idx])
                depth = PILtoTorch(depth,None)
                depth = depth.to(torch.float32)
        else:
            depth = None
            
        if self.image_mask is not None and self.split == "test":
            mask = Image.open(self.image_mask[idx])
            mask = PILtoTorch(mask,None)
            mask = mask.to(torch.float32)[0:1,:,:]
            size_ = tuple([int(h), int(w)])
#            mask = F.interpolate(mask.unsqueeze(0), size=size_, mode='bilinear', align_corners=False).squeeze(0)
        else:
            mask = None

        flow_cams = self.load_flow_cams(idx)
        
        caminfo = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=w, height=h, fid=time, mask=mask,
                              depth=depth, flow_cams=flow_cams,
                              )
        self.map[idx] = caminfo
        return caminfo  


    def do_depth_calibration(self, pcd, point_type="sparse"):
        for idx in range(len(self.i_train)):
            self.calibrate_depth_idx(self.i_train[idx], pcd, point_type=point_type)
        #for idx in range(len(self.i_test)):
        #    self.calibrate_depth_idx(self.i_test[idx], pcd, point_type=point_type)
        self.depth_is_calibrated = True 

    def calibrate_depth_idx(self, idx, pcd, point_type="sparse"):
        # camera info
        caminfo = self.load_raw(idx)
        R = caminfo.R
        T = caminfo.T
        FovY = caminfo.FovY
        FovX = caminfo.FovX
        h = caminfo.height
        w = caminfo.width
        focal_y = fov2focal(FovY, h)
        focal_x = fov2focal(FovX, w)


        if os.path.isfile(os.path.join(self.datadir, 'calibrated_depth', f'2x_{point_type}', caminfo.image_name.replace('.png', '.npy'))):
            return 

        # Target Depth from COLMAP
        depthmap, depth_weight = np.zeros((h,w)), np.zeros((h,w))
        K = np.array([[focal_x, 0, w/2],[0,focal_y,h/2],[0,0,1]])
        cam_coord = np.matmul(K, np.matmul(R.transpose(), pcd.points.transpose()) + T.reshape(3,1)) ### for coordinate definition, see getWorld2View2() function
        valid_idx = np.where(np.logical_and.reduce((cam_coord[2]>0, cam_coord[0]/cam_coord[2]>=0, cam_coord[0]/cam_coord[2]<=w-1, cam_coord[1]/cam_coord[2]>=0, cam_coord[1]/cam_coord[2]<=h-1)))[0]
        pts_depths = cam_coord[-1:, valid_idx]
        cam_coord = cam_coord[:2, valid_idx]/cam_coord[-1:, valid_idx]
        depthmap[np.round(cam_coord[1]).astype(np.int32).clip(0,h-1), np.round(cam_coord[0]).astype(np.int32).clip(0,w-1)] = pts_depths
        depth_weight[np.round(cam_coord[1]).astype(np.int32).clip(0,h-1), np.round(cam_coord[0]).astype(np.int32).clip(0,w-1)] = 1/pcd.errors[valid_idx] if pcd.errors is not None else 1
        depth_weight = depth_weight/depth_weight.max()
    
        # Source Depth
        source_depth = np.array(caminfo.depth)[0]
        source_depth = (source_depth-source_depth.min()) / (source_depth.max()-source_depth.min())
        source_depth = source_depth.max() - source_depth
        source_depth = source_depth * depthmap.max()

        # Refine
        print(f"start refine index: {idx}")
        refine_depth, depthloss = self.optimize_depth(source=source_depth, target=depthmap, mask=depthmap>0.0, depth_weight=depth_weight)

        # Check with images visualization
        
        import cv2
        tmp_color = np.zeros((h,w,3))
        tmp_color[np.round(cam_coord[1]).astype(np.int32).clip(0,h-1), np.round(cam_coord[0]).astype(np.int32).clip(0,w-1), :] = pcd.colors[valid_idx]
        #cv2.imwrite(f"debug/{idx:03d}_rgb.png", tmp_color[:,:,::-1]*255)
        
        os.makedirs(os.path.join(self.datadir, 'calibrated_depth', f'2x_{point_type}'), exist_ok=True)
        os.makedirs(os.path.join(self.datadir, 'calibrated_depth', f'2x_{point_type}_vis'), exist_ok=True)
        np.save(os.path.join(self.datadir, 'calibrated_depth', f'2x_{point_type}', caminfo.image_name.replace('.png', '.npy')), refine_depth)

        #vis_scale = depthmap.max() + 0.5
        vis_scale = 2.0
        target_rgb = depthmap.reshape(h,w,1).repeat(3,2) /vis_scale
        refine_rgb = refine_depth.reshape(h,w,1).repeat(3,2) / vis_scale
        source_rgb = source_depth.reshape(h,w,1).repeat(3,2) / vis_scale
        target_mask = (depthmap>0).reshape(h,w,1).repeat(3,2)
        
        save_img = np.concatenate((tmp_color[:,:,::-1],target_rgb, source_rgb, refine_rgb),1)
        cv2.imwrite(os.path.join(self.datadir, 'calibrated_depth', f'2x_{point_type}_vis', f"{idx:03d}.png"), save_img*255)
    
    def optimize_depth(self, source, target, mask, depth_weight, prune_ratio=0.001):
        """
        Arguments
        =========
        source: np.array(h,w)
        target: np.array(h,w)
        mask: np.array(h,w):
            array of [True if valid pointcloud is visible.]
        depth_weight: np.array(h,w):
            weight array at loss.
        Returns
        =======
        refined_source: np.array(h,w)
            literally "refined" source.
        loss: float
        """
        source = torch.from_numpy(source).cuda()
        target = torch.from_numpy(target).cuda()
        mask = torch.from_numpy(mask).cuda()
        depth_weight = torch.from_numpy(depth_weight).cuda()

        # Prune some depths considered "outlier"     
        with torch.no_grad():
            target_depth_sorted = target[target>1e-7].sort().values
            min_prune_threshold = target_depth_sorted[int(target_depth_sorted.numel()*prune_ratio)]
            max_prune_threshold = target_depth_sorted[int(target_depth_sorted.numel()*(1.0-prune_ratio))]

            mask2 = target > min_prune_threshold
            mask3 = target < max_prune_threshold
            mask = torch.logical_and( torch.logical_and(mask, mask2), mask3)

        source_masked = source[mask]
        target_masked = target[mask]
        depth_weight_masked = depth_weight[mask]
        # tmin, tmax = target_masked.min(), target_masked.max()

        # # Normalize
        # target_masked = target_masked - tmin 
        # target_masked = target_masked / (tmax-tmin)


        scale = torch.ones(1).cuda().requires_grad_(True)
        shift = (torch.ones(1) * 0.5).cuda().requires_grad_(True)

        optimizer = torch.optim.Adam(params=[scale, shift], lr=1.0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8**(1/100))
        loss = torch.ones(1).cuda() * 1e5

        iteration = 1
        loss_prev = 1e6
        loss_ema = 0.0

        while abs(loss_ema - loss_prev) > 1e-5:
            source_hat = scale*source_masked + shift
            loss = torch.mean(((target_masked - source_hat)**2)*depth_weight_masked)

            # penalize depths not in [0,1]
            loss_hinge1 = loss_hinge2 = 0.0
            if (source_hat<=0.0).any():
                loss_hinge1 = 2.0*((source_hat[source_hat<=0.0])**2).mean()
            # if (source_hat>=1.0).any():
            #     loss_hinge2 = 0.3*((source_hat[source_hat>=1.0])**2).mean() 
            
            loss = loss + loss_hinge1 + loss_hinge2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            iteration+=1
            if iteration % 1000 == 0:
                print(f"ITER={iteration:6d} loss={loss.item():8.4f}, params=[{scale.item():.4f},{shift.item():.4f}], lr={optimizer.param_groups[0]['lr']:8.4f}")
                loss_prev = loss.item()
            loss_ema = loss.item() * 0.2 + loss_ema * 0.8
    
        loss = loss.item()
        print(f"loss ={loss:10.5f}, iteration={iteration}")

        with torch.no_grad():
            refined_source = (scale*source + shift) 
        torch.cuda.empty_cache()
        return refined_source.cpu().numpy(), loss

def format_hyper_data(data_class, split):
    if split == "train":
        data_idx = data_class.i_train
    elif split == "test":
        data_idx = data_class.i_test
    # dataset = data_class.copy()
    # dataset.mode = split
    cam_infos = []
    for uid, index in tqdm(enumerate(data_idx)):
        camera = data_class.all_cam_params[index]
        # image = Image.open(data_class.all_img[index])
        # image = PILtoTorch(image,None)
        time = data_class.all_time[index]
        R = camera.orientation.T
        T = - camera.position @ R
        FovY = focal2fov(camera.focal_length, data_class.h)
        FovX = focal2fov(camera.focal_length, data_class.w)
        image_path = "/".join(data_class.all_img[index].split("/")[:-1])
        image_name = data_class.all_img[index].split("/")[-1]
        
        if data_class.image_mask is not None and data_class.split == "test":
            mask = Image.open(data_class.image_mask[index])
            mask = PILtoTorch(mask,None)
            
            mask = mask.to(torch.float32)[0:1,:,:]
            
        
        else:
            mask = None
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=None,
                              image_path=image_path, image_name=image_name, width=int(data_class.w), 
                              height=int(data_class.h), fid=time, mask=mask
                              )
        cam_infos.append(cam_info)
    return cam_infos
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from scene import Scene
import uuid
import numpy as np

from utils.image_utils import psnr, lpips, alex_lpips
from utils.image_utils import ssim as ssim_func
from piq import LPIPS
lpips = LPIPS()
from argparse import Namespace
from pytorch_msssim import ms_ssim
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

@torch.no_grad()
def render_training_image(iteration, time_now, scene: Scene, renderFunc, renderArgs, deform, load2gpu_on_the_fly, type_='train', prefix=""):
    torch.cuda.empty_cache()
    
    ############################################################
    # select cam
    ############################################################
    if type_ == 'train':
        viewpoints = scene.getTrainCameras()
    elif type_ == 'test':
        viewpoints = scene.getTestCameras()
    
    np.random.seed(iteration)
    viewpoint = np.random.choice(viewpoints, 1)[0]

    ############################################################
    # label
    ############################################################
    label1 = f"iter:{iteration}"
    times =  time_now/60
    if times < 1:
        end = "min"
    else:
        end = "mins"
    label2 = "time:%.2f" % times + end
    
    render_base_path = os.path.join(scene.model_path, f"images", prefix + str(type_))
    os.makedirs(render_base_path, exist_ok=True)
    path = os.path.join(render_base_path, f"{iteration}.jpg")

    ############################################################
    # rendering
    ############################################################
    if load2gpu_on_the_fly:
        viewpoint.load2device()
    fid = viewpoint.fid
    xyz = scene.gaussians.get_xyz

    if deform.name == 'mlp':
        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
    elif deform.name == 'node':
        time_input = deform.deform.expand_time(fid)
    else:
        time_input = 0

    d_values = deform.step(xyz.detach(), time_input, feature=scene.gaussians.feature, is_training=False, motion_mask=scene.gaussians.motion_mask, camera_center=viewpoint.camera_center)
    d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
    
    # GT
    gt_image = torch.clamp(viewpoint.original_image, 0.0, 1.0)
    gt_image = gt_image.permute(1, 2, 0).cpu().numpy()
    gt_depth = viewpoint.depth
    if gt_depth is not None:
        gt_depth_np = gt_depth.permute(1,2,0).cpu().numpy()
        gt_depth_np /= gt_depth_np.max()
        #gt_depth_np = 1 - gt_depth_np
        gt_depth_np = np.repeat(gt_depth_np, 3, axis=2)
    
    # Render
    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=deform.d_rot_as_res)           
    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
    depth = render_pkg["depth"]
    image_np = image.permute(1, 2, 0).cpu().numpy()  # 转换通道顺序为 (H, W, 3)
    depth_np = depth.permute(1, 2, 0).cpu().numpy()
    depth_np /= depth_np.max()
    depth_np = np.repeat(depth_np, 3, axis=2)
    
    if gt_depth is not None:
        image_np = np.concatenate((gt_image, image_np, gt_depth_np, depth_np), axis=1)
    else:
        image_np = np.concatenate((gt_image, image_np, depth_np), axis=1)

    ############################################################
    # save
    #############################################################
    image_with_labels = Image.fromarray((np.clip(image_np,0,1) * 255).astype('uint8')) 
    draw1 = ImageDraw.Draw(image_with_labels)
    font = ImageFont.truetype('./utils/TIMES.TTF', size=40)
    text_color = (255, 0, 0)  # 白色

    label1_position = (10, 10)
    label2_position = (image_with_labels.width - 100 - len(label2) * 10, 10)  # 右上角坐标

    draw1.text(label1_position, label1, fill=text_color, font=font)
    draw1.text(label2_position, label2, fill=text_color, font=font)
    image_with_labels.save(path)

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs, deform, load2gpu_on_the_fly, progress_bar=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    test_ssim = 0.0
    test_lpips = 1e10
    test_ms_ssim = 0.0
    test_alex_lpips = 1e10
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                # images = torch.tensor([], device="cuda")
                # gts = torch.tensor([], device="cuda")
                psnr_list, ssim_list, lpips_list, l1_list = [], [], [], []
                ms_ssim_list, alex_lpips_list = [], []
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz

                    if deform.name == 'mlp':
                        time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    elif deform.name == 'node':
                        time_input = deform.deform.expand_time(fid)
                    else:
                        time_input = 0

                    d_values = deform.step(xyz.detach(), time_input, feature=scene.gaussians.feature, is_training=False, motion_mask=scene.gaussians.motion_mask, camera_center=viewpoint.camera_center)
                    d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']

                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz=d_xyz, d_rotation=d_rotation, d_scaling=d_scaling, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=deform.d_rot_as_res)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    l1_list.append(l1_loss(image[None], gt_image[None]).mean())
                    psnr_list.append(psnr(image[None], gt_image[None]).mean())
                    ssim_list.append(ssim_func(image[None], gt_image[None], data_range=1.).mean())
                    lpips_list.append(lpips(image[None], gt_image[None]).mean())
                    ms_ssim_list.append(ms_ssim(image[None], gt_image[None], data_range=1.).mean())
                    alex_lpips_list.append(alex_lpips(image[None], gt_image[None]).mean())

                    # images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    # gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                l1_test = torch.stack(l1_list).mean()
                psnr_test = torch.stack(psnr_list).mean()
                ssim_test = torch.stack(ssim_list).mean()
                lpips_test = torch.stack(lpips_list).mean()
                ms_ssim_test = torch.stack(ms_ssim_list).mean()
                alex_lpips_test = torch.stack(alex_lpips_list).mean()
                total_point = scene.gaussians._xyz.shape[0]

                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                    test_ssim = ssim_test
                    test_lpips = lpips_test
                    test_ms_ssim = ms_ssim_test
                    test_alex_lpips = alex_lpips_test
                    
                if progress_bar is None:
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} MS SSIM{} ALEX_LPIPS {} NUM {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test, ms_ssim_test, alex_lpips_test, total_point))
                else:
                    progress_bar.set_description("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {} MS SSIM {} ALEX_LPIPS {} NUM {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_test, ms_ssim_test, alex_lpips_test, total_point))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', test_ssim, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', test_lpips, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ms-ssim', test_ms_ssim, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - alex-lpips', test_alex_lpips, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr, test_ssim, test_lpips, test_ms_ssim, test_alex_lpips


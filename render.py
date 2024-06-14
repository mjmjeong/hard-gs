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

import torch
from scene import Scene, DeformModel
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from utils.pose_utils import pose_spherical
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams, LoggerParams

import csv
from gaussian_renderer import GaussianModel
import imageio
import numpy as np
from pytorch_msssim import ms_ssim
from piq import LPIPS
lpips = LPIPS()
from utils.image_utils import ssim as ssim_func
from utils.image_utils import psnr, lpips, alex_lpips
from pathlib import Path
from time import time
# iphone
from thirdparty.dycheck.core import metrics
from utils.loss_utils import l1_loss, ssim, l2_loss, lpips_loss, masked_mean
import jax.numpy as jnp
compute_lpips = metrics.get_compute_lpips() 

def render_set(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background, deform, skip_image_save=False, skip_measure=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")


    if not skip_image_save:
        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)
        makedirs(depth_path, exist_ok=True)

    # measure
    if not skip_measure:
        csv_path = os.path.join(model_path, 'result.csv')
    exp_name = Path(model_path).name
    #if csv_path is not None:
    #    
    #    csv_dir = Path(csv_path).parent.absolute()
    #    makedirs(csv_dir, exist_ok=True)
    times = []
    # Measurement
    mask_measure = False
    l1_test, psnr_test, mask_psnr_test, ssim_test = [], [], [], []
    jax_ssim_test, jax_mask_ssim_test = [], []
    jax_lpips_test, jax_mask_lpips_test = [], []

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    renderings = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if load2gpt_on_the_fly:
            view.load2device()
        fid = view.fid
        xyz = gaussians.get_xyz

        time1=time()

        if deform.name == 'mlp':
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        elif deform.name == 'node':
            time_input = deform.deform.expand_time(fid)
        d_values = deform.step(xyz.detach(), time_input, feature=gaussians.feature, motion_mask=gaussians.motion_mask)
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=deform.d_rot_as_res)
        time2=time()
        times.append(time2-time1)

        alpha = results["alpha"]
        rendering = torch.clamp(torch.cat([results["render"], alpha]), 0.0, 1.0)

        # Measurement
        if not skip_measure:
            image = rendering[:3]
            gt_image = torch.clamp(view.original_image.to("cuda"), 0.0, 1.0)
            l1_test.append(l1_loss(image, gt_image).mean().double()) 
            psnr_test.append(psnr(image, gt_image, mask=None).mean().double())

            # jax
            jax_render = jnp.array(image.permute(1,2,0).detach().cpu())
            jax_gt = jnp.array(gt_image.permute(1,2,0).detach().cpu())
            
            jax_ssim_test.append(metrics.compute_ssim(jax_render, jax_gt))
            jax_lpips_test.append(compute_lpips(jax_render, jax_gt))
            
            if name == 'test':    
                if view.mask is not None:
                    # psnr / ssim / lpips
                    jax_mask = jnp.array(view.mask.permute(1,2,0).detach().cpu())                                            
                    mask_psnr_test.append(psnr(image, gt_image, mask=view.mask.repeat(3,1,1)).mean().double())
                    jax_mask_ssim_test.append(metrics.compute_ssim(jax_render, jax_gt, jax_mask))
                    jax_mask_lpips_test.append(compute_lpips(jax_render, jax_gt, jax_mask))
                    mask_measure=True

        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        gt = view.original_image[0:4, :, :]
        if not skip_image_save:
            torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
    
    if not skip_image_save:
        renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
        imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)

    times = times[1:]
    fps_mean1 = len(times) / np.sum(times)
    fps_mean2 = np.mean([1/t for t in times])
    print("FPS:",fps_mean1, fps_mean2)

    # Measurement
    if not skip_measure:
        psnr_test = torch.stack(psnr_test).mean()
        jax_ssim_test = np.mean(jax_ssim_test)
        jax_lpips_test = np.mean(jax_lpips_test)

        if mask_measure:
            mask_psnr_test = torch.stack(mask_psnr_test).mean()
            jax_mask_ssim_test = np.mean(jax_mask_ssim_test)
            jax_mask_lpips_test = np.mean(jax_mask_lpips_test)

        if name == 'train':
            prefix='t-'
        else:
            prefix=''

        save_dict = {
                    'exp': exp_name, 
                    'stage': name, 
                    'iteration': iteration,
                    'FPS': fps_mean1,
                    'pts_num': gaussians.get_xyz.size(0),
                    f'{prefix}psnr': psnr_test.item(),
                    f'{prefix}ssim': jax_ssim_test,
                    f'{prefix}lpips': jax_lpips_test,
                    }
                    
        if mask_measure: 
            save_dict['m-psnr'] = mask_psnr_test.item()
            save_dict['m-ssim'] = jax_mask_ssim_test
            save_dict['m-lpips'] = jax_mask_lpips_test
            
        fieldnames = ['exp', 'stage', 'iteration','FPS', 'pts_num','psnr', 'm-psnr', 'ssim', 'm-ssim', 'lpips', 'm-lpips', 't-psnr', 't-ssim', 't-lpips']

        if not os.path.isfile(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(save_dict)
        else:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames )
                writer.writerow(save_dict)
        print(f"\n[ITER {iteration}] Evaluating {name}: PSNR {psnr_test} SSIM {jax_ssim_test} LPIPS {jax_lpips_test}")

def interpolate_time(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background, deform, skip_image_save='dummy'):
    render_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    frame = 150
    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]
    renderings = []
    for t in tqdm(range(0, frame, 1), desc="Rendering progress"):
        fid = torch.Tensor([t / (frame - 1)]).cuda()
        xyz = gaussians.get_xyz
        if deform.name == 'deform':
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        elif deform.name == 'node':
            time_input = deform.deform.expand_time(fid)
        d_values = deform.step(xyz.detach(), time_input, feature=gaussians.feature, motion_mask=gaussians.motion_mask)
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=deform.d_rot_as_res)
        rendering = results["render"]
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(t) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(t) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def interpolate_all(model_path, load2gpt_on_the_fly, name, iteration, views, gaussians, pipeline, background, deform, skip_image_save='dummy'):
    render_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "renders")
    depth_path = os.path.join(model_path, name, "interpolate_all_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    frame = 150
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, frame + 1)[:-1]], 0)
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)

    idx = torch.randint(0, len(views), (1,)).item()
    view = views[idx]  # Choose a specific time for rendering

    renderings = []
    for i, pose in enumerate(tqdm(render_poses, desc="Rendering progress")):
        fid = torch.Tensor([i / (frame - 1)]).cuda()

        matrix = np.linalg.inv(np.array(pose))
        R = -np.transpose(matrix[:3, :3])
        R[:, 0] = -R[:, 0]
        T = -matrix[:3, 3]

        view.reset_extrinsic(R, T)

        xyz = gaussians.get_xyz
        if deform.name == 'mlp':
            time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
        elif deform.name == 'node':
            time_input = deform.deform.expand_time(fid)

        d_values = deform.step(xyz.detach(), time_input, feature=gaussians.feature, motion_mask=gaussians.motion_mask)
        d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], d_values['d_scaling'], d_values['d_opacity'], d_values['d_color']
        results = render(view, gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, d_opacity=d_opacity, d_color=d_color, d_rot_as_res=deform.d_rot_as_res)
        rendering = torch.clamp(results["render"], 0.0, 1.0)
        renderings.append(to8b(rendering.cpu().numpy()))
        depth = results["depth"]
        depth = depth / (depth.max() + 1e-5)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(i) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(i) + ".png"))

    renderings = np.stack(renderings, 0).transpose(0, 2, 3, 1)
    imageio.mimwrite(os.path.join(render_path, 'video.mp4'), renderings, fps=30, quality=8)


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, mode: str, load2device_on_the_fly=False, skip_image_save=False, skip_measure=False):
    with torch.no_grad():
        
        deform = DeformModel(K=dataset.K, deform_type=dataset.deform_type, is_blender=dataset.is_blender, skinning=dataset.skinning, hyper_dim=dataset.hyper_dim, node_num=dataset.node_num, pred_opacity=dataset.pred_opacity, pred_color=dataset.pred_color, use_hash=dataset.use_hash, hash_time=dataset.hash_time, d_rot_as_res=dataset.d_rot_as_res, local_frame=dataset.local_frame, progressive_brand_time=dataset.progressive_brand_time, max_d_scale=dataset.max_d_scale)
        deform.load_weights(dataset.model_path, iteration=iteration)

        gs_fea_dim = deform.deform.node_num if dataset.skinning and deform.name == 'node' else dataset.hyper_dim
        gaussians = GaussianModel(dataset.sh_degree, fea_dim=gs_fea_dim, with_motion_mask=dataset.gs_with_motion_mask)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if mode == "render":
            render_func = render_set
        elif mode == "time":
            render_func = interpolate_time
        else:
            render_func = interpolate_all

        if not skip_train:
            render_func(dataset.model_path, load2device_on_the_fly, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, deform, skip_image_save, skip_measure)

        if not skip_test:
            render_func(dataset.model_path, load2device_on_the_fly, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, deform, skip_image_save, skip_measure)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    logp = LoggerParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_image_save", action="store_true")    
    parser.add_argument("--skip_measure", action="store_true")    
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mode", default='render', choices=['render', 'time', 'view', 'all', 'pose', 'original'])
    
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--elevation', type=float, default=0, help="default GUI camera elevation")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(10000, 80_0001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    # parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--deform-type", type=str, default='mlp')

    args = get_combined_args(parser)
    if not args.model_path.endswith(args.deform_type):
        args.model_path = os.path.join(os.path.dirname(os.path.normpath(args.model_path)), os.path.basename(os.path.normpath(args.model_path)) + f'_{args.deform_type}')
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.mode, load2device_on_the_fly=args.load2gpu_on_the_fly,
                skip_image_save=args.skip_image_save, skip_measure=args.skip_measure)

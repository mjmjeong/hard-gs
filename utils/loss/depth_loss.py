import torch
import torch.nn.functional as F
from torchmetrics.functional.regression import pearson_corrcoef
from utils.loss_utils import masked_mean
################################################
# Depth loss
################################################

def compute_depth_loss(render_depths, gt_depths, scene, opt):
                
    def min_max_norm(depth):
        return (depth-depth.min()) / (depth.max()-depth.min())
       
    if scene.args.depth_type == 'preprocess': 
        depth_mask = (gt_depths != 0)
        if opt.depth_loss_type == 'mse':
            depth_loss = (render_depths- gt_depths)**2
            depth_loss = masked_mean(depth_loss, depth_mask)

        elif opt.depth_loss_type == 'corrcoef':
            if not opt.no_depth_minmax_norm:
                render_depths = min_max_norm(render_depths)
                gt_depths = min_max_norm(gt_depths)

            depth_flat = render_depths.reshape(-1,1)
            gt_depth_flat = gt_depths.reshape(-1,1)
            depth_loss = 1 - pearson_corrcoef(gt_depth_flat, depth_flat)
    
    else:
        if opt.depth_loss_type == 'l1':
            depth_loss = F.l1_loss(render_depths, gt_depths)
        
        elif opt.depth_loss_type == 'corrcoef':
            if not opt.no_depth_minmax_norm:
                render_depths = min_max_norm(render_depths)
                gt_depths = min_max_norm(gt_depths)

            depth_flat = render_depths.reshape(-1,1)
            gt_depth_flat = gt_depths.reshape(-1,1)
            
            depth_loss = 1 - pearson_corrcoef(gt_depth_flat, depth_flat)
            
    return depth_loss
"""
https://gist.github.com/serycjon/3b3575ade427e1e855d18620348a1bc5
"""

import numpy as np
import cv2
from utils.flow.flow_arrow_viz.vis_utils import cv2_hatch

def show_flow(flow, src_img, dst_img, grid_sz=10,
              occl=None, occl_thr=255,
              arrow_color=(0, 0, 255),
              point_color=(0, 255, 255),
              occlusion_color=(217, 116, 0),
              decimal_places=2):
    """ Flow field visualization
    Args:
        flow             - <H x W x 3> flow with channels (u, v, _)
                            (u - flow in x direction, v - flow in y direction)
        src_img          - <H x W x 3> BGR flow source image
        dst_img          - <H x W x 3> BGR flow destination image
        grid_sz          - visualization grid size in pixels
        occl             - <H x W> np.uint8 soft occlusion mask (0-255)
        occl_thr         - (0-255) occlusion threshold (occl >= occl_thr means occlusion)
        arrow_color      - BGR 3-tuple of flow arrow color
        point_color      - BGR 3-tuple of flow point color
        occlusion_color  - BGR 3-tuple of flow point color
        decimal_places   - number of decimal places to be used for positions
    Returns:
        src_vis - <H x W x 3> BGR flow visualization in source image
        dst_vis - <H x W x 3> BGR flow visualization in destination image
    """
    pt_radius = 0
    line_type = cv2.LINE_AA  # cv2.LINE_8 or cv2.LINE_AA - antialiased, but blurry...
    circle_type = cv2.LINE_8

    shift = int(np.ceil(np.log2(10**decimal_places)))

    H, W = flow.shape[:2]

    src_xs = np.arange(W)
    src_ys = np.arange(H)

    xs, ys = np.meshgrid(src_xs, src_ys)
    flat_xs = xs.flatten()
    flat_ys = ys.flatten()

    pts_dst = flow[flat_ys, flat_xs, :2]
    pts_dst[:, 0] += flat_xs
    pts_dst[:, 1] += flat_ys

    pts_src = np.vstack((flat_xs, flat_ys))
    pts_dst = pts_dst.T

    mask = np.all(np.mod(pts_src, grid_sz) == 0, axis=0)

    pts_src = np.round(pts_src * (2**shift)).astype(np.int32)
    pts_dst = np.round(pts_dst * (2**shift)).astype(np.int32)

    src_vis = src_img.copy()
    dst_vis = dst_img.copy()

    # draw flow lines/arrows
    for i in range(mask.size):
        if mask[i]:
            a = pts_src[:, i]
            b = pts_dst[:, i]

            cv2.line(src_vis,
                     (a[0], a[1]),
                     (b[0], b[1]),
                     arrow_color,
                     lineType=line_type,
                     shift=shift)
            cv2.line(dst_vis,
                     (a[0], a[1]),
                     (b[0], b[1]),
                     arrow_color,
                     lineType=line_type,
                     shift=shift)

    # draw flow points
    for i in range(mask.size):
        if mask[i]:
            a = pts_src[:, i]
            b = pts_dst[:, i]

            if occl is not None and occl[np.unravel_index(i, occl.shape)] >= occl_thr:
                occluded = True
            else:
                occluded = False

            cv2.circle(src_vis,
                       (a[0], a[1]),
                       radius=pt_radius,
                       color=point_color,
                       lineType=circle_type,
                       shift=shift)
            cv2.circle(dst_vis,
                       (b[0], b[1]),
                       radius=pt_radius,
                       color=point_color if not occluded else occlusion_color,
                       lineType=circle_type,
                       shift=shift)

    return src_vis, dst_vis

def vis_flow_watercolors(src_flow, log=False, vmax=None, vmax_hatch=False, plot_legend=False):
    if vmax is None:
        flow = src_flow
    else:
        flow = src_flow.copy()
        flow_lengths = np.sqrt(np.sum(np.square(flow[:, :, :2]), axis=-1))
        longer = flow_lengths > vmax
        flow[longer, :2] *= np.expand_dims(vmax / flow_lengths[longer], axis=-1)

    if log:
        flow_to_vis = flow.copy()
        flow_lengths = np.sqrt(np.sum(np.square(flow[:, :, :2]), axis=-1))
        nonzero_lengths = flow_lengths > 0
        log_lengths = np.log(flow_lengths[nonzero_lengths] + 1)
        flow_to_vis[nonzero_lengths, :2] *= np.expand_dims(log_lengths / flow_lengths[nonzero_lengths], axis=-1)
        vmax = np.log(vmax + 1)
    else:
        flow_to_vis = flow

    vis = ip.flow2color_matlab_numpy(flow_to_vis, max_flow=vmax) / 255.0
    if vmax is not None and vmax_hatch:
        vis = cv2_hatch(vis*255, longer).astype(np.float32) / 255.0

    if plot_legend:
        legend_size = int(np.amin(src_flow.shape[:2]) / 6)
        legend = vis_flow_watercolors_wheel(legend_size)
        vis[-legend_size:, -legend_size:, :] = legend
    return vis

def vis_flow_watercolors_wheel(sz):
    center = (sz - 1) / 2.
    xs, ys = np.meshgrid(np.arange(sz), np.arange(sz))
    flow = np.dstack((xs - center, ys - center, np.ones_like(xs)))
    flow_lengths = np.sqrt(np.sum(np.square(flow[:, :, :2]), axis=-1))
    nonzero_lengths = flow_lengths > 0
    flow[nonzero_lengths, :2] /= flow_lengths[nonzero_lengths, np.newaxis]

    vis_water = vis_flow_watercolors(flow)
    return vis_water

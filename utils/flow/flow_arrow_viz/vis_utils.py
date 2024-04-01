from __future__ import print_function

import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

def cv2_hatch(canvas, mask, color=(0, 0, 0), alpha=1, **kwargs):
    """ Put a hatching over the canvas, where mask is True """
    hatching = hatch_pattern(canvas.shape[:2], **kwargs)
    hatch_mask = np.logical_and(mask,
                                hatching > 0)
    hatch_overlay = np.einsum("yx,c->yxc", hatch_mask, color).astype(np.uint8)
    alpha = np.expand_dims(hatch_mask * alpha, axis=2)
    vis = alpha * hatch_overlay + (1-alpha) * canvas
    return vis.astype(np.uint8)

def _hatch_pattern(shape, normal=(2, 1), spacing=10, **kwargs):
    """ Create a parralel line hatch pattern
    Args:
        shape - (H, W) canvas size
        normal - (x, y) line normal vector (doesn't have to be normalized)
        spacing - size of gap between the lines in pixels
    Outputs:
        canvas - <HxW> np.uint8 image with parallel lines, such that (normal_x, normal_y, c) * (c, r, 1) = 0
    """
    line_type = kwargs.get('line_type', cv2.LINE_8)

    H, W = shape[:2]
    canvas = np.zeros((H, W), dtype=np.uint8)
    normal = np.array(normal)
    normal = normal / np.sqrt(np.sum(np.square(normal)))

    corners = np.array([[0, 0],
                        [0, H],
                        [W, 0],
                        [W, H]])
    distances = np.einsum("ij,j->i", corners, normal)
    min_c = np.amin(distances)
    max_c = np.amax(distances)
    for c in np.arange(min_c, max_c, spacing):
        res = img_line_pts((H, W), (normal[0], normal[1], -c))
        if not res:
            continue
        else:
            pt_a, pt_b = res
            cv2.line(canvas,
                     tuple(int(x) for x in pt_a),
                     tuple(int(x) for x in pt_b),
                     255,
                     1,
                     line_type)
    return canvas

class HatchPatternMemo:
    def __init__(self):
        self.memo = {}

    def __call__(self, *args, **kwargs):
        arg_hash = (args, kwargs.get('normal'), kwargs.get('spacing'), kwargs.get('line_type'))

        if arg_hash not in self.memo:
            self.memo[arg_hash] = _hatch_pattern(*args, **kwargs)

        res = self.memo[arg_hash]
        return copy.deepcopy(res)

hatch_pattern = HatchPatternMemo()

def img_line_pts(img_shape, line_eq):
    """ Return boundary points of line in image or False if no exist
    Args:
        img_shape - (H, W) tuple
        line_eq   - 3-tuple (a, b, c) such that ax + by + c = 0
    Returns:
        (x1, y1), (x2, y2) - image boundary intersection points
        or False, if the line doesn't intersect the image
    """
    a, b, c = (float(x) for x in line_eq)
    H, W = img_shape
    if a == 0 and b == 0:
        raise ValueError("Invalid line equation: {}".format(line_eq))
    elif a == 0:
        y = -c / b
        if y < 0 or y >= H:
            return False
        else:
            return (0, y), (W, y)

    elif b == 0:
        x = -c / a
        if x < 0 or x >= W:
            return False
        else:
            return (x, 0), (x, H)
    else:
        pts = set([])

        X_y0_intersection = -c / a
        X_yH_intersection = (-c - b*H) / a

        y0_in = X_y0_intersection >= 0 and X_y0_intersection <= W
        yH_in = X_yH_intersection >= 0 and X_yH_intersection <= W
        if y0_in:
            pts.add((X_y0_intersection, 0))
        if yH_in:
            pts.add((X_yH_intersection, H))

        Y_x0_intersection = -c / b
        Y_xW_intersection = (-c - a*W) / b

        x0_in = Y_x0_intersection >= 0 and Y_x0_intersection <= H
        xW_in = Y_xW_intersection >= 0 and Y_xW_intersection <= H
        if x0_in:
            pts.add((0, Y_x0_intersection))
        if xW_in:
            pts.add((W, Y_xW_intersection))

        if len(pts) == 0:
            return False
        elif len(pts) == 1:
            return False
        elif len(pts) == 2:
            return pts.pop(), pts.pop()
        else:
            raise RuntimeError("Found {} intersections! {}".format(len(pts), pts))

def find_closest(xs, x, thr=1):
    diffs = np.abs(xs - x)
    pos = np.argmin(diffs)
    if diffs[pos] > thr:
        pos = None

    return pos

def cv2_colorbar(img, vmin, vmax, cmap=plt.cm.plasma,
                 markers=None):
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    cbar_thickness = 20
    separator_sz = 1
    cbar_length = img.shape[1]
    cbar = np.linspace(vmin, vmax, cbar_length, dtype=np.float32)

    marker_positions = []
    if markers is not None:
        for marker_val, color in markers:
            pos = find_closest(cbar, marker_val)
            if pos is not None:
                marker_positions.append((pos, color))

    cbar = np.tile(cbar, (cbar_thickness, 1))
    cbar = (255 * cmap(norm(cbar))[..., [2, 1, 0]]).astype(np.uint8)  # RGBA to opencv BGR

    for pos, color in marker_positions:
        cbar[:, pos, :] = color

    separator = np.zeros((separator_sz, cbar.shape[1], cbar.shape[2]), dtype=img.dtype)

    # .copy() to ensure contiguous array? otherwise cv2.putText fails.
    vis = np.vstack((img, separator, cbar)).copy()

    text_margin = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.5
    thickness = 1

    text_min = '{:.2f}'.format(vmin)
    text_min_size, text_min_baseline = cv2.getTextSize(text_min, font, size, thickness)
    text_min_bl = (text_margin,
                    img.shape[0] - (text_margin + text_min_baseline + separator_sz))
    cv2.putText(vis, text_min,
                text_min_bl, font,
                size, (255, 255, 255), thickness, cv2.LINE_AA)

    text_max = '{:.2f}'.format(vmax)
    text_max_size, text_max_baseline = cv2.getTextSize(text_max, font, size, thickness)
    text_max_bl = (img.shape[1] - (text_margin + text_max_size[0]),
                    img.shape[0] - (text_margin + text_max_baseline + separator_sz))
    cv2.putText(vis, text_max,
                text_max_bl, font,
                size, (255, 255, 255), thickness, cv2.LINE_AA)

    return vis.copy()

def plt_hatch(mask, ax):
    """ https://stackoverflow.com/a/51345660/1705970 """
    ax.contourf(mask, 1, hatches=['', '//'], alpha=0.)

def cv2_colormap(img, cmap=plt.cm.plasma, vmin=None, vmax=None, do_colorbar=True):
    """ E.g.: vis = colormap(img, plt.cm.viridis) """
    if vmin is None:
        vmin = np.nanmin(img)
    if vmax is None:
        vmax = np.nanmax(img)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    vis = (255 * cmap(norm(img))[..., [2, 1, 0]]).astype(np.uint8)  # RGBA to opencv BGR

    vis[np.isnan(img)] = 0

    if do_colorbar:
        vis = cv2_colorbar(vis, vmin, vmax, cmap)

    return vis.copy()



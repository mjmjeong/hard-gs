import torch
import numpy as np
from utils.graphics_utils import fov2focal
from scipy.spatial.transform import Rotation as R
from scene.utils import Camera
from copy import deepcopy


def rotation_matrix_to_quaternion(rotation_matrix):
    """将旋转矩阵转换为四元数"""
    return R.from_matrix(rotation_matrix).as_quat()

def quaternion_to_rotation_matrix(quat):
    """将四元数转换为旋转矩阵"""
    return R.from_quat(quat).as_matrix()

def quaternion_slerp(q1, q2, t):
    """在两个四元数之间进行球面线性插值（SLERP）"""
    # 计算两个四元数之间的点积
    dot = np.dot(q1, q2)

    # 如果点积为负，取反一个四元数以保证最短路径插值
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # 防止数值误差导致的问题
    dot = np.clip(dot, -1.0, 1.0)

    # 计算插值参数
    theta = np.arccos(dot) * t
    q3 = q2 - q1 * dot
    q3 = q3 / np.linalg.norm(q3)

    # 计算插值结果
    return np.cos(theta) * q1 + np.sin(theta) * q3

def bezier_interpolation(p1, p2, t):
    """在两点之间使用贝塞尔曲线进行插值"""
    return (1 - t) * p1 + t * p2
def linear_interpolation(v1, v2, t):
    """线性插值"""
    return (1 - t) * v1 + t * v2
def smooth_camera_poses(cameras, num_interpolations=5):
    """对一系列相机位姿进行平滑处理，通过在每对位姿之间插入额外的位姿"""
    smoothed_cameras = []
    smoothed_times = []
    total_poses = len(cameras) - 1 + (len(cameras) - 1) * num_interpolations
    time_increment = 10 / total_poses

    for i in range(len(cameras) - 1):
        cam1 = cameras[i]
        cam2 = cameras[i + 1]

        # 将旋转矩阵转换为四元数
        quat1 = rotation_matrix_to_quaternion(cam1.orientation)
        quat2 = rotation_matrix_to_quaternion(cam2.orientation)

        for j in range(num_interpolations + 1):
            t = j / (num_interpolations + 1)

            # 插值方向
            interp_orientation_quat = quaternion_slerp(quat1, quat2, t)
            interp_orientation_matrix = quaternion_to_rotation_matrix(interp_orientation_quat)

            # 插值位置
            interp_position = linear_interpolation(cam1.position, cam2.position, t)

            # 计算插值时间戳
            interp_time = i*10 / (len(cameras) - 1) + time_increment * j

            # 添加新的相机位姿和时间戳
            newcam = deepcopy(cam1)
            newcam.orientation = interp_orientation_matrix
            newcam.position = interp_position
            smoothed_cameras.append(newcam)
            smoothed_times.append(interp_time)

    # 添加最后一个原始位姿和时间戳
    smoothed_cameras.append(cameras[-1])
    smoothed_times.append(1.0)
    print(smoothed_times)
    return smoothed_cameras, smoothed_times

# # 示例：使用两个相机位姿
# cam1 = Camera(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([0, 0, 0]))
# cam2 = Camera(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), np.array([1, 1, 1]))

# # 应用平滑处理
# smoothed_cameras = smooth_camera_poses([cam1, cam2], num_interpolations=5)

# # 打印结果
# for cam in smoothed_cameras:
#     print("Orientation:\n", cam.orientation)
#     print("Position:", cam.position)


trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def rodrigues_mat_to_rot(R):
    eps = 1e-16
    trc = np.trace(R)
    trc2 = (trc - 1.) / 2.
    # sinacostrc2 = np.sqrt(1 - trc2 * trc2)
    s = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if (1 - trc2 * trc2) >= eps:
        tHeta = np.arccos(trc2)
        tHetaf = tHeta / (2 * (np.sin(tHeta)))
    else:
        tHeta = np.real(np.arccos(trc2))
        tHetaf = 0.5 / (1 - tHeta / 6)
    omega = tHetaf * s
    return omega


def rodrigues_rot_to_mat(r):
    wx, wy, wz = r
    theta = np.sqrt(wx * wx + wy * wy + wz * wz)
    a = np.cos(theta)
    b = (1 - np.cos(theta)) / (theta * theta)
    c = np.sin(theta) / theta
    R = np.zeros([3, 3])
    R[0, 0] = a + b * (wx * wx)
    R[0, 1] = b * wx * wy - c * wz
    R[0, 2] = b * wx * wz + c * wy
    R[1, 0] = b * wx * wy + c * wz
    R[1, 1] = a + b * (wy * wy)
    R[1, 2] = b * wy * wz - c * wx
    R[2, 0] = b * wx * wz - c * wy
    R[2, 1] = b * wz * wy + c * wx
    R[2, 2] = a + b * (wz * wz)
    return R


def normalize(x):
    return x / np.linalg.norm(x)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = viewmatrix(vec2, up, center)
    return c2w

def render_path_spiral(c2ws, focal, zrate=.1, rots=3, N=300):
    c2w = poses_avg(c2ws)
    up = normalize(c2ws[:, :3, 1].sum(0))
    tt = c2ws[:,:3,3]
    rads = np.percentile(np.abs(tt), 90, 0)
    rads[:] = rads.max() * .05
    
    render_poses = []
    rads = np.array(list(rads) + [1.])
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        # c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        # z = normalize(c2w[:3, 2])
        render_poses.append(viewmatrix(z, up, c))
    render_poses = np.stack(render_poses, axis=0)
    render_poses = np.concatenate([render_poses, np.zeros_like(render_poses[..., :1, :])], axis=1)
    render_poses[..., 3, 3] = 1
    render_poses = np.array(render_poses, dtype=np.float32)
    return render_poses

def render_wander_path(view):
    focal_length = fov2focal(view.FoVy, view.image_height)
    R = view.R
    R[:, 1] = -R[:, 1]
    R[:, 2] = -R[:, 2]
    T = -view.T.reshape(-1, 1)
    pose = np.concatenate([R, T], -1)

    num_frames = 60
    max_disp = 5000.0  # 64 , 48

    max_trans = max_disp / focal_length  # Maximum camera translation to satisfy max_disp parameter
    output_poses = []

    for i in range(num_frames):
        x_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_frames))
        y_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) / 3.0  # * 3.0 / 4.0
        z_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) / 3.0

        i_pose = np.concatenate([
            np.concatenate(
                [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]], axis=1),
            np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]
        ], axis=0)  # [np.newaxis, :, :]

        i_pose = np.linalg.inv(i_pose)  # torch.tensor(np.linalg.inv(i_pose)).float()

        ref_pose = np.concatenate([pose, np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)

        render_pose = np.dot(ref_pose, i_pose)
        output_poses.append(torch.Tensor(render_pose))

    return output_poses

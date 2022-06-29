import cv2
import h5py
import argparse
import numpy as np
try:
    import cPickle as pkl
except:
    import pickle as pkl
    
import os
import multiprocessing
import sys

from opendr.renderer import ColoredRenderer
from opendr.camera import ProjectPoints
from opendr.geometry import VertNormals
from tex.iso import Isomapper, IsoColoredRenderer

from util import im
from util.logger import log
from models.smpl import Smpl

import matplotlib.pyplot as plt
import glob

def read_obj(filename):
    vt, ft = [], []
    for content in open(filename):
        contents = content.strip().split(' ')
        if contents[0] == 'vt':
            vt.append([float(a) for a in contents[1:]])
        if contents[0] == 'f':
            ft.append([int(a.split('/')[1]) for a in contents[1:] if a])
    return np.array(vt, dtype='float64'), np.array(ft, dtype='int32')

def read_obj_track(filename):
    v, f = [], []
    for content in open(filename):
        contents = content.strip().split(' ')
        if contents[0] == 'v':
            v.append([float(a)*1000 for a in contents[1:]])
        if contents[0] == 'f':
            f.append([int(a) for a in contents[1:]])
    return np.array(v, dtype='float64'), np.array(f, dtype='int32')
    
def read_off(filename):
    with open(filename) as off:
        off.readline()
        n_v, n_f, _ = [int(a) for a in off.readline().strip().split()]
        v, f = [], []

        for i in range(n_v):
            v += [[float(a) for a in off.readline().strip().split()]]

        for i in range(n_f):
            f += [[int(a) for a in off.readline().strip().split()[1:]]]

    return np.array(v, dtype='float64'), np.array(f, dtype='int32')

def read_mask(filename):
    return np.array(cv2.imread(filename, 0) > 0, dtype=np.uint8)

def read_real(filename):
    return cv2.imread(filename, 1)

# Set parameters 
# s, e = int(sys.argv[1]), int(sys.argv[2])
# id = int(sys.argv[1])
# s = id * 100
# e = s + 100
proj_dir='example_data/'
# realimg_dir='/HPS/NerfCharacter/work/NSVFdynamic_dataset3/Lingjie_fullred/'

#if dataset is Oleks, change Line193 to j!=22
split = 'testing'
mesh_smooth = 'smooth3e-2'
resolution = 512




output_tex_dir = proj_dir + split + '/tex_modifysmpluv0.1_' + mesh_smooth +'_testwork'
if not os.path.exists(output_tex_dir):
    os.mkdir(output_tex_dir)
    
out_tex_prefix = output_tex_dir + '/frame_'


output_normal_dir = proj_dir + split + '/normal_modifysmpluv0.1_' + mesh_smooth+'_testwork'
if not os.path.exists(output_normal_dir):
    os.mkdir(output_normal_dir)
out_normal_prefix = output_normal_dir + '/frame_'

model_file = 'smpl_model/cut_alongsmplseams.obj'
pose_folder = proj_dir + split + '/output_' + mesh_smooth + '/smpl'
real_folder = proj_dir + split + '/rgb'



cam_num = len(glob.glob(os.path.join(real_folder, '000000/*.png')))

pose_files = sorted(glob.glob(os.path.join(pose_folder, '*.obj')))
bgcolor = np.array([1., 0.2, 1.])

# Load camera
camera_list=[]

cams, P = {}, {}
for vidx in range(cam_num):
    intrin_camera_file = os.path.join(proj_dir, 'intrinsic', '0_train_{:04d}.txt'.format(vidx))
    extrin_camera_file = os.path.join(proj_dir, 'pose', '0_train_{:04d}.txt'.format(vidx))
  

    intrin = np.loadtxt(intrin_camera_file)
    pose = np.loadtxt(extrin_camera_file)
    pose[:3, 3]*=1000
    RT=np.linalg.inv(pose)
    rot = RT[:3,:3]
    trans = RT[:3, 3]
    R, J = cv2.Rodrigues(rot)
    rt_vec=R[:,0]

    cams[vidx]={}
    cams[vidx]['camera_t']=trans
    cams[vidx]['camera_rt']=rt_vec
    cams[vidx]['camera_f']=np.array([intrin[0, 0], intrin[1, 1]])
    cams[vidx]['camera_c']=np.array([intrin[0, 2], intrin[1, 2]]) 
    cams[vidx]['camera_k']= np.zeros(5)

    sample_img_fname = os.path.join(real_folder, '000000/image_c_000_f_000000.png')
    sample_img = read_real(sample_img_fname)
    cams[vidx]['height']=sample_img.shape[0]
    cams[vidx]['width']=sample_img.shape[1]  

    camera_list.append(cams[vidx])
    

vt, ft = read_obj(model_file)
ft -= 1

def get_tex(i, pose_file):
    best_num=3
    tex_agg = np.zeros((resolution, resolution, best_num, 3))
    tex_agg[:] = np.nan
    normal_agg = np.ones((resolution, resolution, best_num)) * 0.2

    static_indices = np.indices((resolution, resolution))

    v, f = read_obj_track(pose_file)
    f -= 1

    iso = Isomapper(vt, ft, f, resolution, bgcolor=bgcolor)
    iso_vis = IsoColoredRenderer(vt, ft, f, resolution)

    print(vt[0][0], vt[0][1])
    print(int(vt[0][0]*255), int(vt[0][1]*255))

    vn = VertNormals(f=f, v=v)
    normal_tex = iso_vis.render(vn / 2.0 + 0.5)
    cv2.imwrite(out_normal_prefix+str(i).zfill(6)+'.png', normal_tex*255) 
    print('processed {} frames'.format(i))

    for j in range(cam_num):
        real_file = os.path.join(real_folder, '{0:06d}/image_c_{1:03d}_f_{2:06d}.png'.format(i, j, i))
        frame = read_real(real_file)

        indices = np.where(np.all(frame == np.array([255, 255, 255]), axis=-1))
        mask_content = np.ones([frame.shape[0], frame.shape[1]])*255
        mask_content[indices[0], indices[1]] = 0
        mask = np.array(mask_content > 0, dtype=np.uint8)



        camera = ProjectPoints(t=camera_list[j]['camera_t'], rt=camera_list[j]['camera_rt'], c=camera_list[j]['camera_c'], f=camera_list[j]['camera_f'], k=camera_list[j]['camera_k'], v=v)



        frustum = {'near': 100., 'far': 10000., 'width': int(camera_list[j]['width']), 'height': int(camera_list[j]['height'])}

        rn_vis = ColoredRenderer(f=f, frustum=frustum, camera=camera, num_channels=1)

        visibility = rn_vis.visibility_image.ravel()
        visible = np.nonzero(visibility != 4294967295)[0]

        proj = camera.r  # projection
        in_viewport = np.logical_and(
            np.logical_and(np.round(camera.r[:, 0]) >= 0, np.round(camera.r[:, 0]) < frustum['width']),
            np.logical_and(np.round(camera.r[:, 1]) >= 0, np.round(camera.r[:, 1]) < frustum['height']),
        )
        in_mask = np.zeros(camera.shape[0], dtype=np.bool)
        idx = np.round(proj[in_viewport][:, [1, 0]].T).astype(np.int).tolist()
        in_mask[in_viewport] = mask[idx]
        faces_in_mask = np.where(np.min(in_mask[f], axis=1))[0]
        visible_faces = np.intersect1d(faces_in_mask, visibility[visible])
        if(visible_faces.size!=0):
            part_tex = iso.render(frame / 255., camera, visible_faces)
        
            # angle under which the texels have been seen
            points = np.hstack((proj, np.ones((proj.shape[0], 1))))
            points3d = camera.unproject_points(points)
            points3d /= np.linalg.norm(points3d, axis=1).reshape(-1, 1)
            alpha = np.sum(points3d * vn.r, axis=1).reshape(-1, 1)
            alpha[alpha < 0] = 0
            iso_normals = iso_vis.render(alpha)[:, :, 0]
            iso_normals[np.all(part_tex == bgcolor, axis=2)] = 0

            # texels to consider
            part_mask = np.zeros((resolution, resolution))
            min_normal = np.min(normal_agg, axis=2)
            part_mask[iso_normals > min_normal] = 1.


            # update best seen texels
            where = np.argmax(np.atleast_3d(iso_normals) - normal_agg, axis=2)
            idx = np.dstack((static_indices[0], static_indices[1], where))[part_mask == 1]
            tex_agg[list(idx[:, 0]), list(idx[:, 1]), list(idx[:, 2])] = part_tex[part_mask == 1]
            normal_agg[list(idx[:, 0]), list(idx[:, 1]), list(idx[:, 2])] = iso_normals[part_mask == 1]



     # merge textures
    log.info('Computing median texture...')
    tex_median = np.nanmedian(tex_agg, axis=2)

    log.info('Inpainting unseen areas...')
    where = np.max(normal_agg, axis=2) > 0.2

    tex_mask = iso.iso_mask
    mask_final = np.float32(where)

    kernel_size = np.int(resolution * 0.1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    inpaint_area = cv2.dilate(tex_mask, kernel) - mask_final

    tex_final = cv2.inpaint(np.uint8(tex_median * 255), np.uint8(inpaint_area * 255), 3, cv2.INPAINT_TELEA)


    cv2.imwrite(out_tex_prefix+str(i).zfill(6)+'.png', tex_final) 


for i, pose_file in enumerate(pose_files):
    get_tex(i, pose_file)
        

log.info('Done.')
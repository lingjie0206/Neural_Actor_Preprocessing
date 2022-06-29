'''
  @ Date: 2021-01-12 17:08:25
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-01-14 20:49:25
  @ FilePath: /EasyMocap/code/demo_mv1pmf_smpl.py
'''
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# show skeleton and reprojection
import pyrender # first import the pyrender
from pyfitting.optimize_simple import optimizeShape, optimizePose
from dataset.mv1pmf import MV1PMF
from dataset.config import CONFIG
from mytools.reconstruction import simple_recon_person, projectN3
from smplmodel import select_nf, init_params, Config

from tqdm import tqdm
import numpy as np
import torch
import random

def load_model(use_cuda=True):
    # prepare SMPL model
    import torch
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    from smplmodel import SMPLlayer
    body_model = SMPLlayer('data/smplx/smpl2', gender='male', device=device,
        regressor_path='data/smplx/J_regressor_body25.npy')
    body_model.to(device)
    return body_model

def load_weight_shape():
    weight = {'s3d': 1., 'reg_shape': 5e-4}
    return weight

def load_weight_pose():
    weight = {
        'k3d': 1., 'reg_poses_zero': 1e-2, 
        'smooth_Rh': 1e-2, 'smooth_Th': 1e-2, 'smooth_poses': 1e-2
    }
    return weight

def mv1pmf_smpl(path, sub, out, mode, args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    config = CONFIG[mode]
    MIN_CONF_THRES = 0.5
    no_img = False
    dataset = MV1PMF(path, cams=sub, config=CONFIG[mode], add_hand_face=False,
        undis=args.undis, no_img=no_img, out=out)
    # kp3ds = []
    start, end = args.start, min(args.end, len(dataset))
    
    # dataset.no_img = True
    # annots_all = []
    # for nf in tqdm(range(start, end), desc='triangulation'):
    #     images, annots = dataset[nf]
    #     conf = annots['keypoints'][..., -1]
    #     conf[conf < MIN_CONF_THRES] = 0
    #     keypoints3d, _, kpts_repro = simple_recon_person(annots['keypoints'], dataset.Pall, ret_repro=True)
    #     kp3ds.append(keypoints3d)
    #     annots_all.append(annots)
    # smooth the skeleton
    # kp3ds = np.stack(kp3ds)
    # optimize the human shape
    body_model = load_model()
    # params_init = init_params(nFrames=1)
    # weight = load_weight_shape()
    # params_shape = optimizeShape(body_model, params_init, kp3ds, weight_loss=weight, kintree=config['kintree'])
    # optimize 3D pose
    # cfg = Config()
    #params = init_params(nFrames=kp3ds.shape[0])
    params = init_params(nFrames=1)
    # params['shapes'] = params_shape['shapes'].copy()
#     params['shapes'] = np.array([[0.164, -1.811, 0.896, 0.972, -0.048, -0.183, 0.061, -0.087, 0.111, 0.043, -0.058, 0.011, -0.049, -0.121, -0.109, -0.088, 0.096, 0.170, 0.260, 0.001, 0.129, 0.036, 0.124, -0.011, 0.052, -0.008, 0.031, 0.041, -0.049, 0.015, 0.008, 0.125, -0.096, 0.044, 0.091, -0.097, 0.034, 0.032, -0.010, 0.062, 0.053, 0.102, 0.002, -0.013, 0.003, -0.028, 0.021, -0.038, 0.027, 0.015, 0.012, 0.039, 0.035, 0.011, 0.029, -0.048, 0.004, 0.009, -0.029, -0.012, 0.005, -0.009, -0.016, -0.006, -0.002, 0.017, 0.025, -0.008, 0.002, -0.003, 0.000, 0.024, -0.020, 0.011, 0.003, 0.010, -0.009, 0.002, -0.003, -0.014, -0.017, -0.001, -0.015, 0.024, -0.003, 0.012, 0.008, 0.001, 0.005, -0.006, -0.001, 0.014, -0.013, 0.007, 0.007, -0.000, -0.005, -0.007, -0.008, -0.002, -0.011, 0.007, -0.002, -0.012, -0.008, -0.004, 0.011, 0.002, 0.003, 0.008, -0.010, 0.000, 0.002, 0.006, 0.013, 0.000, 0.001, -0.004, -0.001, -0.010, -0.015, -0.002, -0.001, 0.007, -0.001, 0.002, 0.002, -0.004, 0.010, 0.003, 0.002, 0.004, -0.007, 0.004,
#   0.009, -0.005, 0.010, -0.003, 0.004, 0.000, 0.009, 0.003, -0.006, 0.003, -0.010, -0.001, -0.006, 0.008, 0.003, 0.007, 0.005, 0.005, 0.004, 0.007, -0.005, -0.001, -0.000, -0.002, -0.004, -0.002, -0.001, -0.007, -0.003, -0.006, -0.001, -0.002, 0.005, -0.008, -0.005, -0.003, -0.002, -0.002, -0.001, -0.004, -0.002, -0.002, 0.010, 0.002, -0.008, 0.001, 0.001, -0.001, -0.002, 0.000, -0.004, -0.004, -0.001, 0.001, 0.002, 0.002, -0.000, 0.002, 0.003, -0.001, -0.005, 0.002, 0.000, 0.006, 0.000, -0.005, -0.003, 0.001, -0.000, 0.000, 0.001, 0.000, 0.001, -0.008, 0.001, -0.002, 0.000, 0.003, -0.001, 0.003, 0.003, -0.003, -0.001, 0.002, -0.002, -0.002, 0.001, 0.003, 0.000, -0.003, -0.000, -0.005, -0.004, -0.002, -0.002, 0.000, -0.002, -0.002, -0.003, -0.002, 0.004, -0.003, -0.005, 0.000, 0.002, 0.001, 0.002, 0.004, -0.001, -0.002, 0.000, 0.001, -0.003, 0.001, -0.001, -0.004, 0.003, -0.000, -0.001, 0.003, 0.003, -0.003, -0.001, 0.000, 0.000, -0.002, 0.002, 0.003, 0.001, -0.001, -0.002, -0.001,
#   -0.002, 0.002, -0.002, 0.002, -0.001, -0.002, -0.002, 0.001, -0.001, 0.003, 0.003, -0.000, -0.002, -0.001, 0.000, 0.001, -0.002, 0.001, 0.000, 0.001, 0.001, -0.000, 0.002, -0.002, 0.002, -0.000, -0.000, -0.000, 0.001, -0.002, 0.000, -0.000, 0.002, -0.001]])
#     weight = load_weight_pose()
    # cfg.OPT_R = True
    # cfg.OPT_T = True
    # params = optimizePose(body_model, params, kp3ds, weight_loss=weight, kintree=config['kintree'], cfg=cfg)
    # cfg.OPT_POSE = True
    # params = optimizePose(body_model, params, kp3ds, weight_loss=weight, kintree=config['kintree'], cfg=cfg)
#    params['poses']=np.array([[0.000, 0.000, 0.000, -0.045, 0.145, 0.165, -0.082, -0.120, -0.115, 0.000, -0.000, -0.000, -0.105, 0.076, -0.089, -0.007, -0.059, 0.073, 0.000, 0.000, -0.000, 0.179, 0.151, -0.081, 0.114, -0.181, 0.091, -0.000, -0.000, -0.000, -0.006, -0.038, 0.064, -0.055, 0.040, -0.107, -0.262, 0.048, 0.090, 0.000, -0.000, 0.000, -0.000, 0.000, -0.000, 0.066, 0.079, -0.057, -0.076, -0.283, -0.256, -0.047, 0.280, 0.194, -0.012, -0.254, 0.258, -0.050, 0.254, -0.260, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, -0.000, -0.000, -0.000, 0.000]])
    # params['shapes']=np.zeros((1,300))
#    params['Th']=np.array([[-0.088, 1.222, -0.091]])
#    params['Rh']=np.array([[-0.034, 3.117, -0.135]])
    # optimize 2D pose
    # render the mesh

    dataset.no_img = not args.vis_smpl
    for nf in tqdm(range(start, end), desc='render'):
        images, annots = dataset[nf]
        p_dict = args.out + '/smpl_backup/{:06d}.json'.format(nf)
        p_dict = eval(" ".join(open(p_dict).readlines()[1:-1]))[0]
        params.update(p_dict)
        # dataset.write_smpl(select_nf(params, nf-start), nf)
        if args.vis_smpl:
            vertices = body_model(return_verts=True, return_tensor=False, **params)
            # vertices = body_model(return_verts=True, return_tensor=False, **select_nf(params, nf-start))
            # import pdb; pdb.set_trace()
            dataset.save_smpl(vertices=vertices, faces=body_model.faces, images=images, nf=nf, sub_vis=args.sub_vis)
            dataset.vis_smpl(
                vertices=vertices, faces=body_model.faces, images=images, nf=nf, sub_vis=args.sub_vis,
                use_white=True)
            

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('multi_view one_person multi_frame skel')
    parser.add_argument('path', type=str)
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--sub', type=str, nargs='+', default=[],
        help='the sub folder lists when in video mode')
    parser.add_argument('--start', type=int, default=0,
        help='frame start')
    parser.add_argument('--end', type=int, default=10000,
        help='frame end')    
    parser.add_argument('--step', type=int, default=1,
        help='frame step')
    parser.add_argument('--body', type=str, default='body15', choices=['body15', 'body25', 'total'])
    parser.add_argument('--undis', action='store_true')
    parser.add_argument('--add_hand_face', action='store_true')
    parser.add_argument('--vis_smpl', action='store_true')
    parser.add_argument('--sub_vis', type=str, nargs='+', default=[],
        help='the sub folder lists for visualization')
    parser.add_argument('--seed', type=str, default=2)
    args = parser.parse_args()
    mv1pmf_smpl(args.path, args.sub, args.out, args.body, args)
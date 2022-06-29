'''
Running the code:
=============================
>	python hello_smpl_pose.py

'''

from smpl_webuser.serialization import load_model
import numpy as np
import cv2
import glob
import json
import os
import math
np.random.seed(2)
## Load SMPL model (here we load the female model)
## Make sure path is correct

# import pdb;pdb.set_trace()
## Write to an .obj file
def save_obj(m, verts, fname):
    with open(fname, 'w') as fp:
        for v in verts:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

        if m is not None:
            for f in m.f+1: # Faces are 1-based, not 0-based in obj files
                fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

    ## Print message
    print('..Output mesh saved to: ', fname)


def recover(verts, weights, A):
    T = A.dot(weights.T)
    T_inv = np.linalg.inv(T.transpose((2,0,1))).transpose((1,2,0))  # N x 4 x 4
    shape_h = np.vstack((verts.T, np.ones((1, verts.shape[0]))))  # 4 x N

    shape_r = (
        T_inv[:,0,:] * shape_h[0, :].reshape((1, -1)) + 
        T_inv[:,1,:] * shape_h[1, :].reshape((1, -1)) + 
        T_inv[:,2,:] * shape_h[2, :].reshape((1, -1)) + 
        T_inv[:,3,:] * shape_h[3, :].reshape((1, -1))).T
    shape_r = shape_r[:,:3]
    return shape_r


m, A, dd = load_model('models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl')


rodrigues = lambda x : cv2.Rodrigues(x)[0]
tracking_folder='example_data/testing/output_smooth3e-2/smpl'
tracking_files=sorted(glob.glob(os.path.join(tracking_folder, '*.json')))
output_folder = 'example_data/testing/transform_smoth3e-2'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)


motions = []
# from tqdm import tqdm
for i, fname in enumerate(tracking_files):
    # if i <6094 or i> 7110:
    #     continue

    with open(fname) as f:
        lines = "\n".join(f.readlines()[1:-1])[:-1]
        track_r=[eval(lines)]
        m.pose[:] = np.array(track_r[0]['poses'])
        m.betas[:] = np.array(track_r[0]['shapes'])
    
        Rh = np.array(track_r[0]['Rh']) # np.array([-0.034, 3.117, -0.135])
        Th = np.array(track_r[0]['Th']) # np.array([-0.088, 1.222, -0.091])
        
        rot = rodrigues(Rh)
        rot_t = rot.transpose(1,0)
        transl = Th

        vertices = np.matmul(m.r, rot_t) + transl
        # save_obj(m, vertices, os.path.join(output_folder + '/{0:06d}.obj'.format(i)))

        joints = np.matmul(m.J_transformed.r, rot_t) + transl
        

        if i == 0:
            motions.append(joints.tolist())
            motions.append(joints.tolist())
        motions.append(joints.tolist())

        transform = {'translation': transl.tolist(), 
                    'rotation': rot_t.tolist(), 
                    'joints': joints.tolist(),
                    'joints_RT': A.r.tolist(), 
                    "pose": track_r[0]['poses']}
        transform['motion'] = motions[-3:]
        with open(output_folder + '/{}'.format(fname.split('/')[-1]), 'w') as fw:
            json.dump(transform, fw)


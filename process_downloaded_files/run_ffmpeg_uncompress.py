import os
import multiprocessing as mp
import argparse
import time
import sys
import os.path
from util import FileLock


FOLDER='/HPS/HumanBodyRetargeting2/work/Release_data_example/'


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--subject", required=True,
	help="subject")
ap.add_argument("-c", "--cam_num", required=True, type=int, 
	help="camera number")
ap.add_argument("-f", "--frame_num", required=True, type=int, 
	help="frame number")
args = vars(ap.parse_args())

####################### Uncompress texture video ########################
if not os.path.exists(FOLDER+args['subject']+'/tex_modifysmpluv0.1_smooth3e-2'):
    os.makedirs(FOLDER+args['subject']+'/tex_modifysmpluv0.1_smooth3e-2')

command = "ffmpeg -nostdin -i %s -qscale:v 2 -start_number 0 %s" % (FOLDER+args['subject']+'/tex_modifysmpluv0.1_smooth3e-2.avi', FOLDER+args['subject']+'/tex_modifysmpluv0.1_smooth3e-2/frame_%06d.png')
os.system(command)
        
####################### Uncompress normal video ########################
if not os.path.exists(FOLDER+args['subject']+'/normal_modifysmpluv0.1_smooth3e-2'):
    os.makedirs(FOLDER+args['subject']+'/normal_modifysmpluv0.1_smooth3e-2')

command = "ffmpeg -nostdin -i %s -qscale:v 2 -start_number 0 %s" % (FOLDER+args['subject']+'/normal_modifysmpluv0.1_smooth3e-2.avi', FOLDER+args['subject']+'/normal_modifysmpluv0.1_smooth3e-2/frame_%06d.png')
os.system(command)

####################### Uncompress rgb video ########################
camera_list = ['{:03d}'.format(x) for x in range(args['cam_num'])]
frame_list = ['{:06d}'.format(x) for x in range(args['frame_num'])]


if not os.path.exists(FOLDER+args['subject']+'/rgb'):
    os.makedirs(FOLDER+args['subject']+'/rgb')

for frame in frame_list:
	if not os.path.exists(FOLDER+args['subject']+'/rgb/'+frame):
		os.makedirs(FOLDER+args['subject']+'/rgb/'+frame)


for cam in range(args['cam_num']):
	command = "ffmpeg -nostdin -i %s -qscale:v 2 -start_number 0 %s" % (FOLDER+args['subject']+'/rgb_video/'+camera_list[cam]+'.avi',FOLDER+args['subject']+'/rgb/%6d/image_c_'+camera_list[cam]+'.png')
	os.system(command)
print('done')  

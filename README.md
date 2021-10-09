# Recruitment-Task

Download the mmdet ckpt file from https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth 
Download the mmpose ckpt file from https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth

Before running the main file please make sure all four videos are in the same location as the main python file(script.py)

Command to run the code - python top_down_pose_tracking_demo_with_mmdet.py {location of mmdet config file} {location of mmdet ckpt file} 
{location of mmpose config file} {location of mmdet ckpt file}

In the main python file I have set the frame_width and frame_height as 500. This is because if I set it to 1920x1080 I run out of gpu memory.


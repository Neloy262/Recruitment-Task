The script is able to estimate the pose in 4 videos at the same time. The ouput frame shows each video in one of four quadrants. 

Download the mmdet ckpt file from https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth 

Download the mmpose ckpt file from https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth

The mmpose config file needs to know the location of the file in the dataset folder. This can be set on the first line of the mmpose config file.

Before running the main file please make sure all four videos are in the same location as the main python file(script.py)

Command to run the code - python script.py {location of mmdet config file} {location of mmdet ckpt file} 
{location of mmpose config file} {location of mmdet ckpt file}

In the main python file I have set the frame_width and frame_height as 500. This is because if I set it to 1920x1080 I run out of gpu memory.

Complete command that I used to run the file - python script.py mmdet_config/faster_rcnn_r50_fpn_coco.py mmdet_ckpt/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth mmpose_config/res50_coco_256x192.py mmpose_ckpt/res50_coco_256x192-ec54d7f3_20200709.pth


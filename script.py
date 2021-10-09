# Copyright (c) OpenMMLab. All rights reserved.
import os
import warnings
from argparse import ArgumentParser
from itertools import zip_longest
import cv2
import numpy as np
from mmpose.apis import (get_track_id, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_tracking_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

frame_width = 500
frame_height = 500
out = cv2.VideoWriter('output2.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

# Generator function to generate each frame
def gen(filename):
    cap = cv2.VideoCapture(filename)

    # Check if camera opened successfully
    if (cap.isOpened()== False):
        print("Error opening video file")
    while(cap.isOpened()):
	
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            #frame = cv2.resize(frame, (frame_width, frame_height))
            # Display the resulting frame
            # cv2.imshow('Frame', frame)
            
            yield frame
            # Press Q on keyboard to exit
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     cap.release()
            #     cv2.destroyAllWindows()
            #     break

        # Break the loop
        else:
            # yield np.zeros((300,300,3))
            # cap.release()
            # cv2.destroyAllWindows()
            break

def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    # parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--show',
        action='store_true',
        default=True,
        help='whether to show visualizations.')
    # parser.add_argument(
    #     '--out-video-root',
    #     default='',
    #     help='Root of the output video file. '
    #     'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--euro',
        action='store_true',
        help='Using One_Euro_Filter for smoothing')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    # cap = cv2.VideoCapture(args.video_path)
    fps = None

    # assert cap.isOpened(), f'Faild to load video file {args.video_path}'

    # if args.out_video_root == '':
    #     save_out_video = False
    # else:
    #     os.makedirs(args.out_video_root, exist_ok=True)
    #     save_out_video = True # Make True to save ouput file

    # if save_out_video:
    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     videoWriter = cv2.VideoWriter(
    #         os.path.join(args.out_video_root,
    #                      f'vis_{os.path.basename(args.video_path)}'), fourcc,
    #         fps, size)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    next_id = 0
    pose_results = []
    # while (cap.isOpened()):

    # Loop through four frames at a time, merge them and insert into mmpose mmdet pipeline
    for a,b,c,d in zip_longest(gen("1.mp4"),gen("2.mp4"),gen("3.mp4"),gen("4.mp4")):
        pose_results_last = pose_results

        # if any video return None create a new frame with all zeros
        if a is None:
            a = np.zeros([1080,1920,3],dtype=np.uint8)
        if b is  None:
            b = np.zeros([1080,1920,3],dtype=np.uint8)
        if c is  None:
            c = np.zeros([1080,1920,3],dtype=np.uint8)
        if d is  None:
            d = np.zeros([1080,1920,3],dtype=np.uint8)
        
        # print("d:",d)
        # print(type(a))

        # creating new image using four frames
        a = np.hstack((a,b))
        b = np.hstack((c,d))
        img = np.vstack((a,b))
        img = cv2.resize(img, (frame_width, frame_height))
        # flag, img = cap.read()
        # if not flag:
        #     break
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, img)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # get track id for each person instance
        pose_results, next_id = get_track_id(
            pose_results,
            pose_results_last,
            next_id,
            use_oks=args.use_oks_tracking,
            tracking_thr=args.tracking_thr,
            use_one_euro=args.euro,
            fps=fps)

        # show the results
        vis_img = vis_pose_tracking_result(
            pose_model,
            img,
            pose_results,
            radius=args.radius,
            thickness=args.thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            show=False)
        # print(vis_img.shape)
        
        # if args.show:
        cv2.imshow('Image', vis_img)

        # if save_out_video:
        out.write(vis_img)
            

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

        # cap.release()
        # if save_out_video:
        #     videoWriter.release()
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

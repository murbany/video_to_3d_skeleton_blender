from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import shutil

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import json

from tqdm import tqdm
import imageio

import sys
sys.path.append("../lib")
import time

# import _init_paths
import models
from config import cfg
from config import update_config
from core.inference import get_final_preds
from utils.transforms import get_affine_transform

from utils.tools import *
from utils.learning import *
from utils.utils_data import flip_data
from data.dataset_wild import WildDetDataset
from utils.vismo import render_and_save

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_person_detection_boxes(model, img, threshold=0.5):
    pil_image = Image.fromarray(img)  # Load the image
    transform = transforms.Compose([transforms.ToTensor()])  # Defing PyTorch Transform
    transformed_img = transform(pil_image)  # Apply the transform to the image
    pred = model([transformed_img.to(CTX)])  # Pass the image to the model
    # Use the first detected person
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].cpu().detach().numpy())]  # Bounding boxes
    pred_scores = list(pred[0]['scores'].cpu().detach().numpy())

    person_boxes = []
    # Select box has score larger than threshold and is person
    for pred_class, pred_box, pred_score in zip(pred_classes, pred_boxes, pred_scores):
        if (pred_score > threshold) and (pred_class == 'person'):
            person_boxes.append(pred_box)

    return person_boxes


def get_pose_estimation_prediction(pose_model, image, centers, scales, transform):
    rotation = 0

    # pose estimation transformation
    model_inputs = []
    for center, scale in zip(centers, scales):
        trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
        # Crop smaller image of people
        model_input = cv2.warpAffine(
            image,
            trans,
            (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)

        # hwc -> 1chw
        model_input = transform(model_input)#.unsqueeze(0)
        model_inputs.append(model_input)

    # n * 1chw -> nchw
    model_inputs = torch.stack(model_inputs)

    # compute output heatmap
    output = pose_model(model_inputs.to(CTX))
    coords, maxvals = get_final_preds(
        cfg,
        output.cpu().detach().numpy(),
        np.asarray(centers),
        np.asarray(scales))

    return coords, maxvals


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def prepare_output_dirs(video_base_name, prefix='/output/'):
    pose_dir = os.path.join(prefix, video_base_name)
    if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
        shutil.rmtree(pose_dir)
    os.makedirs(pose_dir, exist_ok=True)
    return pose_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--videoFile', type=str, required=True)
    parser.add_argument('--outputDir', type=str, default='/output/')
    parser.add_argument('--inferenceFps', type=int, default=10)
    parser.add_argument('--writeBoxFrames', action='store_true')

    parser.add_argument("--config", type=str, default="../configs/pose3d/MB_ft_h36m_global_lite.yaml", help="Path to the config file.")
    parser.add_argument('-e', '--evaluate', default='../checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('--pixel', action='store_true', help='align with pixle coordinates')
    parser.add_argument('--focus', type=int, default=None, help='target person id')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')
    
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    
    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def main():
    # transformation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)
    pose_dir = prepare_output_dirs(os.path.splitext(os.path.basename(args.videoFile))[0], args.outputDir)
    csv_output_rows = []
    json_output = []

    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_model.to(CTX)
    box_model.eval()
    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    pose_model.to(CTX)
    pose_model.eval()

    # Loading an video
    vidcap = cv2.VideoCapture(args.videoFile)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    if fps < args.inferenceFps:
        print('desired inference fps is '+str(args.inferenceFps)+' but video fps is '+str(fps))
        exit()
    skip_frame_cnt = round(fps / args.inferenceFps)
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    outcap = cv2.VideoWriter('{}/{}_pose.avi'.format(args.outputDir, os.path.splitext(os.path.basename(args.videoFile))[0]),
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (frame_width, frame_height))

    count = 0
    while vidcap.isOpened():
        total_now = time.time()
        ret, image_bgr = vidcap.read()
        count += 1

        if not ret:
            break

        # if count % skip_frame_cnt != 0:
        #     continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Clone 2 image for person detection and pose estimation
        if cfg.DATASET.COLOR_RGB:
            image_per = image_rgb.copy()
            image_pose = image_rgb.copy()
        else:
            image_per = image_bgr.copy()
            image_pose = image_bgr.copy()

        # Clone 1 image for debugging purpose
        image_debug = image_bgr.copy()

        # object detection box
        now = time.time()
        pred_boxes = get_person_detection_boxes(box_model, image_per, threshold=0.9)
        then = time.time()
        print("Find person bbox in: {} sec".format(then - now))

        # Can not find people. Move to next frame
        if not pred_boxes:
            count += 1
            continue

        if args.writeBoxFrames:
            for box in pred_boxes:
                cv2.rectangle(image_debug, tuple(int(x) for x in box[0]), tuple(int(x) for x in box[1]), color=(0, 255, 0),
                              thickness=3)  # Draw Rectangle with the coordinates

        # pose estimation : for multiple people
        centers = []
        scales = []
        for box in pred_boxes:
            center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
            centers.append(center)
            scales.append(scale)

        now = time.time()
        pose_preds, maxvals = get_pose_estimation_prediction(pose_model, image_pose, centers, scales, transform=pose_transform)
        then = time.time()
        print("Find person pose in: {} sec".format(then - now))

        new_csv_row = [count]
        new_json_row = []
        for coords, maxval in zip(pose_preds, maxvals):
            # Draw each point on image
            for i in range(len(coords)):
                x_coord, y_coord = int(coords[i][0]), int(coords[i][1])
                cv2.circle(image_debug, (x_coord, y_coord), 4, (255, 0, 0), 2)
                new_csv_row.extend([x_coord, y_coord])
                new_json_row.extend([x_coord, y_coord, float(maxval[i][0])])

        total_then = time.time()

        text = "{:03.2f} sec".format(total_then - total_now)
        cv2.putText(image_debug, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("pos", image_debug)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        csv_output_rows.append(new_csv_row)
        img_file = os.path.join(pose_dir, 'pose_{:08d}.jpg'.format(count))
        json_output.append({
            "image_id": img_file,
            "category_id": 1,
            "keypoints": new_json_row,
            "score":  3
        })
        cv2.imwrite(img_file, image_debug)
        outcap.write(image_debug)


    # write csv
    csv_headers = ['frame']
    for keypoint in COCO_KEYPOINT_INDEXES.values():
        csv_headers.extend([keypoint+'_x', keypoint+'_y'])

    csv_output_filename = os.path.join(args.outputDir, '{}_pose-data.csv'.format(os.path.splitext(os.path.basename(args.videoFile))[0]))
    with open(csv_output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csv_headers)
        csvwriter.writerows(csv_output_rows)

    json_output_filename = os.path.join(args.outputDir, '{}_pose-data.json'.format(os.path.splitext(os.path.basename(args.videoFile))[0]))
    with open(json_output_filename, 'w') as jsonfile:
        json.dump(json_output, jsonfile)
    vidcap.release()
    outcap.release()

    cv2.destroyAllWindows()

    # 2D to 3D pose-lifting
    config_args = get_config(args.config)
    model_backbone = load_backbone(config_args)
    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    print('Loading checkpoint', args.evaluate)
    checkpoint = torch.load(args.evaluate, map_location=lambda storage, loc: storage)
    model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
    model_pos = model_backbone
    model_pos.eval()
    testloader_params = {
            'batch_size': 1,
            'shuffle': False,
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': True,
            'drop_last': False
    }

    vid = imageio.get_reader(args.videoFile,  'ffmpeg')
    fps_in = vid.get_meta_data()['fps']
    vid_size = vid.get_meta_data()['size']
    os.makedirs(args.outputDir, exist_ok=True)

    if args.pixel:
        # Keep relative scale with pixel coornidates
        wild_dataset = WildDetDataset(json_output_filename, clip_len=args.clip_len, vid_size=vid_size, scale_range=None, focus=args.focus)
    else:
        # Scale to [-1,1]
        wild_dataset = WildDetDataset(json_output_filename, clip_len=args.clip_len, scale_range=[1,1], focus=args.focus)

    test_loader = DataLoader(wild_dataset, **testloader_params)

    results_all = []
    with torch.no_grad():
        for batch_input in tqdm(test_loader):
            N, T = batch_input.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            if config_args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if config_args.flip:    
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip) # Flip back
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
            else:
                predicted_3d_pos = model_pos(batch_input)
            if config_args.rootrel:
                predicted_3d_pos[:,:,0,:]=0                    # [N,T,17,3]
            else:
                predicted_3d_pos[:,0,0,2]=0
                pass
            if config_args.gt_2d:
                predicted_3d_pos[...,:2] = batch_input[...,:2]
            results_all.append(predicted_3d_pos.cpu().numpy())

    results_all = np.hstack(results_all)
    results_all = np.concatenate(results_all)
    render_and_save(results_all, '{}/{}_X3D.mp4'.format(args.outputDir, os.path.splitext(os.path.basename(args.videoFile))[0]), keep_imgs=False, fps=fps_in)
    if args.pixel:
        # Convert to pixel coordinates
        results_all = results_all * (min(vid_size) / 2.0)
        results_all[:,:,:2] = results_all[:,:,:2] + np.array(vid_size) / 2.0
    np.save('{}/{}_X3D.npy'.format(args.outputDir, os.path.splitext(os.path.basename(args.videoFile))[0]), results_all)


if __name__ == '__main__':
    main()

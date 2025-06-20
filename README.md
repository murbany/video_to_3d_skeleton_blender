# Video to 3D Keypoint Pipeline for Generating Animated Skeletons in Blender

## Introduction
This is a machine learning pipeline for extracting 3D pose representations from single-person video and ingesting them into 3D design software (in this case Blender). The pipeline takes advantage of two major models: HRNet from [Deep High-Resolution Representation Learning for Human Pose Estimation](https://arxiv.org/pdf/1902.09212.pdf) and MotionBERT from [MotionBERT: A Unified Perspective on Learning Human Motion Representations](https://arxiv.org/pdf/2210.06551.pdf). HRNet is used to extract 2D pose estimation following the COCO dataset keypopints format. Advnatages of this model are that it maintains a high-resolution representation through the parallel multi-resolution subnetworks that exchange information via repeated multi-scale fusion trhough training and inference. HRNet produces a set of 17 keypoints in screen space which while useful for direct frame-to-frame comparisons has significatn limitations for potential use in 3D applications. To that end, the 2D single-person pose estimations from HRNet are then lifted to 3D pose estimations using MotionBERT too extract 17 body keypoints from the [H36M format](https://github.com/JimmySuen/integral-human-pose/blob/master/pytorch_projects/common_pytorch/dataset/hm36.py#L32). MotionBERT makes use of a Dual-stream Spatio-temporal transformer composed of multi-head self attention blocks to that cpasure intra-frame and inter-frames body- joint interactions. Producing a numpy array of 3d keypoints, the resulting file can than be used in 3D animation software to generate a basic animated skeleton which can then be bound to a mesh. This is reflected in an add-on for Blender that takes as input the pose data file and builds the joints matching the location while also adjusting for bone rotation by keeping them aligned with the next join in the chain (i.e. right hip points towards right knee which points towards right ankle)

## 2D Pose Estimation
![](/demo/2d_keypoint.gif)

## 2D to 3D Pose Lifting
![](/demo/3d_keypoint.gif)

## Blender Import and Animation
![](/demo//blender_skel.gif)

## Limitations
This works specifically for single-person pose estimation.
The skeleton created in blender does not have any of the control handles commonly found in production enviroment rigs. It also does not account for bone roll. Ideally a rig with controls in a t-pose that matches the 17 keypoint skeleton would be made to which then the pose and rotations would be applied. This skeleton also does not include any inverse or forwards kinematics for either. Initial impressions demonstrate that the skeleton can be used to drive a mesh although the lack of a starting t-pose that fits the mesh produces severe deformation of the mesh. While missing functionality from a true production standpoint the workling result is fertile ground for additional improvements and refinements.

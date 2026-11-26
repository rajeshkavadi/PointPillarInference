Objective
- Build a CPU-only PointPillars inference pipeline that reuses Livox ROS1 bag playback/PointCloud2/RViz infrastructure, replacing the repository's CenterPoint-inspired detector with the user's OpenPCDet-trained checkpoint.
- First milestone: load checkpoint on CPU, extract a LiDAR frame from the bag, run inference, generate valid 3D boxes; then ROS/RViz integration.
Requirements
- CPU ONLY: no CUDA, TensorRT, Jetson, .cuda(), CUDA events, CUDA voxelization, or CUDA NMS.
- Inspect upstream repo first; reuse ROS subscriber, PointCloud2 conversion, and MarkerArray/RViz visualization patterns; do not reuse CenterPoint model/preprocessing/detection logic.
- Do not change model architecture, checkpoint weights, voxel size, ranges, anchors, or detection parameters; do not invent missing config; validate compatibility.
- Verify actual bag topic/format; handle missing intensity explicitly without silently fabricating features; preserve LiDAR frame and timestamps.
- Deliver repo analysis, install instructions, checkpoint report, inference code, ROS node, RViz config, execution commands, troubleshooting, benchmarks.
Decisions
- Use upstream master branch as reference because it contains tools/test_ros.py, tools/vis_ros.py, and tools/rviz.rviz; topics reused: /pointcloud, /detect_box3d, /text_det, frame livox_frame.
- Use exact OpenPCDet KITTI PointPillars config because checkpoint was proven bit-identical to official pointpillar_7728.pth.
- Implement CPU equivalents: dense Conv2d BEV backbone, deterministic numpy voxelizer, pure torch rotated 3D IoU NMS.
- Bag has no intensity (x,y,z,rgb); default INTENSITY_MODE=zero fills 4th VFE feature with 0.0 and prints a loud warning; do not treat rgb as intensity.
- No C++ extension/OpenPCDet install needed: use standalone torch/numpy package.
Work State
Completed
- Inspected workspace: demo1.bag, demo2.bag, pointpillar_7728_ft.pth; verified no repo/config present initially.
- Installed CPU PyTorch torch==1.10.2+cpu; installed gdown, pycryptodomex, rospkg, python-gnupg so Python3 rospy/rosbag imports work under ROS Melodic.
- Cloned and inspected /tmp/livox_detection_inspect (master + v1.0/v1.1) and /tmp/OpenPCDet_inspect (HEAD 233f849).
- Verified bags: only /livox/lidar, sensor_msgs/PointCloud2, frame livox_frame, fields [x,y,z,rgb], no intensity; demo1 101 messages, demo2 141 messages.
- Loaded checkpoint on CPU: epoch=80, it=9280, model_state with 127 tensors; shapes match OpenPCDet PointPillars.
- Downloaded /tmp/pointpillar_7728_official.pth using gdown ID 1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm; all 127 tensors identical to user's checkpoint.
- Created /home/rajesh/live_inference_pp/livox_detection/ structure and copied checkpoint/reference ROS files/data symlinks.
- Wrote configs, CPU model, voxelizer, postprocess, pipeline, ROS utils, bag frame reader, and tools/pointpillars_cpu_inference.py.
Active
- Remaining deliverables not yet written: tools/pointpillars_ros_cpu.py, tools/extract_bag_frame.py, launch/pointpillars_cpu.launch, rviz/pointpillars_cpu.rviz, README.md, requirements.txt.
- Written code has not yet been executed/tested; strict state_dict load, IoU/NMS correctness, and single-frame inference still need validation.
- Minor cleanup needed in tools/pointpillars_cpu_inference.py: no-op if args.save_frame and frame["index"] == args.frame and frame["topic"].endswith(".bag") is False: pass.
Blocked
- No hard blocker; known limitation: checkpoint is KITTI forward-view with intensity, bags are Livox 360° without intensity, so detection quality may degrade even though pipeline can run.
- Open3D not installed; --open3d is optional and currently prints install hint if missing.
Next Move
1. Run Test 1: python3 tools/pointpillars_cpu_inference.py --validate-only; fix any strict-load/shape errors.
2. Validate rotated 3D IoU/NMS numerically with simple overlap cases.
3. Run Test 2: python3 tools/pointpillars_cpu_inference.py --bag ../data/demo1.bag --frame 0 --save-frame ../data/demo1_frame0.npy, inspect detections/timings.
4. Finish ROS node, bag frame extraction tool, launch file, RViz config, README/requirements.
5. Run ROS integration (roscore, node, rosbag play, RViz) and benchmark several frames; record results.
Relevant Files
- /home/rajesh/live_inference_pp/livox_detection/configs/pointpillars.yaml: validated OpenPCDet KITTI model config plus inference plumbing.
- /home/rajesh/live_inference_pp/livox_detection/pointpillars/model.py: standalone CPU PointPillars with OpenPCDet-compatible parameter names.
- /home/rajesh/live_inference_pp/livox_detection/pointpillars/pipeline.py: checkpoint loading, preprocessing, timing, detect().
- /home/rajesh/live_inference_pp/livox_detection/pointpillars/config.py: config loader/validation and CPU enforcement.
- /home/rajesh/live_inference_pp/livox_detection/pointpillars/postprocess.py: CPU 3D IoU and class-agnostic NMS.
- /home/rajesh/live_inference_pp/livox_detection/pointpillars/voxelizer.py: CPU pillar generation.
- /home/rajesh/live_inference_pp/livox_detection/pointpillars/ros_utils.py: PointCloud2 parsing and RViz MarkerArray builders.
- /home/rajesh/live_inference_pp/livox_detection/pointpillars/bag_frame.py: Python3 rosbag frame reader.
- /home/rajesh/live_inference_pp/livox_detection/tools/pointpillars_cpu_inference.py: validation/single-frame/benchmark CLI.
- /home/rajesh/live_inference_pp/livox_detection/checkpoints/pointpillars_checkpoint.pth: user checkpoint copy.
- /home/rajesh/live_inference_pp/demo1.bag, /home/rajesh/live_inference_pp/demo2.bag: verified Livox bags.
- /tmp/OpenPCDet_inspect: reference OpenPCDet source used for exact architecture/config.
- /tmp/livox_detection_inspect: upstream Livox ROS repo source used for ROS/RViz patterns.
- /tmp/pointpillar_7728_official.pth: official checkpoint used for bit-identical validation.
- /home/rajesh/procedural_dem_creatoin/live_inference_pp/pointpillars_jetson: prior CPU implementation studied for reusable patterns, different checkpoint.
Important Context
- Checkpoint validation result: pointpillar_7728_ft.pth == official OpenPCDet KITTI pointpillar_7728.pth; 127/127 tensors identical, epoch=80, it=9280.
- Exact model config from OpenPCDet KITTI: CLASS_NAMES: ['Car','Pedestrian','Cyclist'], POINT_CLOUD_RANGE: [0,-39.68,-3,69.12,39.68,1], VOXEL_SIZE: [0.16,0.16,4], MAX_POINTS_PER_VOXEL: 32, test MAX_NUMBER_OF_VOXELS: 40000, anchor rotations [0,1.57], feature_map_stride: 2, DIR_OFFSET: 0.78539, NUM_DIR_BINS: 2, SCORE_THRESH: 0.1, NMS_THRESH: 0.01, NMS_PRE_MAXSIZE: 4096, NMS_POST_MAXSIZE: 500, class-agnostic NMS_TYPE: nms_gpu.
- Bag topic was verified, not assumed: /livox/lidar; PointCloud2 fields [x, y, z, rgb], no intensity; frame_id livox_frame; point_step 32; bag data uncompressed.
- Python3 ROS Melodic fixes: pip3 install pycryptodomex rospkg python-gnupg; rosbag emits harmless LZ4 warning.
- Upstream CUDA-specific code intentionally not reused: torch.cuda.Event, .cuda(), torch.cuda.synchronize(), ros_numpy, CenterPoint LD_base.
- No git repo in /home/rajesh/live_inference_pp; all project changes are uncommitted files.

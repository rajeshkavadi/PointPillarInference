# CPU-only PointPillars inference for Livox ROS1 bags

This project runs the **user's OpenPCDet PointPillars checkpoint** on **CPU only**,
using the Livox ROS1 bag / PointCloud2 / RViz infrastructure from the upstream
`livox_detection` repository. The CenterPoint-inspired model and its CUDA
preprocessing are **not** reused.

First milestone (checkpoint load → bag frame → valid 3D boxes) is working, and
both viewers have been exercised end-to-end: the ROS1/RViz path and the default
**local three.js viewer** (`tools/pointpillars_web_view.py`), which needs no
ROS at all, shows the cloud colourised by distance with 5 m radar rings, loops
the bag until Ctrl+C and can hand out a public Cloudflare link for remote
viewers.

---

## 0. Quick start — launch the viewer

All commands are run from this package directory (except the `roslaunch` ones
must come from a shell with your catkin workspace sourced). Step 0 and the
optional Cloudflare share link need the network; everything else runs offline.

```bash
# 0) dependencies (CPU wheels), once per machine
python3 -m pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu

# 1) three.js viewer from a bag — no roscore, no RViz
python3 tools/pointpillars_web_view.py --bag /path/to/demo1.bag
#    -> open http://127.0.0.1:8080
#    -> share https://<random>.trycloudflare.com (public link, bag loops until Ctrl+C)

# 2) same, but skip the neural net (instant cloud only, for a quick look)
python3 tools/pointpillars_web_view.py --bag data/demo1.bag --no-infer

# 3) ROS mode: mirror a running detector node
rosrun livox_detection pointpillars_ros_cpu.py      # terminal A (or your Livox driver)
rosrun livox_detection pointpillars_web_view.py --ros   # terminal B -> same URL

# 4) everything at once through roslaunch
source /opt/ros/melodic/setup.bash
source ~/ws_livox/devel/setup.bash
roslaunch livox_detection pointpillars_cpu.launch          # web viewer on :8080
roslaunch livox_detection pointpillars_cpu.launch launch_rviz:=true   # + RViz
roslaunch livox_detection pointpillars_cpu.launch launch_web:=false   # RViz only
roslaunch livox_detection pointpillars_cpu.launch web_port:=8081 bag:=/path/to/demo2.bag

# optional: set the defaults once instead of passing flags every time
export LIVOX_BAG=/path/to/demo1.bag                # used when --bag is omitted
export LIVOX_DETECTION_ROOT=/path/to/livox_detection   # ckpt / config / webapp root
```

Terminal output prints the exact URLs (`open: http://127.0.0.1:8080` and, when
`cloudflared` is installed, `share: https://<random>.trycloudflare.com`). Full
flag reference and what the page shows:
[section 6](#6-local-threejs-viewer-default-replaces-rviz).

---

## 1. Repository analysis (what was reused)

Reference repositories inspected:

- `livox_detection` upstream `master` — contains `tools/test_ros.py`,
  `tools/vis_ros.py`, `tools/rviz.rviz`.
- OpenPCDet at commit `233f849` — source of the exact KITTI PointPillars config
  and CPU-equivalent model implementation.

Reused from upstream ROS infrastructure:

| Item | Upstream source | Here |
|---|---|---|
| LiDAR subscription | `/livox/lidar`, `sensor_msgs/PointCloud2` | same topic |
| RViz point cloud | `/pointcloud` | same topic |
| 3D boxes | `/detect_box3d`, `MarkerArray` | same topic |
| Text labels | `/text_det`, `MarkerArray` | same topic |
| Fixed frame | `livox_frame` | preserved from message header |
| Wireframe/arrow/label markers | `tools/vis_ros.py` pattern | same style/color map |

Deliberately **not** reused:

- `livoxdetection.models.ld_base_v1.LD_base` (different model)
- `.cuda()`, `torch.cuda.Event`, `torch.cuda.synchronize()`
- `spconv`, CUDA voxelization, `iou3d_nms_cuda` compiled extension
- `ros_numpy` (replaced with an explicit PointCloud2 parser)
- Hard-coded KITTI/CenterPoint sensor calibration (`point_cloud_range`,
  `offset_ground`, etc.)

CPU replacements implemented here:

- deterministic NumPy pillar/voxel generation,
- dense `Conv2d`/`ConvTranspose2d` BEV backbone (mathematically equivalent to
  sparse convolution on this single dense BEV plane),
- pure-torch rotated 3D IoU and greedy class-agnostic NMS,
- explicit PointCloud2 field parsing with missing-intensity detection.

---

## 2. Checkpoint report

See [`docs/CHECKPOINT_REPORT.md`](docs/CHECKPOINT_REPORT.md). Short version:

- `checkpoints/pointpillars_checkpoint.pth` is **byte-identical** to the official
  OpenPCDet KITTI `pointpillar_7728.pth`:
  `sha256 = 4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2`.
- `epoch=80`, `it=9280`, 127 tensors in `model_state`.
- `strict=True` load succeeds with zero missing/unexpected keys.
- Model parameters: 4,834,888. Classes: `Car`, `Pedestrian`, `Cyclist`.
- Original KITTI config values are used unchanged:
  range `[0,-39.68,-3, 69.12,39.68,1]`, voxel `[0.16,0.16,4]`,
  anchors `[0, 1.57]`, score threshold `0.1`, NMS threshold `0.01`.

No checkpoint weights, voxel sizes, ranges, anchors, or detection parameters
were changed.

---

## 3. Verified bag facts

Verified with `rosbag` Python API, not assumed:

| Bag | Topic | Type | Messages | Duration | Frame | Fields |
|---|---|---|---|---|---|---|
| `demo1.bag` | `/livox/lidar` | `sensor_msgs/PointCloud2` | 101 | 9.99 s | `livox_frame` | `x, y, z, rgb` |
| `demo2.bag` | `/livox/lidar` | `sensor_msgs/PointCloud2` | 141 | — | `livox_frame` | `x, y, z, rgb` |

`point_step = 32`, data uncompressed. **There is no intensity field.** The
`rgb` field is *not* treated as intensity. Default `INTENSITY_MODE=zero` fills
the VFE's 4th feature with `0.0` and prints a loud warning.

Note: the bag's record time and the message `header.stamp` differ. The ROS node
publishes markers with the **incoming `header.stamp`** (the sensor timestamp),
not `rospy.Time.now()` and not the bag record time.

---

## 4. Install

Tested on Ubuntu 18.04 / ROS Melodic / Python 3.6.9. Nothing below depends on a
particular user or home directory: every script resolves its own package root,
and paths are only ever given explicitly (`--bag`, `--ckpt`, ...).

```bash
cd <this package>            # e.g. ~/live_inference_pp/livox_detection
python3 -m pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu
```

Installed versions used here:

```
torch==1.10.2+cpu
numpy==1.19.5
PyYAML==3.12
rospkg==1.6.3
pycryptodomex==3.21.0
python-gnupg==0.5.7
```

`pycryptodomex`, `python-gnupg` and `rospkg` let ROS Melodic's Python-2-era
`rosbag` import under Python 3. The harmless LZ4 warning can be ignored for
these uncompressed bags.

For `roslaunch`/`rosrun`, add this directory as a catkin package (the path is
whatever your workspace is):

```bash
mkdir -p ~/ws_livox/src
ln -sfn "$PWD" ~/ws_livox/src/livox_detection
cd ~/ws_livox
catkin_make
source devel/setup.bash
```

If you do not want to modify the workspace, set:

```bash
export ROS_PACKAGE_PATH="$(dirname "$PWD"):$ROS_PACKAGE_PATH"
```

---

## 5. CPU / offline tests

All commands below are run from the package directory.

### Test 1 — checkpoint validation

```bash
python3 tools/pointpillars_cpu_inference.py --validate-only
```

Expected: `CHECKPOINT VALIDATION PASSED`, `strict load OK (127 tensors)`.

### Correctness test — rotated 3D IoU / NMS

The CPU IoU was checked against an independent scalar polygon-clipping
reference on 100 random rotated box pairs (max error `1.0e-7`) and on exact
cases:

- identical boxes → IoU `1.0`
- disjoint boxes → IoU `0.0`
- half-overlap boxes → IoU `0.3333`
- vertically disjoint boxes → IoU `0.0`

Implementation note: OpenPCDet's `boxes_overlap_bev_gpu` returns the BEV
**intersection area**, not BEV IoU. `boxes_overlap_bev()` in
`pointpillars/postprocess.py` matches that, and `boxes_iou3d()` applies the
height overlap and 3D union exactly as OpenPCDet's `iou3d_nms_utils.py`.

### Test 2 — single frame from a bag

```bash
python3 tools/pointpillars_cpu_inference.py \
  --bag data/demo1.bag --frame 0 \
  --save-frame data/demo1_frame0.npy
```

`data/demo1.bag` and `data/demo2.bag` are relative symlinks to the bags one
directory above the package. If you move the package without the bags, pass
`--bag /path/to/file.bag` (or set `LIVOX_BAG`) instead.

Observed on frame 0:

- 167,616 original points → 90,308 inside range → 17,538 pillars
- 31 valid 3D boxes; top detection:
  `Car score=0.9546 x=36.35 y=6.94 z=-0.74 dx=4.20 dy=1.70 dz=1.56 yaw=3.086`
- timing: preprocess 78.8 ms, network 4206.6 ms, decode 95.0 ms,
  NMS 647.9 ms, **total 5028.3 ms (0.20 FPS)**

The saved `.npy` is `N x 4` (`x, y, z, intensity-or-zero`); a `.json` sidecar
stores topic/frame/stamp/fields.

Standalone frame extraction:

```bash
python3 tools/extract_bag_frame.py data/demo1.bag --frame 0 \
  --out data/demo1_frame0.npy
python3 tools/extract_bag_frame.py data/demo1.bag --list-topics
```

### Optional — Open3D visualization (no ROS)

Open3D is not installed here. If installed:

```bash
python3 tools/pointpillars_cpu_inference.py \
  --points data/demo1_frame0.npy --open3d
```

### Benchmark

```bash
python3 tools/pointpillars_cpu_inference.py \
  --bag data/demo1.bag --frame 0 --frames 3 --print-every 0
```

Measured on this 4-core machine (steady-state over 2 frames after warm-up):

| Metric | Value |
|---|---|
| Mean | 4649.8 ms (0.22 FPS) |
| Min | 3955.9 ms |
| Max | 5343.6 ms |
| Network share | ~3.5–4.3 s |
| NMS share | ~0.65–0.95 s |

This is a CPU-only implementation; it is **not real-time** at 10 Hz LiDAR rate.

---

## 6. Local three.js viewer (default; replaces RViz)

The browser page is the primary viewer. It needs **no ROS, no rosbridge, no
RViz and no extra Python packages**: the WebSocket/HTTP server is plain
`http.server` + `socket`, and three.js/OrbitControls are vendored in
`webapp/static/vendor/`, so it also works fully offline. Sharing over the
internet is optional and uses an external `cloudflared` binary. A browser
window is opened automatically when a desktop session is available (`--no-open`
disables that); on a headless machine just open the printed URL.

### Bag mode — one command, no roscore

```bash
cd <this package>
python3 tools/pointpillars_web_view.py --bag /path/to/demo1.bag
# open  http://127.0.0.1:8080
# share https://<random>.trycloudflare.com
```

The bag replays in a loop until Ctrl+C (disable with `--no-loop`). Unless
`--no-share` is passed, the viewer also starts a **Cloudflare quick tunnel** in
the background (frames start immediately, the local URL never waits for it) and
prints a public `https://<random>.trycloudflare.com` link that anyone on another
network can open — no Cloudflare account needed. Sharing needs `cloudflared` on
`PATH` (or `CLOUDFLARED_BIN`); if it is missing the viewer says so and keeps
running on localhost only:

```bash
mkdir -p ~/.local/bin && curl -fL -o ~/.local/bin/cloudflared \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
  && chmod +x ~/.local/bin/cloudflared
```

The page shows:

- the cloud **colourised by distance** (blue near → red far); the toolbar also
  offers Height (Z) and Intensity colouring,
- **radar range rings every 5 m out to 70 m**, each labelled in metres, plus
  cardinal spokes, the sensor origin and a live range legend,
- 3D wireframe boxes with `class score · distance` labels (toggleable),
- TOP / 3D / FRONT camera presets, orbit/zoom/pan, and a point budget selector,
- a **persistent map**: every incoming scan is added to what is already on
  screen instead of replacing it, so the road you have driven stays visible,
- **Follow sensor** keeps the LiDAR in the middle while it moves, and the
  **RECENTER** button snaps the view (and the camera target) back onto it.

### Map mode — build the map while you drive

Detection always runs on the **current frame only**. The persistent map is a
viewer feature, not part of the model: `pointpillars/map_builder.py` projects
each frame onto a coarse top-down occupancy grid and cross-correlates it with
the previous frame (FFT) to estimate how far the sensor moved. That pose places
the scan in a world frame, and only the points that are not already in the map
(`0.4 m` voxels) are streamed to the browser, which simply appends them.

- In map mode the cloud, the boxes and the range rings are drawn in map
  coordinates and the sensor marker moves with the estimated pose.
- A bag loop restarts the map (the vehicle jumps back to the first frame); in
  `--ros` mode with a live sensor the map keeps growing.
- `--no-loop` now keeps the finished map on screen until Ctrl+C instead of
exiting immediately.
- The estimate is translation-only (no yaw) and scan-based, so on a sharp turn
  or a featureless corridor it can drift; use `--no-map` for the old
  single-frame view.
- `data/demo1.bag` drives straight ahead at 1.5–2 m per frame, which is what
  the map is built for; the demo loop produces a corridor about 170 m long.

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--bag` | `data/demo1.bag` | bag to replay (or `LIVOX_BAG`) |
| `--no-infer` | off | show the cloud only, no network (fast, bag-rate replay) |
| `--start-frame` / `--max-frames` | 0 / 0 | frame window (0 = all, forever) |
| `--no-loop` | loop on | play the bag once instead of repeating |
| `--play-rate` | 1.0 | pacing of `--no-infer` replay |
| `--score-thresh` / `--nms-thresh` | config | detection thresholds |
| `--host` / `--port` | `127.0.0.1` / `8080` | bind address; localhost by default |
| `--no-share` | share on | skip the Cloudflare quick tunnel |
| `--cloudflared` | on `PATH` | explicit path to the `cloudflared` binary |
| `--no-map` | map on | disable the persistent map (single-frame view) |
| `--map-voxel` | `0.4` | map resolution in metres (one point per cube) |
| `--map-max-points` | `1500000` | upper bound on accumulated map points |
| `--no-open` | open on | do not try to open a browser window |

`--no-infer` is handy for checking the viewer itself in seconds; full
inference still costs the usual ~4–6 s per frame on CPU.

### ROS mode — mirror the detector node

```bash
# terminal 1: detector (bag or live Livox driver)
rosrun livox_detection pointpillars_ros_cpu.py
# terminal 2: web viewer
rosrun livox_detection pointpillars_web_view.py --ros
```

The node publishes `/pointcloud` (points) and now also `/detect_json` (compact
boxes + timings) for the viewer. `--points-topic` / `--json-topic` override the
names. RViz remains fully supported and can run alongside it.

The launch file starts the web viewer by default (which also opens a Cloudflare
share link, since sharing is on by default); RViz is opt-in. The launch file
has no flag for that yet — run the viewer command directly with `--no-share` if
you want a local-only session:

```bash
roslaunch livox_detection pointpillars_cpu.launch                       # web viewer
roslaunch livox_detection pointpillars_cpu.launch launch_rviz:=true     # + RViz
roslaunch livox_detection pointpillars_cpu.launch launch_web:=false     # RViz only
roslaunch livox_detection pointpillars_cpu.launch web_port:=8081
```

### Portability (run it on another machine)

- No absolute paths are baked into code: the package root is derived from the
  script location (or `rospkg`, or `LIVOX_DETECTION_ROOT`), and the bag is
  resolved from `--bag` → `LIVOX_BAG` → `data/demo1.bag` → `../demo1.bag` →
  `./demo1.bag`.
- The viewer adds no dependencies to `requirements.txt`; the only Node/npm-free
  assets (three.js r128, OrbitControls) ship inside `webapp/`.
- It binds to `127.0.0.1` only; nothing is exposed to the network unless you
  explicitly pass `--host 0.0.0.0`. The optional Cloudflare quick tunnel exposes
  the viewer through an outbound connection only; the link dies with the process.

---

## 7. ROS + RViz integration

### One-command launch

```bash
source /opt/ros/melodic/setup.bash
source ~/ws_livox/devel/setup.bash   # or set ROS_PACKAGE_PATH
roslaunch livox_detection pointpillars_cpu.launch
```

Default launch behavior:

- starts `pointpillars_ros_cpu.py`
- plays `data/demo1.bag` at `--rate 0.1` (10× slower than real time, because
  one CPU pass takes seconds)
- starts the three.js viewer on `http://127.0.0.1:8080` (`launch_web:=true`)
- does **not** start RViz; pass `launch_rviz:=true` for that

Useful launch arguments:

```bash
roslaunch livox_detection pointpillars_cpu.launch \
  bag:=/path/to/demo2.bag play_rate:=0.05 web_port:=8081
roslaunch livox_detection pointpillars_cpu.launch launch_web:=false launch_rviz:=true
```

To run without bag playback (e.g. with a real Livox driver):

```bash
roslaunch livox_detection pointpillars_cpu.launch play_bag:=false
```

### Manual nodes

```bash
roscore
rosrun livox_detection pointpillars_ros_cpu.py
rosbag play --clock --rate 0.1 data/demo1.bag
rviz -d rviz/pointpillars_cpu.rviz
```

### Topics and node behavior

| Direction | Topic | Type |
|---|---|---|
| in | `/livox/lidar` | `sensor_msgs/PointCloud2` |
| out | `/pointcloud` | `sensor_msgs/PointCloud2` (`x,y,z,intensity`; intensity is 0 here) |
| out | `/detect_box3d` | `visualization_msgs/MarkerArray` (wireframes + heading arrows) |
| out | `/text_det` | `visualization_msgs/MarkerArray` (class + score + pose text) |
| out | `/detect_json` | `std_msgs/String` (compact boxes + timings for the web viewer) |

Each MarkerArray includes its own `DELETEALL` marker, so RViz does not retain
stale objects. Marker colors follow upstream: Car cyan, Pedestrian yellow,
Cyclist green.

The subscriber keeps only the **newest** `PointCloud2` (`queue_size=1`) and the
main loop processes frames as fast as possible, explicitly counting skipped
stale frames. This avoids 4–5 s blocking callbacks. The observed first frame:

```
[INFO] frame 1 | stamp=1606273185.612 frame=livox_frame | points=179472 boxes=27 | infer=8383 ms wall=11524 ms | skipped=0
```

`infer` is the network time; `wall` includes parsing, decode, NMS and marker
publishing, and is inflated by cold-start caching on the first frame.

### Node parameters / CLI flags

| ROS param | CLI flag | Default | Meaning |
|---|---|---|---|
| `~cfg_file` | `--cfg_file` | bundled config | OpenPCDet-style YAML |
| `~ckpt` | `--ckpt` | bundled checkpoint | `.pth` file |
| `~lidar_topic` | `--lidar_topic` | `/livox/lidar` | input topic |
| `~points_topic` | `--points_topic` | `/pointcloud` | republished cloud |
| `~box_topic` | `--box_topic` | `/detect_box3d` | box markers |
| `~text_topic` | `--text_topic` | `/text_det` | label markers |
| `~json_topic` | `--json_topic` | `/detect_json` | web viewer detections |
| `~frame_id` | `--frame_id` | `""` | `""` preserves incoming frame_id |
| `~intensity_mode` | `--intensity-mode` | `zero` | `zero`/`constant`/`field` |
| `~z_offset` | `--z-offset` | `0.0` | known calibration only |
| `~queue_size` | `--queue-size` | `1` | subscriber queue |
| `~log_every` | `--log-every` | `10` | log every N processed frames |

`--score-thresh`, `--nms-thresh` and `--no-republish-points` are also available
as CLI flags. Score/NMS defaults come from the original config and should not
be changed for a fidelity check.

---

## 8. Configuration and missing-intensity policy

Model/sensor parameters live in `configs/pointpillars.yaml`; they are the exact
OpenPCDet KITTI values plus an `INFERENCE` section for plumbing only.

Intensity handling:

- `zero` (default): VFE 4th feature = `0.0`, loud warning. Use for these bags.
- `constant`: VFE 4th feature = `INTENSITY_VALUE`.
- `field`: use the real intensity field; raises an error if the cloud has none.
  This is the only faithful mode for the KITTI-trained checkpoint.

`Z_OFFSET` and `Z_ROTATION_DEG` are left at `0.0`. Change them only with a known
calibration; they are sensor plumbing, not model tuning.

---

## 9. File map

```
livox_detection/
├── CMakeLists.txt                 # catkin package
├── package.xml
├── requirements.txt
├── README.md
├── configs/
│   ├── pointpillars.yaml          # exact OpenPCDet KITTI model config
│   └── dataset.yaml
├── checkpoints/
│   └── pointpillars_checkpoint.pth
├── data/
│   ├── demo1.bag -> ../../demo1.bag
│   └── demo2.bag -> ../../demo2.bag
├── docs/
│   └── CHECKPOINT_REPORT.md
├── launch/
│   └── pointpillars_cpu.launch
├── rviz/
│   └── pointpillars_cpu.rviz
├── pointpillars/
│   ├── config.py                  # config loader + CPU enforcement
│   ├── model.py                   # standalone PointPillars, identical state keys
│   ├── voxelizer.py               # deterministic CPU pillar generation
│   ├── postprocess.py             # rotated 3D IoU + class-agnostic NMS
│   ├── pipeline.py                # checkpoint load + detect()
│   ├── ros_utils.py               # PointCloud2 parse + MarkerArray builders
│   ├── bag_frame.py               # Python3 rosbag frame reader
│   ├── map_builder.py             # persistent map + scan-to-scan motion estimate
│   ├── web_server.py              # stdlib WebSocket/HTTP server + frame codec
│   ├── tunnel.py                  # Cloudflare quick tunnel for remote viewers
│   └── __init__.py
├── webapp/                        # three.js viewer (served by web_server.py)
│   ├── index.html
│   └── static/
│       ├── css/style.css
│       ├── js/viewer.js           # map layer, follow/recenter, rings, boxes
│       └── vendor/                # three.js r128 + OrbitControls (offline)
├── tools/
│   ├── pointpillars_cpu_inference.py   # validate / single frame / benchmark
│   ├── pointpillars_ros_cpu.py         # ROS1 node
│   ├── pointpillars_web_view.py        # bag or ROS -> browser viewer
│   └── extract_bag_frame.py            # bag frame -> .npy + .json
└── reference/                     # untouched upstream ROS files for comparison
```

---

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Failed to load Python extension for LZ4 support` | Harmless; these bags are uncompressed. Ignore or install rosbag LZ4 support. |
| `INTENSITY WARNING ... 4th VFE feature with 0.0` | Expected for these bags. Use an intensity-bearing bag and `INTENSITY_MODE=field` for faithful inference. |
| `INTENSITY_MODE=field but ... no intensity field` | Correct refusal; do not fabricate intensity. Use `zero` or a real intensity source. |
| `CHECKPOINT ERROR: checkpoint/architecture mismatch` | Wrong checkpoint. Run the validation command; only the stock KITTI `pointpillar` checkpoint is supported. |
| `Cannot locate node of type [pointpillars_ros_cpu.py]` | Script not executable or package not found. `chmod +x tools/*.py`; source catkin devel or set `ROS_PACKAGE_PATH`. |
| RViz shows nothing | Confirm Fixed Frame is `livox_frame`; check `rostopic echo /detect_box3d`; make sure `play_bag:=true` or an external driver publishes `/livox/lidar`. |
| Web page is blank / "Waiting for the first LiDAR frame" | The page loaded but no frame arrived yet. In bag mode the first CPU frame takes ~5–10 s (the startup banner says so); in `--ros` mode check `rostopic hz /pointcloud` and `rostopic echo -n1 /detect_json`. |
| Page never loads / HTTP 000 | Different problem: the process already exited (check the terminal) or another program owns the port (`--port 8081`). The printed `open:` URL is the one to use. |
| The map smears or jumps | The pose comes from the scans alone (translation only), so sharp turns and long featureless stretches drift. Pass `--no-map` for the single-frame view or expect a coarser map. |
| `cannot listen on 127.0.0.1:8080` | Another viewer is already running; use `--port 8081` (or `web_port:=8081`). |
| `share unavailable: cloudflared is not installed` | Install `cloudflared` (command in [section 6](#6-local-threejs-viewer-default-replaces-rviz)) or pass `--no-share`. The viewer itself still works on localhost. |
| `share unavailable: no Cloudflare URL after 30s` | Outbound HTTPS to Cloudflare is blocked or the network is down. Test with `cloudflared tunnel --url http://127.0.0.1:8080` in another terminal, then use `--no-share` or fix egress. |
| Remote share link shows Cloudflare error 1033 / 502 | The viewer process exited (tunnel links die with it) or the tunnel was restarted, which issues a new random URL. Keep the command running and share the latest printed link. |
| Web viewer has no boxes in `--ros` mode | The node must be this package's `pointpillars_ros_cpu.py` (it publishes `/detect_json`); `rostopic echo -n1 /detect_json` should print a JSON line. |
| RViz exits with `libGL error: MESA-LOADER: failed to open i965/swrast` | Your shell is exporting Snap/terminal-specific graphics paths. Launch with `env -u LIBGL_DRIVERS_PATH -u GDK_PIXBUF_MODULEDIR -u GDK_PIXBUF_MODULE_FILE LIBGL_ALWAYS_SOFTWARE=1 roslaunch ...`. |
| Very few frames processed / large `skipped` counter | Expected: CPU inference is ~0.2 FPS. Lower `play_rate` (e.g. `0.05`) or use a faster machine. |
| Import error for `torch` in ROS node | Use Python 3 with the CPU wheel: `python3 -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu`. |

---

## 11. Known limitations

- The checkpoint is a **KITTI forward-view** model
  (`x ∈ [0, 69.12]`, `y ∈ [-39.68, 39.68]`) trained **with intensity**. These
  bags are 360° Livox scans **without intensity**. The pipeline runs and yields
  geometrically valid boxes, but accuracy is degraded and false positives are
  expected (31 boxes on frame 0 at score threshold 0.1).
- CPU inference is ~0.2 FPS here; this is a functional/validation pipeline,
  not a real-time or safety-critical detector.
- The persistent map's pose is estimated from the scans (BEV cross-correlation,
  translation only, no IMU/odometry). It is good enough to keep a driving scene
  together for a while, but it is not SLAM: sharp yaw changes and featureless
  areas make it drift, and no loop closure is performed. Detection is never run
  on the accumulated map — only on the current frame.
- No CUDA, TensorRT, Jetson or compiled extensions are used or supported by
  this CPU implementation.

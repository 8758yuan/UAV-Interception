# YOLO weights

Place detection weights here. The default launch loads `balloon_yolo26n.pt`
from this directory after `colcon build --symlink-install`. The node accepts
any detection weight file through the ROS `model_path` parameter.

Use Ultralytics 8.4 or later. Download the official YOLO26n base weights:

```bash
curl -fL -o src/ibvs_perception/models/yolo26n.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
```

`balloon_yolo26n.pt` is the YOLO26n model fine-tuned for the red spherical
Gazebo target and the `balloon` class at 1280-pixel input size. The COCO base
`yolo26n.pt` does not contain the `balloon` class and cannot replace these
fine-tuned weights directly. The previous `balloon_yolo11n.pt` is retained for
comparison and rollback.

Training uses the existing Matterport balloon photos and labeled Gazebo
renders: 596 training images and 133 validation images. A separate set of 40
positive and 20 empty-scene Gazebo renders is reserved for evaluation.

To reproduce training from the workspace root in the ROS-compatible virtual
environment (the dataset is stored separately from this repository):

```bash
source /home/yuan/.venvs/ibvs-yolo/bin/activate
python scripts/train_balloon_yolo26n.py \
  --data /home/yuan/balloon_training/dataset/data.yaml
# For the first run; later runs receive a numbered output directory.
cp results/perception/balloon_yolo26n/weights/best.pt \
  src/ibvs_perception/models/balloon_yolo26n.pt
colcon build --symlink-install --packages-select ibvs_perception ibvs_sim
```

Training retains the previous recipe: up to 80 epochs, batch size 4, and seed
2026. Model files are ignored by Git and must be copied or trained locally
before building. The repository's default launch uses confidence 0.25 and
input size 1280.

Offline comparison on 2026-10-20 used the same 60 held-out Gazebo images,
confidence 0.25, IoU threshold 0.45, input size 1280, batch size 1, and an RTX
4060 Laptop GPU with Ultralytics 8.4.163. Both models detected all 40 positive
images (all selected boxes had IoU >= 0.5) and produced no detections on the
20 empty scenes.

| Model | Mean center error | Mean selected-box IoU | Mean predict time |
| --- | --- | --- | --- |
| balloon_yolo11n | 0.130 px | 0.953 | 7.24 ms |
| balloon_yolo26n | 0.119 px | 0.955 | 6.81 ms |

Timing covers the Python prediction call after warmup, excluding image file
decoding and ROS transport. Full per-image results are saved locally in
`results/perception/yolo26n_comparison.json`. This small offline comparison
does not establish flight interception performance or closed-loop stability.

To use other weights, set `model_path:=/path/to/best.pt` and set
`target_class` to an exact class name contained in those weights.

The model is loaded from a local file at startup. The node does not silently
download weights or accept a class name absent from the model.

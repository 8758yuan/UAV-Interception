# YOLO weights

Place detection weights here. The default launch loads `balloon_yolo11n.pt`
from this directory after `colcon build --symlink-install`. The node accepts
any detection weight file through the ROS `model_path` parameter.

Download the official lightweight weight before building:

```bash
curl -fL -o src/ibvs_perception/models/yolo11n.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
```

`balloon_yolo11n.pt` is fine-tuned for the red spherical Gazebo target and the
`balloon` class. It was trained at 1280-pixel inference size. The original COCO
`yolo11n.pt` is retained as the fine-tuning base; its `sports ball` class does
not reliably detect the Gazebo target.

Training used Matterport's balloon sample set plus labeled Gazebo renders of
the project's red sphere. The weights were evaluated on a separate set of 40
positive and 20 empty-scene Gazebo renders.

To use other weights, set `model_path:=/path/to/best.pt` and set
`target_class` to an exact class name contained in those weights.

The model is loaded from a local file at startup. The node does not silently
download weights or accept a class name absent from the model.

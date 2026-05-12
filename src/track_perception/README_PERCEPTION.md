# Track Perception

ROS 2 perception package for the racing car. It runs object detection and semantic segmentation from the shared-memory camera stream.

## Nodes

### `object_detection_node`

Inputs:
- shared memory video stream

Outputs:
- `/detection/results`: `Float32MultiArray`, layout `[class_id, confidence, x1, y1, x2, y2, cx, cy] * N`
- `/detection/labels`: comma-separated class labels

### `perception_decision_node`

Inputs:
- shared memory video stream
- `/detection/results`, used only for GuideBoard branch selection and visualization

Outputs:
- `/segmentation/center_offset`: `Float32`, normalized track offset
- `/segmentation/is_valid`: `Bool`

Current decision logic:
- segment road mask into scan bands
- detect branches from separated road segments
- lock to the configured branch side during an intersection
- fit a centerline and publish the smoothed offset
- optionally use GuideBoard detection to override branch side

Removed legacy logic:
- far/near ROI width intersection detection
- branch mask / masked offset calculation
- mask image publishing

## Launch

```bash
ros2 launch track_perception perception.launch.py
```

The launch file loads:

```text
config/intersection_params.yaml
```

You can override the config path:

```bash
ros2 launch track_perception perception.launch.py config_file:=/path/to/config.yaml
```

## Build

```bash
colcon build --packages-select track_perception
source install/setup.bash
```

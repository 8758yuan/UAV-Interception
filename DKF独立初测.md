
## 首次构建

```bash
source /opt/ros/humble/setup.bash
source /home/yuan/px4_ros2_ws/install/setup.bash
MicroXRCEAgent udp4 -p 8888
```



```bash
DKF_TEST_CASE=C

gnome-terminal --window --title="2：PX4 与 Gazebo" -- bash -c '
cd /home/yuan/PX4-Autopilot
export PX4_GZ_MODEL_POSE="0,0,0,0,0,0"
export PX4_GZ_GUI_CONFIG=/home/yuan/ibvs_ws/src/ibvs_sim/config/camera_3d_and_first_person.config
make px4_sitl gz_x500_mono_cam
exec bash
'

gnome-terminal --window --title="3：DKF $DKF_TEST_CASE 目标与检测" -- bash -c '
source /opt/ros/humble/setup.bash
source /home/yuan/px4_ros2_ws/install/setup.bash
source /home/yuan/.venvs/ibvs-yolo/bin/activate
source /home/yuan/ibvs_ws/install/setup.bash
if [ "$1" = "C" ]; then
  ros2 launch ibvs_sim p7_moving_target.launch.py \
    target_x:=30.0 target_y:=0.0 target_z:=12.0 \
    target_pattern:=circle target_tilt_deg:=45.0 target_speed_m_s:=2.0 \
    figure8_radius_x_m:=6.0 figure8_radius_y_m:=6.0 device:=0
elif [ "$1" = "D" ]; then
  ros2 launch ibvs_sim p4_vision_monitor.launch.py \
    target_x:=30.0 target_y:=0.0 target_z:=10.0 device:=0
else
  ros2 launch ibvs_sim p4_vision_monitor.launch.py \
    target_x:=30.0 target_y:=0.0 target_z:=12.0 device:=0
fi
exec bash
' bash "$DKF_TEST_CASE"

gnome-terminal --window --title="4：DKF $DKF_TEST_CASE 独立测试" -- bash -c '
source /opt/ros/humble/setup.bash
source /home/yuan/px4_ros2_ws/install/setup.bash
source /home/yuan/ibvs_ws/install/setup.bash
ros2 launch ibvs_control dkf_standalone_test.launch.py \
  test_case:="$1" trial_id:="dkf_${1}_$(date +%Y%m%d_%H%M%S)"
exec bash
' bash "$DKF_TEST_CASE"
```


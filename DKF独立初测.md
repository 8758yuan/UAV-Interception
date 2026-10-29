# DKF 独立初测命令

A/B/C 气球初始位置为 Gazebo ENU `(30, 0, 12) m`，D 为 `(30, 0, 10) m`；
无人机起飞至 10 m。

工况 A：双方静止，记录 20 s。
工况 B：静止目标；无人机以 10 m 高度为中心，ENU 前后 ±4 m、左右 ±6 m、
上下 ±2 m，周期分别为 20、15、18 s，记录 60 s。
工况 C：无人机沿用 B 的三轴正弦路线，以约 `(0, 0, 10) m` 为中心运动；
目标沿当前拦截试验的 6 m 半径、45° 倾斜圆运动，速度 2 m/s，记录 60 s。
工况 D：目标静止在 `(30, 0, 10) m`；无人机从约 `(0, 0, 10) m` 出发，
沿与 C 中气球相同的位移轨迹运动，半径 6 m、倾斜 45°、速度 2 m/s，记录 60 s。
以测试开始为 `t=0`，令 `θ=t/3`，无人机 ENU 位移为
`(6(cosθ−1)/√2, 6sinθ, 6(cosθ−1)/√2) m`；周期约 18.85 s，
高度约 1.515～10 m。路线从圆周边缘起步，机头仍固定朝 +X。

所有工况机头朝 Gazebo +X（PX4 NED yaw=π/2）。位置轨迹由 PX4 执行，
DKF 输出只用于评估。每次测试单独启动一套仿真；不要同时运行拦截控制、
其他 Offboard 节点或另一套观测器。每次运行合并命令前，结束上次四个终端中的
进程，包括 DDS Agent，避免沿用上次飞行位置、残留目标或重复占用端口。

## 首次构建

```bash
cd /home/yuan/ibvs_ws
source /opt/ros/humble/setup.bash
source /home/yuan/px4_ros2_ws/install/setup.bash
colcon build --packages-select interception_interfaces ibvs_control --symlink-install
```

## 合并启动：一次打开四个终端

需要桌面环境和 `gnome-terminal`。只修改第一行的 `DKF_TEST_CASE`：
`A` 为静止基线，`B` 为无人机运动，`C` 为无人机与目标同时运动。
`D` 为无人机沿上述倾斜圆运动、目标静止；运行 D 时将下方第一行改为 `DKF_TEST_CASE=D`。
完整复制下面一个命令块执行，无需另外启动 DDS Agent、目标或测试节点。

```bash
DKF_TEST_CASE=C

gnome-terminal --window --title="1：DDS Agent" -- bash -c '
source /opt/ros/humble/setup.bash
source /home/yuan/px4_ros2_ws/install/setup.bash
MicroXRCEAgent udp4 -p 8888
exec bash
'

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

该命令会解锁、起飞到 10 m，使用现有单帧已知尺寸初始化，开始记录和规定运动，
到时自动降落。落地后测试进程自动退出并生成报告，终端窗口保留。
测试期间 DKF 不接收 Gazebo 真值；独立入口覆盖拦截执行循环，仅发送位置指令，
复用原有起飞/降落检查。无视觉尺寸或匹配姿态时会保持悬停并打印初始化等待原因。

计时从发送初始化/reset 开始，包含初始化瞬态；目标 C 始终沿原轨迹运动，
其非零初速与现有零目标速度先验的差异也属于本次测试。
记录器通过专用 `/interception/dkf_test/status` 识别测试开始和结束，
不会向 `/interception/control/debug` 伪造拦截状态。

## 结果：继续使用现有每 0.5 秒对比表

```text
/home/yuan/ibvs_ws/results/p4/dkf/<trial_id>/每0.5秒对比.xlsx
/home/yuan/ibvs_ws/results/p4/dkf/<trial_id>/测试结果.html
```

沿用原表的 18 个状态、真实值/估计值/误差、0 秒/每 0.5 秒/结束行及误差汇总。
起飞和降落不计入表格。原始全频率数据仍写入同目录的 `states.csv`，
指标保存在 `metrics.csv`；新测试使用独立 trial 目录，不覆盖或拼接旧试验。
可通过 `output_directory:=...` 指定结果根目录。

先各跑一次确认数据链路，再各重复三次。比较位置和速度误差、漂移、滞后及不同圈次的变化。
速度真值仍是 Gazebo 实测位姿差分近似，IMU 偏置无可靠真值，表中为不可用。
测试输入保留现有 70 ms 图像延迟和 DKF 50 Hz、D=4 配置；
该测试评价规定运动下的观测性能，不代表拦截闭环性能或稳定性验证。

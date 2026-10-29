# DKF 状态观测器交付包

本包从现有 IBVS 工程当前工作区提取 18 维延迟卡尔曼观测器。算法核心、ROS 节点、坐标转换、启动文件及消息定义均原样复制；未修改控制律、观测模型、初始化条件或延迟回放行为。

交付副本仅精简构建文件及依赖，`design_parameters.yaml` 只保留 `paper_state_observer` 参数段。源码来源、原文件 SHA-256 和提取方式见 `SOURCE_MANIFEST.json`，交付文件校验见 `SHA256SUMS`，本次验证结果见 `VALIDATION.md`。

## 文件与边界

```text
src/ibvs_control/
  ibvs_control/paper_state_observer.py       # NumPy 算法核心
  ibvs_control/paper_state_observer_node.py  # ROS 2 接口
  ibvs_control/frames.py                    # PX4/ROS 坐标转换
  config/paper_state_observer.yaml          # 话题、外参及初始化条件
  config/design_parameters.yaml            # 当前观测器参数
  launch/paper_state_observer.launch.py
  test/                                    # 算法测试、ROS 消息接口测试
src/interception_interfaces/                # 四种自定义消息及构建文件
examples/minimal_dkf.py                     # 无 ROS 的合成调用示例
```

不包含飞行控制器、目标检测器、粗估距生成模块、PX4 固件、Gazebo、训练权重、实验日志或录包。接收方提供输入数据；本包不产生推力、角速度或飞行指令。

保留原 ROS 包名 `ibvs_control`、`interception_interfaces`。请在独立工作区使用，避免与完整工程中同名包混用。

## 纯 Python 使用

Python >= 3.10，核心运行仅依赖 NumPy。以下命令均在解压后的包根目录运行：

```bash
python3 -m pip install -r requirements.txt
PYTHONPATH="$PWD/src/ibvs_control" python3 examples/minimal_dkf.py

# 可选：运行现有算法测试；无需 ROS。
python3 -m pip install -r requirements-test.txt
PYTHONPATH="$PWD/src/ibvs_control" python3 -m pytest -q src/ibvs_control/test/test_paper_state_observer.py
```

算法入口是 `PaperStateObserver(config, noise)`，不会自动读取 YAML。初始化、配置与噪声需由调用者传入；示例显式设置当前主要参数，并提供已知的合成初始状态。

| 调用 | 输入/用途 | 返回 |
|---|---|---|
| `initialize(state, relative_covariance=..., image_position_covariance=...)` | 完整 18 维初值；可选 6×6 相对状态协方差及 2×3 图像/位置互协方差 | `ObserverSnapshot` |
| `initialize_from_image(q, image_xy, camera_depth_m, ...)` | 另一种已有初始化接口，使用姿态、图像及已知正光轴深度；深度不是斜距 | `ObserverSnapshot` |
| `predict(gyro_rad_s_b, accel_m_s2_b, dt_s)` | FLU 三轴角速度、FLU 加速度计比力、正时间步长秒；单步不超过 `maximum_dt_s` | `ObserverSnapshot` |
| `complete_dkf_cycle()` | 每完成一个 DKF 周期调用一次，保存该周期的 IMU 子步 | 无 |
| `correct_delayed_image(image_xy)` | 延迟图像的两个归一化坐标；D>0 时需先有 D 个完整历史周期 | `ObserverSnapshot` |
| `correct_image(image_xy)` | D=0 的当前图像校正接口 | `ObserverSnapshot` |
| `snapshot()` / `reset()` | 读取当前输出 / 丢弃当前估计和历史 | 快照 / 无 |

调用顺序：初始化 → 逐样本预测 → 完成一个 DKF 周期 → 有有效延迟图像且 `delay_history_ready` 为真时校正 → 读取输出。预测可以在视觉缺失时继续；`complete_dkf_cycle()` 按 DKF 周期调用，不是每个 IMU 样本调用。

核心默认值与 ROS 配置不同：核心默认 `D=0`、相机位置 `[0,0,0]`，随包 ROS 配置为 `D=4`、相机位置 `[0.14,0,0]` m。不要依靠核心默认值复现项目配置。噪声参数为标准差，代码内部平方后构造协方差。

## ROS 2 构建和启动

参考环境为 ROS 2 Humble / Python 3.10。依赖 `rclpy`、`launch`、`launch_ros`、NumPy、`std_msgs`、`geometry_msgs`、`builtin_interfaces`、ROS 消息生成工具和与 PX4 配套的 `px4_msgs`。自定义消息已随包提供。

`px4_msgs` 是外部依赖，未打包。本机参考仓库为 `https://github.com/PX4/px4_msgs.git`，提交为 `8fb8277cf95c4ba87094146bc8922e76c4ac0799`；两个使用的消息定义哈希见来源清单。接收方应使用与其 PX4 固件匹配的版本，不能只因版本更新就替换消息定义。

```bash
source /opt/ros/humble/setup.bash
# 按接收方实际安装路径加载已构建的 px4_msgs 工作区。
source /path/to/px4_ros2_ws/install/local_setup.bash

# 在本包根目录构建两个独立包。
colcon build --base-paths src --packages-select interception_interfaces ibvs_control
source install/local_setup.bash
ros2 launch ibvs_control paper_state_observer.launch.py
```

默认 `use_sim_time: true`，外部必须发布 `/clock`。实机或无仿真时钟运行时，先在交付副本的 `paper_state_observer.yaml` 中设置为 `false`。当前 launch 不声明同名命令行参数。

节点默认等待 reset、初始化姿态、有效 bootstrap 和有效图像，不会仅凭 IMU 自动初始化。独立接入时由外部发送 reset，再提供有效图像及时间匹配的 bootstrap；不能只启动本节点就期待状态输出。

```bash
ros2 topic pub --once /interception/observer/reset std_msgs/msg/Empty '{}'
ros2 topic echo /interception/observer/state
```

## ROS 输入

以下话题名来自当前配置。IMU、姿态和视觉订阅为 best-effort / volatile、队列深度 5；reset 与 bootstrap 使用默认可靠传输、队列深度 10。

| 话题 | 类型 | 使用的内容 |
|---|---|---|
| `/fmu/out/sensor_combined` | `px4_msgs/msg/SensorCombined` | `timestamp`（微秒，决定预测步长）、`gyro_rad[3]`（FRD，rad/s）、`accelerometer_m_s2[3]`（FRD 比力，m/s²） |
| `/fmu/out/vehicle_attitude` | `px4_msgs/msg/VehicleAttitude` | `q[4]`，Hamilton `[w,x,y,z]`，FRD→NED；仅用于初始化姿态，不持续作为量测校正 |
| `/interception/vision/raw_feature` | `interception_interfaces/msg/VisionFeature` | `x_norm`、`y_norm`、`valid`、`reason`；初始化时还使用 `capture_stamp` |
| `/interception/observer/bootstrap` | `interception_interfaces/msg/ObserverBootstrap` | `stamp`、`p_r`、`v_r`、`relative_covariance[36]`、正的 `range_m` 和 `range_std_m`；`sample_count` 不参与算法 |
| `/interception/observer/reset` | `std_msgs/msg/Empty` | 丢弃估计并允许下一次初始化 |
| `/clock`（仅仿真时间） | `rosgraph_msgs/msg/Clock` | 驱动 ROS 时钟与 DKF 定时器，需由外部提供 |

图像归一化定义为 `x_norm=(u-cx)/fx`、`y_norm=(v-cy)/fy`，相机光学轴为右、下、前；坐标不是像素，也不是除以图像宽高。`u/v`、`width_norm`、`area_px`、`sequence`、`publish_stamp` 在当前 DKF 节点中不参与状态校正。bootstrap 的已知尺寸估距在上游完成；DKF 本身不会利用框宽重新计算距离。

加速度输入是加速度计比力，算法内部经旋转后加上重力，不能传入已去重力的世界系线加速度。水平静止时，PX4 FRD 输入约为 `[0,0,-9.80665]`，转换后核心 FLU 输入约为 `[0,0,+9.80665]`。

bootstrap 的 `p_r = p_vehicle - p_target`、`v_r = v_vehicle - v_target` 均在 ENU，位置基于机体原点；协方差为 `[p_r(3),v_r(3)]` 顺序的 6×6 矩阵，按行展开。`range_m` 为米、`range_std_m` 为米；它们参与初始化数据合法性检查，实际状态及协方差来自对应字段。

当前默认 `require_interception_reset=true`、`require_range_bootstrap=true`、`bootstrap_max_age_s=0.5`。初始化图像采集时间须不早于 reset；初始化时 bootstrap 年龄须在 0～0.5 s 内，且与图像采集时间相差不超过 0.5 s。bootstrap 和图像时间戳应与 ROS 时钟处于同一时间域。节点按两者时间差对初始位置及相对协方差作已有的速度传播。

## 输出与状态约定

主输出 `/interception/observer/state`：`interception_interfaces/msg/ObserverState`。初始化成功后通常随 50 Hz DKF 定时器发布；初始化状态或等待信息还可能随视觉回调发布。输出默认可靠传输、队列深度 10。

18 维排列严格为：`[q(4), p_r(3), v_r(3), image_xy(2), b_gyr(3), b_acc(3)]`。

| 字段 | 维数 | 含义/单位 |
|---|---:|---|
| `q` | 4 | FLU 机体系→ENU，Hamilton `[w,x,y,z]` |
| `p_r` | 3 | 无人机位置减目标位置，机体原点，ENU，m |
| `v_r` | 3 | 无人机速度减目标速度，ENU，m/s |
| `image_xy` | 2 | 归一化图像坐标，无量纲 |
| `b_gyr` | 3 | FLU 陀螺仪偏置，rad/s |
| `b_acc` | 3 | FLU 加速度计偏置，m/s² |
| `covariance` | 324 | 上述顺序的 18×18 协方差，按行展开；单位随对应状态分量 |
| `prediction_count` / `correction_count` | 各 1 | 预测子步/图像校正计数 |
| `innovation_norm` | 1 | 最近一次图像校正的创新二范数，无量纲 |
| `initialized` / `valid` / `reason` | — | 初始化、有效标志和状态说明 |
| `stamp` | — | ROS 发布时间，不是图像采集时刻或严格的 IMU 状态时间 |

没有有效图像时，节点继续 IMU 预测，输出仍可标为 `valid=true`；该标志不代表视觉持续可见或估计精度达标。未初始化时的默认数值字段不能当作有效估计。

评估输出 `/interception/observer/evaluation`：`ObserverEvaluation`，包含主状态副本和最近处理的 `imu_timestamp_us`；仅在存在订阅者时发布。时间戳是收到的 PX4 IMU 时间，单位微秒，不保证与 ROS 发布时间具有相同历元。

纯 Python 输出 `ObserverSnapshot` 包含 `state`、18×18 NumPy `covariance`、两个计数及 `innovation_norm`，不包含 ROS 消息的时间戳、`valid`、`reason` 等字段。

## 延迟模型、参数及验证边界

当前 ROS 配置：DKF 50 Hz、`D=4`，对应 80 ms 的固定周期延迟补偿；IMU 200 Hz 是示例采样率，节点按实际 IMU 时间差传播并在每个 DKF 周期内合并子步。每周期最多消费一条最新图像，历史未填满时继续预测并等待。

当前实现按 D 个历史周期回退、校正并回放 IMU，不依据每帧采集时间自动选择任意历史时刻。`capture_stamp` 主要用于初始化时序检查；应让实际图像延迟与配置的固定补偿相匹配。此包没有添加动态延迟估计、额外反馈或门控。

当前相机外参为光学系→FLU 旋转矩阵 `[[0,0,1],[-1,0,0],[0,-1,0]]`，相机位置 `[0.14,0,0]` m。默认重力 ENU `[0,0,-9.80665]`，单次预测步长上限 0.05 s，ROS 节点会拆分较长的正 IMU 间隔。接收方应按真实安装确认外参。

代码相对速度传播中没有目标加速度输入，使用无人机加速度传播相对速度，即采用目标加速度为零的模型；这不等于已验证任意机动目标的估计性能。

默认 bootstrap 初始化使用传入的相对状态及协方差，`initial_camera_depth_m=25` 的固定深度参数不会在该路径中替代 bootstrap；它属于已有的另一初始化接口。

核心测试保留原断言，只将依赖控制器的 `rodrigues` 导入改为测试本地数学函数；该函数及其三个辅助函数从原控制器逐字提取，未携带控制器。ROS 测试原样复制，其中显式关闭 bootstrap 要求以测试 reset、IMU 与延迟图像链路，不覆盖默认粗估距上游流程。

导入检查、合成示例、公式/雅可比测试、ROS 消息接口测试和包构建结果均不等于实际闭环验证。本包不附带实飞、Gazebo 拦截或新采集数据的性能结论。

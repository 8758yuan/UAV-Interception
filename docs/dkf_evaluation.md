# DKF 仿真精度记录

实际代码 `paper_state_observer.py` 的 `STATE_DIM=18`，与尚未实现的
`singer_dkf.md` 中 21 维方案不同。本模块按运行代码记录，不追加假状态。

| 索引（从 0 开始） | 状态 | 坐标系、定义 | 单位 |
| --- | --- | --- | --- |
| 0–3 | q_w, q_x, q_y, q_z | Hamilton 四元数，机体 FLU → 世界 ENU | 无量纲 |
| 4–6 | p_r_e,n,u | 无人机机体原点位置 − 目标模型原点位置，ENU | m |
| 7–9 | v_r_e,n,u | 无人机速度 − 目标速度，ENU | m/s |
| 10–11 | image_x,y | 相机光学坐标 x/z、y/z，右、下为正 | 无量纲 |
| 12–14 | b_gyr_f,l,u | 陀螺仪偏置，机体 FLU | rad/s |
| 15–17 | b_acc_f,l,u | 加速度计偏置，机体 FLU | m/s² |

## 自动运行与结果位置

按项目原来的 `启动.md` 启动仿真和视觉拦截即可，无需额外终端或手动运行记录器。
静止目标和运动目标都由终端 4 的
`vision_direct_interception_trial.launch.py` **默认启动** DKF 记录器，
与 `record_bag` 开关无关。沿用本次测试的 `trial_id`，自动保存到：

```
/home/yuan/ibvs_ws/results/p4/dkf/<trial_id>/测试结果.html
/home/yuan/ibvs_ws/results/p4/dkf/<trial_id>/每0.5秒对比.xlsx
/home/yuan/ibvs_ws/results/p4/dkf/<trial_id>/states.csv
/home/yuan/ibvs_ws/results/p4/dkf/<trial_id>/metrics.csv
/home/yuan/ibvs_ws/results/p4/dkf/<trial_id>/metadata.json
```

**平时只需用浏览器打开 `测试结果.html`**：先看位置、速度的中文误差汇总，
再选择状态查看真实值/估计值曲线和每 0.5 秒对比表，其余状态折叠收起。
`每0.5秒对比.xlsx` 可直接用 Excel 打开（已设置分组表头、列宽及冻结窗格，另含误差汇总工作表）。每行对应一个时刻，
包含全部 18 个状态的真实值、估计值和误差。按六组变量展示，每组的真实值、估计值和误差
各占一列，分量写在同一括号内，所有数值保留三位小数。分量顺序为：四元数 `(w, x, y, z)`，
位置和速度 `(东, 北, 上)`，图像坐标 `(x, y)`，IMU 偏置 `(前, 左, 上)`。
HTML 中使用相同横表，向右滚动查看全部列。默认只显示位置和速度，可勾选其他类别、
一键显示全部，或切换“只看误差”。时间列与表头固定，隔行底色便于对齐。
筛选只影响 HTML 显示，不删除每行状态，不改变完整 CSV。

若设置了 `results_directory`，则位置为 `<results_directory>/dkf/<trial_id>/`。
正常试验结束时，现有 coordinator 退出触发 launch 关闭，记录器自动写入最终
统计；Ctrl+C 结束终端 4 也会保存。每行 CSV 实时刷新，统计每秒更新。
复用已存在的 `trial_id` 会明确报错，避免覆盖旧结果；原启动命令已使用日期后缀。
开始和结束时终端都会打印记录位置。

首次更新需要构建，随后继续使用原有启动命令：

```bash
cd /home/yuan/ibvs_ws
source /opt/ros/humble/setup.bash
source /home/yuan/px4_ros2_ws/install/setup.bash
colcon build --packages-select interception_interfaces ibvs_control --symlink-install
source install/setup.bash
```

普通 `vision_direct_interception.launch.py` 也默认带记录器，结果位于
`/home/yuan/ibvs_ws/test_results/dkf/<日期与唯一后缀>/`。
单独运行记录器仍受支持，但正常试验不需要这一步。

记录器默认使用实际机型 `x500_mono_cam_0`，试验 launch 自动读取观测器 YAML
中的相机旋转和安装位置。若仿真改变世界名或机型，在控制 launch 中传入与
仿真一致的 `world:=... vehicle_model_name:=...`。Gazebo 世界需提供
`/world/<world>/pose/info`；两个模型原点分别对应机体原点及视觉目标中心。
记录器只读取 Gazebo 原生真值，不向控制器或 DKF 发送真值。

### 只记录拦截区间

记录器只订阅现有 `/interception/control/debug` 确定起止，不向控制器发送消息。
首次进入 `ACTIVE` 的调试消息时间为开始，首次离开 `ACTIVE` 为结束；
接触退出、失败退出和超时都按实际控制状态执行，不添加新的飞行判据。
起飞、搜索、PRESTREAM 以及退出后的恢复、悬停和降落数据不计入状态 CSV、
误差统计或最终报告。记录使用已对时的状态物理时间进行闭区间筛选；
无法对时的输出只能按发布时间判断是否处于区间，其真值误差仍为不可用。
记录器等待控制时间线确认后再落盘，避免不同话题到达顺序造成区间外样本混入。

元数据保存精确开始和结束时间戳。Excel 的 0 秒行和最后一行分别表示开始和结束，
中间每 0.5 秒一行。结束不是 0.5 秒整数倍时，额外保留结束行。
边界没有恰好对应的 DKF 输出时，首尾使用区间内最近的原始样本，
“实际取样”列明确给出真实采样时间；不会把样本重标为边界时刻或插值补造状态。
若从未进入 ACTIVE，则无拦截数据；若中途退出程序且没有收到退出 ACTIVE，
报告明确标记“结束信号未收到”，不将最后一个样本假称正常结束。
这些记录条件仅影响测试文件，不改变现有状态机、DKF 或控制律。

### 自动时间映射

默认读取现有 `/fmu/out/timesync_status` 的 DDS 来源记录，不需要手动填偏移。
当前 PX4 Gazebo bridge 将单调时钟同步到 Gazebo 仿真时间；DDS 序列化将
`timestamp` 加上 `-TimesyncStatus.estimated_offset`。因此记录器反变换为：

```
Gazebo 微秒 = SensorCombined.timestamp + TimesyncStatus.estimated_offset
```

使用 IMU 时间戳之前最近一条 DDS 同步状态，避免未来更新追溯改变已记录数据。
每行额外保存实际使用的 `clock_offset_us`，所有换算使用整数微秒，避免大纪元
时间相减的浮点损失。同步状态本身有更新周期，序列化期间偏移变化产生的残余
对时误差仍受 PX4 的同步精度和发布周期影响；不以到达时间伪造精确同步。
若 DDS 同步状态未收到，则该行明确标记 `clock_unavailable`，真值和误差 NaN。
此时应检查已有时间同步话题，不能把不可用数据解释为零误差。

仅在单独运行且确实关闭 DDS 时间转换的其他配置中，才需要显式参数
`px4_to_gazebo_offset_s`；参数含义为 Gazebo 秒减去收到的 PX4 秒。
正常启动自动使用当前项目已有的 DDS 同步，不修改 PX4 时间参数。

## 同步和真值边界

观测器新增只读测试输出 `/interception/observer/evaluation`，封装原有完整
`ObserverState` 和最后已处理 IMU 的原始 DDS 时间戳。原有控制话题、状态消息、
DKF 算法与控制律保持原样。没有记录器订阅时，不发布测试副本。

视觉校正在 k−D 进行，随后回放至 k；因此记录的是回放后的**当前 IMU 时刻**，
不从发布时间再减 D×20 ms。原有 `ObserverState.stamp` 是发布时间，单独保留。
初始化尚无 IMU 时间锚点，以及无效输出，保留记录但不计算真值误差。这里记录
的是现有算法的名义状态时刻；初始化图像/姿态的混合时间以及固定 D 与实际
图像延迟不一致引起的误差属于被测系统，本模块不修改或补偿其内部算法。

真值来自 Gazebo 原生位姿流，未采用 PX4 EKF 估计、运动指令或理想轨迹冒充真值。
等待真值样本覆盖状态时刻后，位置线性插值、姿态最短弧 SLERP；禁止时间外推。
每行保存真值两侧样本的时间，便于评估插值分辨率。真值缓冲最多 4096 帧，待写
输出最多等待至 4096 条积压；超出覆盖范围、结束时未覆盖的输出仍写入，真值
标为不可用，绝不悄悄丢掉输出。此容量只限制记录器内存，不影响控制。

相对速度真值列是**两侧实测位姿差分所得的区间平均速度近似**，不是 Gazebo
瞬时速度；尤其目标通过 set_pose 离散移动时，可能表现为台阶差分。因此速度
统计含采样/差分误差，不应解释成精确瞬时速度精度。`truth_status` 和元数据均
明确标记此点。若需要精确瞬时速度指标，需要仿真端提供与实际 set_pose 运动
语义一致的带时间戳速度真值；不能用理想轨迹导数替代实际运动。

归一化图像真值按实际几何计算：
`p_c = C.T @ (R(q).T @ (p_target - p_vehicle) - camera_position_b)`。
非正深度时图像真值不可用。真值四元数选择与原始估计同号，避免 q/−q 产生
虚假分量误差；估计值本身保持原样。

六个 IMU 偏置没有可靠真值来源，记录其估计值、NaN 真值和 NaN 误差，不将
噪声、EKF 偏置或配置标准差当作实际偏置。消息未初始化时估计状态也标为 NaN。

## 结果

每次试验自动生成独立目录（见上文），包含：

- `测试结果.html`：离线中文报告，内嵌对比曲线，不需要网络。
- `每0.5秒对比.xlsx`：每 0.5 秒选取最近的实际输出，保留名义时间和实际取样时间。
  0 秒从进入 ACTIVE 的时间开始，首尾时刻均保留；缺失采样区间标记不可用，不插值，
  不拿远处样本填充缺失区间。汇总统计仍使用全频率原始数据。
- `states.csv`：拦截区间内每次收到的 DKF 输出一行；发布时间、原始 IMU 时间、映射后的
  状态时间、有效性、真值来源状态、真值时间区间、所用时钟偏移，以及全部 18 维的
  `<状态>_true / _estimate / _error`；error = estimate − true。
- `metrics.csv`：每维可用配对数量、RMSE、MAE、最大绝对误差；只统计有限值。
  无可靠真值的状态 count=0，指标 NaN。每秒更新，正常退出时最终更新。
- `metadata.json`：模型名、相机外参、时间偏移、状态顺序和真值近似说明。

CSV 每行刷新；强制终止进程时，统计可能比数据落后至多一个更新周期。
报告在退出时由独立进程使用系统 Matplotlib 生成，不增加飞行期间的绘图开销。
使用 ROS Python 的 `-s` 模式避免用户目录 NumPy 覆盖系统 Matplotlib 的二进制依赖；
不改动系统或虚拟环境的库版本。

验证涵盖合成坐标/符号/时间插值、全部状态 CSV 列、缺失真值与误差统计，
自动 DDS 时钟映射、默认 launch 接入、最终统计保存，以及原有观测器测试和 ROS 消息构建。它不构成 Gazebo 闭环精度或稳定性验证。

已有记录可补生成报告（不修改原始 CSV）：

```bash
/usr/bin/python3 -s /home/yuan/ibvs_ws/src/ibvs_control/ibvs_control/dkf_evaluation_report.py \
  /home/yuan/ibvs_ws/results/p4/dkf/<trial_id>
```

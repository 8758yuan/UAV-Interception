# IBVS 恒定接近速度分量：实现与稳定性复核

依据：项目本地论文 *High-Speed Interception Multicopter Control by
Image-Based Visual Servoing*，正文式 (12)–(29) 和附录式 (37)–(43)。
速度参考使用完整导数；固定相机的安装偏移补偿见第 2、5 节。保留 SO(3) 姿态反馈、
角速度范数饱和、推力投影及物理饱和。未增加门控、限速、模式或接近阶段。

`position_gain` 和 `velocity_gain` 均按 ENU 世界坐标系 `[x, y, z]` 配置，
具体取值见 `src/ibvs_control/config/design_parameters.yaml`。
普通 DKF 拦截和真值测试使用同一控制律及配置，没有测试专用覆盖。
以下位置能量与位置、速度反馈均按正定对角矩阵表示。

## 1. 速度参考和完整导数

沿用相对状态定义 $p=p_u-p_t$、$v=v_u-v_t=\dot p$。令

$$
d=\|p\|>0,\qquad n=-p/d,\qquad P=I-nn^T,
$$

$$
v_d=(k_1d+v_c)n=-k_1p+v_cn,\qquad z_2=v-v_d.
$$

$v_c$ 为恒定正参数，单位 m/s；不是速度上限或实际碰撞速度保证。
由 $\dot d=-n^Tv$、$\dot n=-Pv/d$，得到

$$
\boxed{\dot v_d=-k_1v-\frac{v_c}{d}Pv.}
$$

仅使用 $-k_1v$ 会遗漏视线方向变化产生的横向加速度。实现未用差分估计该导数。
速度参考中的 $n$ 严格由 $p$ 构造；图像反投影方向 $n_i$ 继续用于原 IBVS 项。
这两个方向在论文光心重合、状态精确的假设下相同；有安装偏置或观测误差时可不同。

## 2. 期望加速度

令 $q=n_{td}$、$z_1=1-q^Tn_i$、$b=z_1/(k_b^2-z_1^2)$、
$K_p=\operatorname{diag}(k_x,k_y,k_z)\succ0$，其中
$[k_x,k_y,k_z]=\texttt{position_gain}$；令
$K_v=\operatorname{diag}(\texttt{velocity_gain})\succ0$。
安装位置为机体系常向量 $r$，定义

$$
s=Rr,\quad \ell=-p-s,\quad \rho=\|\ell\|>0,\quad
P_i=I-n_in_i^T,\quad \lambda=k_1+v_c/d.
$$

距离加权前的基线期望加速度与视觉角速度为

$$
\boxed{
a_d=\dot v_d-K_vz_2-K_pp
      -\frac{mb}{\rho}P_iq+a_t,
}
$$

$$
h=q\times n_i-\frac{s\times(P_iq)}{\rho},\qquad
u=\lambda\frac{s\times n_i}{\rho+s^Tn_i},\qquad
\omega_1=R^T(bh+u).
$$

$u$ 是原机体径向速度参考引起的图像漂移补偿，直接由现有状态和外参解析计算；
不恢复光心平移动力学或角加速度差分反馈。$r=0$ 时恢复原视觉项和式 (13)。
视觉项使用光心距离 $\rho$，速度参考及其导数仍使用机体距离 $d$。
保留视觉项中的质量系数 $m$ 和目标加速度补偿 $a_t$（在线仍默认零）。推力与姿态仍由
$F_d=m(a_d-g)-f_{drag}$、$n_{fd}=F_d/\|F_d\|$ 和原最短倾斜旋转生成。
SO(3) 姿态误差反馈 $\omega_2$、执行器限制和原有故障退出流程保持原结构。

### 2.1 视觉加速度距离加权

当前视觉加速度采用用户确认的
$a_{vis,new}=(1+\rho/L)a_{vis,old}$，即将上式中的 $1/\rho$
替换为 $1/\rho+1/L$。$L=\texttt{visual_l_m}>0$，单位为米，
在 `config/design_parameters.yaml` 中配置，初始值为 10 m。
仅视觉加速度项使用此权重，$\omega_1$ 的公式不变；$R_d$ 和 $\omega_2$
会随新的总期望加速度改变。没有增加 10 m 门控或切换条件。

以下第 3–5 节保留基线推导。对于当前有限 $L$ 的控制律，令
$\Delta a_{vis}=-mbP_iq/L$，在采用新 $a_d$、$R_d$ 计算执行残差时，
对应能量导数右侧还须加上 $z_2^T\Delta a_{vis}$；该项不保证非正。
公式测试显式计入此项，不代表距离加权后的完整闭环稳定性或拦截性能已验证，
也不保证 10 m 内视觉项始终占主导。

## 3. 理想平移闭环的 Lyapunov 核对

本节先假设：$k_1,k_x,k_y,k_z,v_c,m>0$、$K_v\succ0$、$0<k_b<2$；
相机光心与机体原点重合，$n_i=n$；状态、模型及目标加速度精确；
连续时间运行，期望力非零且姿态参考局部可微。记 $\delta a=a_u-a_d$ 为实际加速度实现误差，
$\omega$ 为实际机体角速度，$c=q\times n$。
在正深度、Barrier 有效域 $0\le z_1<k_b$ 内，取

$$
B=\frac12\log\frac{k_b^2}{k_b^2-z_1^2},\qquad
V_T=mB+\frac12p^TK_pp+\frac12\|z_2\|^2.
$$

这里使用与原视觉系数匹配的加权候选，而不是不加权照抄论文；
不需要假设 $m=1$，也不需要在代码中改动原视觉增益。
因为 $Pv_d=0$，

$$
\dot B=\frac b d q^TPz_2-bc^TR\omega,
$$

$$
\frac{d}{dt}\left(\frac12p^TK_pp\right)
=-\left(k_1+\frac{v_c}{d}\right)p^TK_pp+p^TK_pz_2,
$$

$$
\dot z_2=-K_vz_2-K_pp-\frac{mb}{d}Pq+\delta a.
$$

视觉横向交叉项和位置交叉项逐项抵消，得到

$$
\boxed{
\dot V_T=-\left(k_1+\frac{v_c}{d}\right)p^TK_pp-z_2^TK_vz_2
         -mbc^TR\omega+z_2^T\delta a.
}
$$

新增 $v_c$ 产生额外的负项 $-(v_c/d)p^TK_pp$。
正定对角位置增益保留位置交叉项抵消；三轴相等时退化为原标量形式。

## 4. 零偏移时完整 SO(3)、推力及角速度闭环的条件

不能仅由上一节认定实际完整闭环稳定。定义

$$
Q=R_d^TR,\quad e=\operatorname{vex}(Q-Q^T),\quad
w=bR^Tc-e,\quad \omega_{cmd}=\operatorname{sat}(w,\omega_m)=\kappa w,
$$

其中 $\kappa=1$（未饱和）或 $\omega_m/\|w\|$（饱和）。
设 $\dot R_d=R_d[\Omega_d]_\times$，$\eta_d=R^TR_d\Omega_d$，
$\delta\omega=\omega-\omega_{cmd}$。对

$$
\Psi=\operatorname{tr}(I-Q),\qquad V=V_T+m\Psi,
$$

完整姿态项导数是 $\dot\Psi=e^T(\omega-\eta_d)$，
**随时间变化的 $R_d$ 会产生 $-e^T\eta_d$，不能省略**。因此

$$
\boxed{
\dot V=-D+z_2^T\delta a-mw^T\delta\omega-me^T\eta_d,
}
$$

$$
D=\left(k_1+\frac{v_c}{d}\right)p^TK_pp+z_2^TK_vz_2+m\kappa\|w\|^2.
$$

推力项中的误差必须使用实际投影和饱和后的推力计算：

$$
\delta a=g+\frac{f_{cmd}}mRe_3+\frac{f_{drag}}m-a_d,
\quad
f_{cmd}=\operatorname{clip}((Re_3)^TF_d,0,f_{max}).
$$

即使推力没有饱和，只要实际推力方向尚未对齐 $F_d$，通常仍有 $\delta a\ne0$。
由于当前 SO(3) 控制保留原结构，没有加入期望姿态角速度前馈，
也不能默认期望姿态运动项已被抵消。

一个可核查的充分下降条件是：沿接触前轨迹存在固定 $0<\epsilon<1$，满足

$$
z_2^T\delta a-mw^T\delta\omega-me^T\eta_d+r_{other}
\le (1-\epsilon)D.
$$

此时 $\dot V\le-\epsilon D$。更保守的充分条件是用各残差范数乘积上界替代左侧。
$r_{other}$ 需包含相机几何、状态估计、目标加速度误差和采样保持带来的残差。
这只是分析条件，**没有变成新增软件阈值或门控**。

若初始 $V$ 有限且上述条件持续成立，Barrier 项有界，从而在到达接触边界前
不会触及 $z_1=k_b$。这不是对任意执行器饱和、观测误差或外扰的无条件保证。

## 5. 非零安装偏移及实际闭环的残差

当前相机位置为 `[0.14, 0, 0]` m，观测器仍估计机体原点状态；
相机补偿不更改 DKF。状态、姿态和图像几何一致，即 $n_i=\ell/\rho$ 时，

$$
\dot\ell=-v-R(\omega\times r),\qquad
\dot B=\frac b\rho q^TP_iv-bh^TR\omega.
$$

由 $v_d=\lambda(\rho n_i+s)$ 及第 2 节的 $u$ 可验证

$$
\frac1\rho q^TP_iv_d=h^Tu.
$$

因此，同一个机体原点能量候选满足

$$
\dot V_T=-\lambda p^TK_pp-z_2^TK_vz_2
          -mbh^T(R\omega-u)+z_2^T\delta a.
$$

光心距离和杆臂梯度修正抵消平移视觉交叉项；$u$ 抵消径向参考引起的漂移。
仅将视觉项的 $d$ 换成 $\rho$ 会遗漏后者。定义

$$
e=\operatorname{vex}(R_d^TR-R^TR_d),\quad g_b=bR^Th,\quad
u_b=R^Tu,\quad w=g_b-e,\quad \omega_{cmd}=\kappa(w+u_b).
$$

这里 $\kappa$ 由完整角速度 $w+u_b$ 的原范数饱和确定。
保留实际推力投影、饱和、角速度跟踪误差以及时变 $R_d$ 后，完整导数为

$$
\boxed{
\begin{aligned}
\dot V={}&-\lambda p^TK_pp-z_2^TK_vz_2-m\kappa\|w\|^2\\
 &+z_2^T\delta a-mw^T\delta\omega-me^T\eta_d\\
 &+m\big[(1-\kappa)g_b+\kappa e\big]^Tu_b.
\end{aligned}
}
$$

最后一项没有被忽略：未饱和时仍有 $me^Tu_b$，饱和时还有漂移补偿无法完整执行的影响。
在第 4 节的充分下降条件中，须将该项纳入残差左侧，并用本节的 $w$ 计算 $D$；
它与姿态、推力执行残差都可能为正，不能据此宣称完整闭环稳定。
零偏移时 $u_b=0$，退化为第 4 节的恒等式。

此外还存在以下实际差异：

- 在线控制使用 DKF 估计值。EKF 的线性化协方差更新不是完整非线性闭环误差收敛证明。
  状态、图像和姿态不一致时，几何恒等式还会产生估计残差，须另行验证下降条件。
- 外环 50 Hz 保持输出、内环 200 Hz 更新。连续时间恒等式不能直接作为离散闭环证明；
  若把 $R_d$ 看作分段保持量，还需计入外环更新时 $V$ 的跳变量。
- SO(3) 反向姿态存在经典退化临界点，当前最短倾斜旋转也在反平行情况下不唯一。
  不据此声称全局姿态渐近稳定。

本次验证的是几何与能量导数恒等式及其执行残差；**不能宣称当前带观测器、偏置相机、
采样和饱和的整个飞行系统已获得无条件稳定性证明**。

## 6. 接触性质与奇异点

距离动力学为

$$
\dot d=-k_1d-v_c-n^Tz_2.
$$

若接触前始终有 $\|z_2\|\le\zeta<v_c$，则
$\dot d\le-k_1d-(v_c-\zeta)$，由比较原理得到有限接触时间上界

$$
T\le\frac1{k_1}\log\left(1+\frac{k_1d(0)}{v_c-\zeta}\right).
$$

该跟踪误差界需要证明或验证，不是代码中新加的条件；它也不由 $k_1,v_c>0$、$K_v\succ0$
自动保证。$z_2\equiv0$ 时有相应的理想运动学等式，但由于保留了位置反馈，
$z_2=0$ 通常不是完整加速度闭环在 $p\ne0$ 时的不变流形。

$d=0$ 时 $n$ 无定义，且到达时参考速度模长趋于 $v_c>0$，所以目标点不是
“位置和速度都渐近归零”的平衡点。证明的有效区间是接触前 $d>0$，不能将控制律
直接延拓到穿越目标后的 $d=0$。
横向项 $v_cPv/d$ 在小距离且仍有横向速度时可很大；有界前馈要求 $Pv=O(d)$。
物理执行器饱和继续保留，不能用理论期望加速度替代实际可实现加速度。

实现将原有微米量级的距离数值奇异性检查同时用于机体距离和光心距离；
$\rho+s^Tn_i=0$ 时漂移补偿无定义，走原有无效控制退出路径。
当前前置安装和前向有效视线下该分母为正。保留 Barrier 定义域和有限值检查，
没有新增接近距离门限、分母正则化或输出清零。

## 7. 参数与使用

`src/ibvs_control/config/design_parameters.yaml` 中的 `v_c` 是在线参数入口，
当前 `0.1` m/s，并非飞行调参结论。普通拦截和真值测试 launch 均加载此文件。
修改参数后重启控制节点；未增加运行中突然切换参考速度的回调。
纯数学配置 `OuterLoopConfig.v_c` 默认仍为 `0.5` m/s。
专用低速 P2 离线基准的 `default_controller_config()` 显式采用 `0.1` m/s，
对应 `offline_interception.yaml`；其原有 2 m/s、20° 验收条件保持不变。
这两套参数分别用于在线控制和已有低速基准，不包含运行时模式切换。
参数拒绝零、负数、NaN 和无穷值。

`position_gain` 是长度为 3 的浮点数组，例如 `[0.2, 0.3, 0.8]`，
分别控制 ENU 世界坐标系 x、y、z 方向的位置反馈。各分量须为有限正数。
当前全局调参值为 `[0.05, 0.05, 1.0]`。
修改 YAML 后重启控制节点生效；原标量配置需改为三元素数组。

`velocity_gain` 是长度为 3 的有限正数数组，当前全局值为 `[0.8, 0.8, 3.0]`，
替代原标量 `paper_k2`，反馈为 `-diag(velocity_gain) * z2`。
控制节点默认值与全局 YAML 一致；不再读取 `paper_k2`。
纯数学配置改用 `OuterLoopConfig.velocity_gain`；等轴数组 `[k2, k2, k2]`
与原标量反馈等价。已有低速离线基准沿用等价的 `[20.0, 20.0, 20.0]`。
正定分轴增益保留耗散项 `-z2.T * K_v * z2`，这不构成完整闭环稳定性证明。
本次没有改变推力投影、视线控制项、姿态反馈或执行器约束。

## 8. 验证范围

回归测试覆盖不同距离的径向参考、带横向运动的中心差分导数、
图像与几何方向不一致时的参考定义、完整加速度、ROS 参数传递，以及任意正质量下
含推力/角速度误差和期望姿态运动项的 Lyapunov 导数恒等式。
这些测试核查实现与推导的一致性，不替代闭环误差界证明或 Gazebo/实机验证。

2026-10-19 恒定接近速度修改时的历史验证结果（分轴增益修改前）：

- `colcon build --symlink-install --packages-select ibvs_control` 构建通过。
- 控制包非 linter 测试：188 项通过，1 项已有旧增益断言失败。
  失败用例 `test_flight_gains_limit_far_range_acceleration` 硬编码
  `k2=2.5`、`position_gain=0.3`，当前已有配置为 `3.0`、`0.8`，本次未改这些增益。
- 专用低速离线参数 `v_c=0.1` 的静止、移动目标观测器闭环均命中，
  用时分别为 26.26 s、27.53 s，最大速度约为 1.66 m/s、1.67 m/s，
  最大倾角约为 16.10°、16.09°，保留原验收条件。
- `v_c=0.5` 在同一个低速离线基准中的最大速度约 2.4 m/s，
  超出该基准原有 2 m/s 验收条件。这说明参数选择仍需结合具体场景验证；
  在线默认值不是经过飞行标定的结论。
- 未执行新的 Gazebo 或实机飞行。

### 三轴位置增益验证（2026-10-19）

- 控制律、离线拦截、观测器闭环、ROS 参数接入与数据流检查共 78 项通过。
- 验证单独调整一个增益仅改变对应 ENU 加速度反馈分量。
- 对等轴和不等轴正增益分别核对完整 Lyapunov 导数恒等式及中心差分结果。
- 原固定增益断言已改为三轴配置合法性检查，不再锁定历史调参值。
- 未执行新的 Gazebo 或实机飞行；分轴可调不等于已验证水平平滑性改善。

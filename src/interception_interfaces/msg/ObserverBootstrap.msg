# Relative state estimated from pre-interception bearings and onboard motion.
# p_r = p_vehicle - p_target; v_r = v_vehicle - v_target, both in ENU.
builtin_interfaces/Time stamp
geometry_msgs/Vector3 p_r
geometry_msgs/Vector3 v_r
# Row-major covariance for [p_r(3), v_r(3)].
float64[36] relative_covariance
uint32 sample_count
float64 range_m
float64 range_std_m

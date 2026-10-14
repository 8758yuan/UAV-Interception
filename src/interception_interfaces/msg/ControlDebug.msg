builtin_interfaces/Time stamp
float64 z1
geometry_msgs/Vector3 z2
geometry_msgs/Vector3 acceleration_d
float64[9] attitude_d
geometry_msgs/Vector3 omega1_b
geometry_msgs/Vector3 omega2_b
geometry_msgs/Vector3 omega_d_b
float64 thrust_n
float64 thrust_normalized
bool thrust_saturated
bool thrust_mapping_saturated
bool rate_saturated
bool valid
string reason

# V3 radial reference diagnostics; these do not gate the control output.
geometry_msgs/Vector3 relative_velocity_d
geometry_msgs/Vector3 relative_acceleration_d
bool approach_profile_enabled
float64 estimated_distance_m
float64 reference_distance_m
float64 closing_speed_m_s
float64 approach_target_speed_m_s
float64 approach_speed_m_s
float64 approach_acceleration_m_s2
float64 approach_weight
float64 approach_horizon_s
uint64 approach_replan_count

# Always-populated transition diagnostics.  These make the two seconds around
# visual terminal entry auditable even after the IBVS output itself is retired.
geometry_msgs/Vector3 attitude_rpy_rad
geometry_msgs/Vector3 angular_velocity_b
geometry_msgs/Vector3 velocity_ned
geometry_msgs/Vector3 commanded_body_rates
geometry_msgs/Vector3 commanded_velocity_ned
geometry_msgs/Vector3 commanded_acceleration_ned
float64 commanded_thrust_normalized
float64 image_error_norm
float64 target_area_ratio
bool target_detected
bool impact_detected
string control_state

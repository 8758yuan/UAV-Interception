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

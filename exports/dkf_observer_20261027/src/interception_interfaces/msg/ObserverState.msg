# Paper state order: x = [q, p_r, v_r, image_xy, b_gyr, b_acc] in R^18.
builtin_interfaces/Time stamp
float64[4] q
geometry_msgs/Vector3 p_r
geometry_msgs/Vector3 v_r
float64[2] image_xy
geometry_msgs/Vector3 b_gyr
geometry_msgs/Vector3 b_acc
float64[324] covariance
uint64 prediction_count
uint64 correction_count
float64 innovation_norm
bool initialized
bool valid
string reason

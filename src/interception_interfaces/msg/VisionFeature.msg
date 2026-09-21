# Image-plane target observation.  Both timestamps use the shared simulation
# clock so the DKF can audit capture-to-publication delay.
builtin_interfaces/Time capture_stamp
builtin_interfaces/Time publish_stamp
uint64 sequence
float64 u
float64 v
float64 x_norm
float64 y_norm
float64 area_px
bool valid
string reason

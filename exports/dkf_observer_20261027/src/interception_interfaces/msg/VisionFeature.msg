# Image-plane target observation.  Both timestamps use the shared simulation
# clock so the DKF can audit capture-to-publication delay.
builtin_interfaces/Time capture_stamp
builtin_interfaces/Time publish_stamp
uint64 sequence
float64 u
float64 v
float64 x_norm
float64 y_norm
# Bounding-box width in original image pixels divided by camera fx.
float64 width_norm
float64 area_px
bool valid
string reason

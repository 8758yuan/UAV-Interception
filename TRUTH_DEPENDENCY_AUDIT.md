# Truth dependency audit

## Removed command-path dependencies

The former `vision_direct_state` path subscribed to
`/interception/truth/relative_state`, copied `p_r` and `v_r`, combined them
with an image-derived LOS, and fed that hybrid message to the truth
coordinator.  The coordinator then used relative distance for terminal attack
and Gazebo contact for its flight-state transition.  That hybrid node and its
executable have been removed from the active visual launch.

The command-producing visual node is now
`vision_interception_coordinator`.  It has no `RelativeState`, target odometry,
Gazebo contact, target position, target velocity, relative position, relative
velocity, range, or time-to-collision subscription/input.

The former truth coordinator is no longer installed as a ROS executable and
also rejects `enable_flight_commands=true` if its Python module is invoked
directly.  It remains only as historical/evaluation code for old P2 records.

## Active data flow

```text
Gazebo onboard camera /camera/image_raw
  -> red_target_detector
     -> (u, v, x_norm, y_norm, area_px)
  -> vision_interception_coordinator
     -> image LOS barrier term (paper equation 13 form)
     -> image-scale approach-speed schedule
     -> desired thrust direction + attitude-rate feedback
  -> Px4RateThrustCommand
  -> /fmu/in/vehicle_rates_setpoint
  -> PX4
```

PX4 vehicle attitude/velocity/local position are the interceptor's onboard
state.  They are used to realize and safety-limit commands and to recover after
the terminal commit.  They are never combined with any target world state.

## Visual terminal decision

The primary terminal commit requires both:

- segmented target area / image area >= `terminal_area_ratio`;
- normalized image-center error <= `terminal_max_center_error`.

A visual-only fallback covers the case where a propeller/contact-induced color
change removes the red segmentation just before the primary scale threshold.
It commits only when the last valid frame was recent (at most 0.30 s), centered,
and already close by image area.  This test uses only the current detection bit
and cached camera features; it has no contact or simulator-state input.

The controller then retires IBVS, preserves the measured velocity for the
short bounded `post_hit_coast_duration_s`, and smoothly decays that velocity
to zero before entering hover.  No distance threshold, relative state, or
contact event enters this decision.

## Evaluation-only sidecar

`p4_vision_monitor.launch.py` still includes the simulation truth/contact
monitor so bags can measure miss distance and prove physical collision.  The
trial launch records those evaluation topics, but the controller has no
subscriptions to them.  `test_vision_control_dataflow.py` enforces this
separation at source/launch level.

## SITL validation

The latest corrected PX4/Gazebo run entered the visual impact cue at wall time
`1789564420.4098`.  The independent Gazebo contact evaluator reported the
airframe/target collision at `1789564420.5002`, about `0.09 s` later, and
confirmed the red-to-green material change at `1789564420.6081`.  The vehicle
then completed its smooth stop and entered stable `HOVER`.

The fresh `live_postimpact_check_02` rerun again triggered the primary image
scale cue, physically contacted the target, received the evaluation-only green
confirmation, and reached `HOVER`.  During that run,
`ros2 node info /vision_interception_coordinator` listed only
the camera feature topic plus PX4 interceptor attitude/status/odometry inputs.
It listed no `/interception/truth/relative_state`, target odometry, or Gazebo
contact subscription.  The contact event therefore validates the physical
outcome but cannot cause any controller or flight-state transition.

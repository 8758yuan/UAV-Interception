# Post-impact instability audit

## Root cause

The failing `vision_verify_green_01` run proves that the IBVS law was not
continuing after the target crossed the camera.  The coordinator switched from
body-rate IBVS to velocity mode 74 ms before physical contact.  The red target
became invalid 176 ms after contact, when IBVS was already inactive.

The target model was instead declared `<static>true>`.  Gazebo therefore
treated the 0.25 m target as an infinite-mass rigid obstacle.  The terminal
velocity setpoint also continued to request about 1.43 m/s through that
obstacle.  The result was a 1511.8 N peak contact force, not a plausible
light-balloon disturbance.

In the first 0.25 s after contact, body rates reached 16.0, 13.6, and
24.7 rad/s.  During the following two seconds, all four motor outputs reached
their 0/1 limits.  Before contact, the IBVS rate command was only about
0.071 rad/s, normalized thrust was about 0.74, and neither rate nor thrust
saturation was active.

## Minimal corrections

- The balloon is now dynamic, has 0.02 kg mass, has gravity disabled so it
  floats before impact, and uses low-friction, low-restitution soft contact.
- The pre-impact visual IBVS law is unchanged.
- Visual terminal entry immediately retires IBVS.  The state flow is
  `IMPACT_DETECTED -> EXIT_INTERCEPTION -> STABILIZE -> HOVER`.
- Exit starts with the measured vehicle velocity, holds it for 0.20 s to clear
  the balloon without a setpoint jump, and then applies a three-second cubic
  smooth-stop profile.  Velocity never reverses and acceleration is zero at
  both profile endpoints.
- `ControlDebug` remains active after IBVS exit and records attitude RPY, body
  rates, velocity, commanded body rates / velocity / acceleration / thrust,
  image error, target detection, impact cue, and control state.

## SITL comparison

| Metric | Before | After |
|---|---:|---:|
| Vehicle-contact duration | 0.243 s | 0.105 s |
| Peak vehicle/balloon contact force | 1511.8 N | 30.0 N |
| Peak body rates in first 2 s, max axis | 27.19 rad/s | 0.73 rad/s |
| Motor saturation samples in first 2 s | 20 / 20 | 0 / 20 |
| PX4 thrust setpoint range in first 2 s | -1.00 to -0.12 | -0.739 to -0.722 |
| Post-impact outcome | repeated flips, recovery timeout | stable `HOVER` |

The corrected `vision_postimpact_fix_01` run retained real target/vehicle
contact and published `green_confirmed=true`.  It reached `STABILIZE` 0.228 s
after contact and `HOVER` 4.229 s after contact.  In hover, roll and pitch were
approximately -0.09 and -0.05 degrees and vehicle speed was about 0.02 m/s.

Gazebo contact and truth remain evaluation-only.  The active controller's
impact cue is derived from image scale and centering.  A secondary image-only
cue handles abrupt loss of a recently centered, already-large red target; it
uses only cached camera features and their age.  Neither path reintroduces
target truth or Gazebo contact into the flight-control path.

## Fresh runtime verification

The `live_postimpact_check_02` no-bag rerun triggered the primary scale cue,
then independently confirmed real contact and the red-to-green target change.
The observed state chain was `ACTIVE -> IMPACT_DETECTED -> EXIT_INTERCEPTION
-> STABILIZE -> HOVER`.  The image cue preceded contact by about 0.20 s.

In the PX4 ULog impact window, maximum absolute roll/pitch were 0.25/3.79
degrees, body-rate norm peaked at 0.354 rad/s, thrust stayed between -0.740 and
-0.734, and none of 60 motor samples saturated.  A live hover sample measured
roll/pitch at 0.08/-0.18 degrees, body-rate norm below 0.006 rad/s, and speed
about 0.019 m/s.  The full workspace test run passed 119 tests.

# Architecture

```text
CameraPacket / PirPacket
  -> marker detector + object detector adapters
  -> VisionSensorFusion -> SensorFrame
  -> cart authorization + waiting window + hazard-zone fusion
  -> orchestrator + state machines -> RuntimeOutput / ActuatorOutput
  -> optional signal sinks
```

`FusionDecisionEngine` consumes observations rather than raw images. Camera
health checks, startup/FPS gates, and signal transport are additional layers;
calling the fusion engine alone does not automatically apply all those layers.

The waiting-window module supports several experimental stability strategies.
The anchor-history strategy checks whether a marker remains near its anchor
over a configurable period instead of relying only on instantaneous velocity.
Thresholds depend on scale, camera geometry, and the selected mode.

Object detectors return typed observations. ROI motion and skin-gated motion
use distinct labels because neither establishes semantic human identity.
Human/limb labels are treated separately from cart-body exemptions. This
distinction is covered with synthetic regression scenes, not real-world
detection-rate evidence.

Camera monitors inspect image quality and timing. A consumer-side watchdog
detects stale signals independently of the producer process. These components
are software utilities, not a redundant or certified safety architecture.

The demo actuator and business adapters are mocks. Live signal sinks are
explicit opt-ins; the MQTT sink establishes a network connection if selected.
External actuator integration, authenticated messaging, mechanical stopping
time, sensor coverage, and complete system risk evaluation are outside the
validated offline reference.

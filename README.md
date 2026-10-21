# SafeZone Fusion

An experimental Python framework for camera-assisted docking and hazard-zone
decisions. It combines marker authorization, waiting-window stability, object
observations, health gates, and watchdogs into inspectable decisions.

Maintained by [wenzhu1014](https://github.com/wenzhu1014). This candidate is
prepared from the owner's June 2026 Python release archive. It is intended for
learning, simulation, and integration research. It is not a certified safety
controller; automated tests do not establish safe operation of machinery.

## What is included

- ArUco marker recognition and multiple waiting-window stability strategies,
  including position history around an anchor.
- Layered object-detector interfaces: mock observations, ROI occupancy/motion,
  skin-gated motion, and an optional MediaPipe hand adapter.
- Configurable following zones, cart-body exemptions, and explicit human/limb
  handling in the fusion engine.
- Camera health and operating-state monitors, signal sequencing, heartbeat,
  consumer watchdog utilities, and diagnostic records.
- Synthetic scene tests, fake-camera tests, and offline examples.

See [ARCHITECTURE.md](ARCHITECTURE.md) for boundaries and
[VALIDATION.md](VALIDATION.md) for what has actually been tested.

## Start with the offline examples

Run these commands from the repository root using Python 3.11:

```sh
python -m venv .venv
# Activate .venv using the command appropriate to your shell.
python -m pip install -r requirements-dev.txt
python -m safezone_fusion.demo
python -m pytest safezone_fusion/tests -q
```

The default demo uses synthetic observations. It does not open cameras or
connect to a broker. The default test selection excludes the `slow` recorded
clip suite because its original input videos are not distributed.

`safezone_fusion/config.yaml` contains example identifiers and thresholds,
not a validated deployment configuration. Mock object detection is the default.
Marker IDs are visual identifiers, not proof of identity or secure credentials.

## Optional live work

The camera tools are separate entry points. Running them explicitly can open
local cameras, write images or diagnostic files, and, if signal sinks are
specified, send messages. They have not been exercised on hardware during
preparation of this candidate. A semantic detector, ROI geometry, and camera
parameters require their own evaluation; motion and skin-motion outputs are not
equivalent to confirmed people or hands.

The MediaPipe adapter is optional. It requires a compatible MediaPipe Tasks
environment and a user-supplied local HandLandmarker `.task` model via
`mediapipe_hand.model_path`. It does not download models automatically. Model
files and third-party dependency licenses are separate from this repository.
Real MediaPipe inference has not been validated in this candidate.

## Repository layout

```text
safezone_fusion/
  config.yaml          # demonstration configuration
  demo.py              # offline synthetic scenarios
  src/                 # fusion, state, geometry, monitors, adapters
  src/vision/          # camera packets and detector implementations
  tests/               # synthetic and fake-camera regression tests
  tools/               # explicit camera/calibration/replay entry points
  docs/design/         # historical prototype engineering notes
```

Historical design notes retain their original context. Their earlier test
counts, proposed thresholds, and acceptance instructions are not current
certification or performance claims.

## Contributing and licensing

See [CONTRIBUTING.md](CONTRIBUTING.md), [PROVENANCE.md](PROVENANCE.md), and
[LICENSE](LICENSE). The project code is distributed under the MIT license.
AI assistance was used to prepare documentation, packaging,
review, and targeted changes for this public candidate. No adoption, production
deployment, certification, or benchmark result is claimed without evidence.

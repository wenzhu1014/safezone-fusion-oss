# Validation record

Checked on 2026-10-22 on local Windows with Python 3.11.7 and a clean
GitHub-hosted Windows runner with Python 3.11.9.

## Default regression suite

```text
python -m pytest safezone_fusion/tests -q
212 passed, 1 deselected
```

The deselected test is marked `slow` and requires original recorded camera
clips, which are not included. The passing suite uses synthetic observations,
generated image arrays, fake capture objects, temporary files, and controlled
dependency-failure tests. It does not demonstrate actual MediaPipe inference,
camera coverage, mechanical stopping behavior, or deployed system reliability.

The unchanged archive initially produced 209 passes, one failure, and one
deselected test in the current environment. The failure concerned a test that
expected MediaPipe dependency loading at construction time, while the adapter
loads lazily. The revised test exercises the dependency error at first actual
use. Two additional regressions verify explicit local model requirements and
that disabled/empty inputs do not initialize a model.

## Offline demo and build

`python -m safezone_fusion.demo` completed with synthetic inputs. Human/limb
scenarios emitted STOP decisions. This demo does not simulate elapsed time
between all observations, so initial authorized-cart scenarios may still wait
for the configured stability interval; it is not a timing acceptance test.

A Python wheel was built locally, including package code, demo configuration,
license, and metadata. The wheel was installed without dependency downloads into
a separate temporary target; all 14 synthetic demo scenarios completed from
that installed package, with six STOP decisions. The default environment has unrelated legacy build
plugins and an older standalone `packaging`; the build used the newer
packaging bundled with the installed setuptools. Local online dependency
installation did not complete; the clean GitHub runner subsequently installed
the declared dependencies and passed all 212 default regression tests. The
demo explicitly writes UTF-8 so its Chinese scenario descriptions also work
on redirected Windows output.

## Dependency snapshot

The local passing run used NumPy 2.4.6, OpenCV runtime 4.13.0, PyYAML 6.0.2,
Pillow 12.3.0, and pytest 7.4.0. Requirements pin the corresponding installed
distribution versions for reproduction. The existing host has multiple OpenCV
distributions installed; use just the declared `opencv-python` distribution in
a clean environment. The optional MediaPipe runtime is not part of the default
dependency set or a validated inference setup.

The included GitHub workflow installs the declared dependencies, runs the
default suite, and executes the synthetic demo. Its current result is available
in [GitHub Actions](https://github.com/wenzhu1014/safezone-fusion-oss/actions).
No cross-platform support matrix or performance benchmark is claimed.

## Publication review

Static review found no common API-key/private-key patterns, literal credential
assignments, email addresses, or personal home-directory paths in the curated
source set. Generic Windows font paths and localhost MQTT examples remain.
This is a bounded static check, not an exhaustive security audit.

No recordings, model binaries, compiled Android SDK, private Git history,
duplicate Word documents, or calibration outputs are included. Example
thresholds and identifiers are not evidence of validated field parameters.

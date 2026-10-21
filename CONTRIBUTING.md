# Contributing

Use Python 3.11 and an isolated environment. Install `requirements-dev.txt`,
then run `python -m pytest safezone_fusion/tests -q` from the project root.
Default tests use synthetic frames and fake capture objects. Do not add live
camera access, broker connections, model downloads, or machine actuation to
the default test suite.

For changes to thresholds, exemptions, signal freshness, or state transitions,
describe the input scene, expected decision, and relevant regression coverage.
Keep motion-only observations distinct from semantic human labels. Document
which checks run outside the core fusion engine.

Do not commit recorded people, identifiable site images, local model files,
credentials, deployment identifiers, or private configurations. Use synthetic
examples when reporting a bug. Claims about real hardware, timing, or detection
performance need a stated setup and reproducible evidence.

Issues and pull requests are welcome. Identify
AI-assisted changes and explain the review and validation you performed.

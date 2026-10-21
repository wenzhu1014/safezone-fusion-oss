# Provenance and scope

The owner, GitHub user `wenzhu1014`, confirmed that SafeZone is a personal project
which they may open source. This public candidate was prepared from
`safezone-fusion-release-20260622-203124.zip` on 2026-10-22.

An existing private repository contains 11 commits linked by GitHub to that
account, dated 2026-06-18 through 2026-06-22. This is private development
history, not a claim that the project was publicly available on those dates.
This candidate does not import that private repository's complete history.

The archive is a related but different snapshot: comparison against the
private repository at `e70f1ec8cd612755924bce97afdec355c1397179` found 58
byte-identical files, 17 differing same-path files, and 49 files without a
same-path counterpart after stripping the archive's top directory. Relocated
documents are included in the last category; it is not a count of new code.

Preparation retained the Python implementation, synthetic tests, interfaces,
and selected engineering notes. It added repository packaging and public
documentation. It removed duplicate Word documents, commercial integration and
certification-oriented briefs, a machine-specific launcher, and the Word
conversion utility. Recordings, models, calibration results, private Git
history, and the Android SDK binary are not included.

Targeted code changes make the optional MediaPipe model explicitly local and
avoid initialization when the adapter is disabled or input is empty. Its
dependency-failure test was corrected to exercise first use rather than lazy
construction. The fusion, geometry, and state-decision algorithms were retained.

AI assistance supported this candidate's review, documentation, packaging, and
targeted changes. Repository ownership and linked commits do not establish
exclusive manual authorship of every line. No users, downloads, stars,
certification, or production adoption are claimed here.

The proposed MIT license applies to project code published with this candidate.
Dependencies and separately obtained models retain their own licenses.

# Submitting Grid Mapper 1.5.0 to the official QGIS Plugin Repository

1. Update `https://github.com/MugambiD/qgis-grid-mapper` so its `grid_mapper/` source matches this build.
2. Smoke-test `grid_mapper-1.5.0.zip` in a clean QGIS profile, including the resumable country scanner.
3. Create/tag `v1.5.0` in GitHub and optionally attach the ZIP to a GitHub Release.
4. Sign in to the QGIS Plugin Repository with an OSGeo ID and upload **`grid_mapper-1.5.0.zip`**.
5. Review automated security and compatibility checks and fix blocking findings before manual approval.
6. Confirm homepage, repository and issue-tracker links resolve publicly.

The package has one top-level `grid_mapper/` folder with `metadata.txt`, `__init__.py` and `LICENSE`, and intentionally omits model weights and compiled runtimes.

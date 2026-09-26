# Grid Mapper v1.5.1 — security hardening and repository cleanup

## Plugin fixes

- Escape generated XML labels with `core/xmltext.py`, without importing an XML parser.
- Use SHA-256 for plugin cache keys, AI deduplication, feedback sample IDs and dataset splits.
- Use fixed SQL statements and bound parameters for country-scan queue lookups.
- Report failures while probing TFLite runtimes.
- Rename Roboflow key constants to `KEY_PARAM`, preserving the public `API_KEY` Processing parameter.

These fixes are already present in the canonical plugin and match the v1.5.1 reference package. This cleanup preserves that implementation, including its identifiers and splitting behavior.

## Upgrade note

The v1.5.1 hashing change means newly captured feedback does not deduplicate against v1.5.0 sample IDs. Existing feedback records retain their stored split. Older cache entries may be regenerated. This repository cleanup introduces no further hashing changes.

## Repository and release checks

- Keep `grid_mapper/` as the single plugin package; remove 32 verified duplicate root source and asset files, including stray `__init__` copies.
- Preserve root documentation, previous release notes, notebooks, scripts, license and workflows.
- Update the root README to v1.5.1; plugin metadata and its release test already expect 1.5.1.
- Ignore build output, Python caches, virtual environments and local security reports.
- Run Bandit on the shipped plugin and detect-secrets on tracked repository files in both quality and release workflows. Potential secrets fail CI; scanner output alone is not treated as a passing check.

## Validation

Run the commands in the root README before release. The build script packages only `grid_mapper/`, excluding tests, notebooks and Python cache files. Automated checks do not replace a live QGIS smoke test.

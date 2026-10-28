"""Formula 1 Race Intelligence — real-data analytics package.

Data sources (both open, no credentials required):
  * OpenF1  — official timing-derived laps, sectors, stints, pit stops, telemetry
  * Jolpica — Ergast-compatible results, grid, qualifying, standings

Nothing in this package fabricates data. Every figure in ``outputs/`` is
derived from a downloaded session and carries its provenance in
``data/processed/manifest.json``.
"""

__version__ = "2.0.0"

"""Persistent, resumable state for country-scale Grid Mapper scans.

The state database is deliberately plain SQLite so it can be inspected outside
QGIS and unit-tested without QGIS dependencies. OSM collection and optional AI
imagery scanning use independent queues: a country can be fully mapped from OSM
first, then the same segments can be processed by a model later.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

SCHEMA_VERSION = 1

# Every statement below is a fixed string: the only values that vary between
# runs (attempt ceiling, row limit) are bound as SQL parameters.
_LIMIT_CLAUSE = " LIMIT ?"

_OSM_QUEUE_SQL = (
    "SELECT q.* FROM segments q WHERE q.status IN ('pending','retry') AND q.attempts < ? "
    "ORDER BY CASE q.status WHEN 'retry' THEN 0 WHEN 'failed' THEN 1 ELSE 2 END, q.segment_id"
)
_OSM_QUEUE_WITH_FAILED_SQL = (
    "SELECT q.* FROM segments q WHERE q.status IN ('pending','retry','failed') AND q.attempts < ? "
    "ORDER BY CASE q.status WHEN 'retry' THEN 0 WHEN 'failed' THEN 1 ELSE 2 END, q.segment_id"
)
_AI_QUEUE_SQL = (
    "SELECT a.*, s.row_idx,s.col_idx,s.west,s.south,s.east,s.north "
    "FROM ai_segments a JOIN segments s ON s.segment_id=a.segment_id "
    "WHERE s.status='done' AND a.status IN ('pending','retry') AND a.attempts < ? "
    "ORDER BY CASE a.status WHEN 'retry' THEN 0 WHEN 'failed' THEN 1 ELSE 2 END, a.segment_id"
)
_AI_QUEUE_WITH_FAILED_SQL = (
    "SELECT a.*, s.row_idx,s.col_idx,s.west,s.south,s.east,s.north "
    "FROM ai_segments a JOIN segments s ON s.segment_id=a.segment_id "
    "WHERE s.status='done' AND a.status IN ('pending','retry','failed') AND a.attempts < ? "
    "ORDER BY CASE a.status WHEN 'retry' THEN 0 WHEN 'failed' THEN 1 ELSE 2 END, a.segment_id"
)
_STATUS_COUNT_SQL = {
    "segments": "SELECT status,COUNT(*) FROM segments GROUP BY status",
    "ai_segments": "SELECT status,COUNT(*) FROM ai_segments GROUP BY status",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class CountryScanStore:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _init_schema(self):
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS segments (
                segment_id TEXT PRIMARY KEY,
                row_idx INTEGER NOT NULL,
                col_idx INTEGER NOT NULL,
                west REAL NOT NULL,
                south REAL NOT NULL,
                east REAL NOT NULL,
                north REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                feature_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                started_utc TEXT,
                updated_utc TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_segments_status ON segments(status, segment_id);
            CREATE TABLE IF NOT EXISTS ai_segments (
                segment_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                tiles INTEGER NOT NULL DEFAULT 0,
                detections INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                started_utc TEXT,
                updated_utc TEXT NOT NULL,
                FOREIGN KEY(segment_id) REFERENCES segments(segment_id)
            );
            CREATE INDEX IF NOT EXISTS idx_ai_segments_status ON ai_segments(status, segment_id);
            CREATE TABLE IF NOT EXISTS seen_assets (
                asset_key TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                first_segment_id TEXT NOT NULL,
                created_utc TEXT NOT NULL
            );
            """
        )
        self.set_meta("schema_version", SCHEMA_VERSION, overwrite=False)
        self.conn.commit()

    def set_meta(self, key, value, overwrite=True):
        encoded = json.dumps(value, separators=(",", ":"), sort_keys=True)
        if overwrite:
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, encoded),
            )
        else:
            self.conn.execute("INSERT OR IGNORE INTO meta(key,value) VALUES(?,?)", (key, encoded))

    def get_meta(self, key, default=None):
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row[0])
        except (TypeError, ValueError):
            return row[0]

    def initialize(self, metadata, segments):
        """Create a scan if empty, otherwise verify country/segment geometry.

        The stable signature deliberately excludes AI model settings. This allows
        a user to finish the OSM country crawl, register a model later, and resume
        the independent AI queue in the same workspace.
        """
        existing = self.get_meta("signature")
        signature = metadata.get("signature")
        if not signature:
            raise ValueError("metadata.signature is required")
        if existing and existing != signature:
            raise ValueError(
                "This workspace belongs to a different country/area or segment geometry. "
                "Choose a new workspace folder, or resume using the original area and segment size."
            )
        if not existing:
            for k, v in metadata.items():
                self.set_meta(k, v)
            now = utc_now()
            rows = [
                (s["segment_id"], int(s["row_idx"]), int(s["col_idx"]),
                 float(s["west"]), float(s["south"]), float(s["east"]), float(s["north"]), now)
                for s in segments
            ]
            self.conn.executemany(
                """INSERT OR IGNORE INTO segments(
                       segment_id,row_idx,col_idx,west,south,east,north,status,updated_utc
                   ) VALUES(?,?,?,?,?,?,?,'pending',?)""",
                rows,
            )
            self.conn.executemany(
                "INSERT OR IGNORE INTO ai_segments(segment_id,status,updated_utc) VALUES(?,'pending',?)",
                [(r[0], now) for r in rows],
            )
            self.set_meta("created_utc", now)
            self.conn.commit()
        return self.summary()

    def reset_interrupted(self):
        now = utc_now()
        a = self.conn.execute(
            "UPDATE segments SET status='retry', last_error='Previous run was interrupted', updated_utc=? "
            "WHERE status='running'",
            (now,),
        ).rowcount
        b = self.conn.execute(
            "UPDATE ai_segments SET status='retry', last_error='Previous run was interrupted', updated_utc=? "
            "WHERE status='running'",
            (now,),
        ).rowcount
        self.conn.commit()
        return a + b

    @staticmethod
    def _queue(conn, sql, limit, max_attempts):
        """Run one of the fixed queue statements above."""
        args = [int(max_attempts)]
        if int(limit) > 0:
            sql = sql + _LIMIT_CLAUSE
            args.append(int(limit))
        return [dict(r) for r in conn.execute(sql, args).fetchall()]

    def next_segments(self, limit=0, retry_failed=True, max_attempts=5):
        sql = _OSM_QUEUE_WITH_FAILED_SQL if retry_failed else _OSM_QUEUE_SQL
        return self._queue(self.conn, sql, limit, max_attempts)

    def next_ai_segments(self, limit=0, retry_failed=True, max_attempts=5):
        sql = _AI_QUEUE_WITH_FAILED_SQL if retry_failed else _AI_QUEUE_SQL
        return self._queue(self.conn, sql, limit, max_attempts)

    def mark_running(self, segment_id):
        now = utc_now()
        self.conn.execute(
            "UPDATE segments SET status='running', attempts=attempts+1, started_utc=?, updated_utc=?, last_error=NULL "
            "WHERE segment_id=?",
            (now, now, segment_id),
        )
        self.conn.commit()

    def mark_done(self, segment_id, feature_count=0):
        now = utc_now()
        self.conn.execute(
            "UPDATE segments SET status='done', feature_count=?, last_error=NULL, updated_utc=? WHERE segment_id=?",
            (int(feature_count), now, segment_id),
        )
        self.conn.commit()

    def mark_retry(self, segment_id, error, terminal=False):
        now = utc_now()
        self.conn.execute(
            "UPDATE segments SET status=?, last_error=?, updated_utc=? WHERE segment_id=?",
            ("failed" if terminal else "retry", str(error)[:2000], now, segment_id),
        )
        self.conn.commit()

    def mark_ai_running(self, segment_id):
        now = utc_now()
        self.conn.execute(
            "UPDATE ai_segments SET status='running', attempts=attempts+1, started_utc=?, updated_utc=?, last_error=NULL "
            "WHERE segment_id=?",
            (now, now, segment_id),
        )
        self.conn.commit()

    def mark_ai_done(self, segment_id, tiles=0, detections=0):
        now = utc_now()
        self.conn.execute(
            "UPDATE ai_segments SET status='done', tiles=?, detections=?, last_error=NULL, updated_utc=? WHERE segment_id=?",
            (int(tiles), int(detections), now, segment_id),
        )
        self.conn.commit()

    def mark_ai_retry(self, segment_id, error, terminal=False):
        now = utc_now()
        self.conn.execute(
            "UPDATE ai_segments SET status=?, last_error=?, updated_utc=? WHERE segment_id=?",
            ("failed" if terminal else "retry", str(error)[:2000], now, segment_id),
        )
        self.conn.commit()

    def asset_seen(self, asset_key):
        return self.conn.execute("SELECT 1 FROM seen_assets WHERE asset_key=?", (asset_key,)).fetchone() is not None

    def register_asset(self, asset_key, kind, segment_id):
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO seen_assets(asset_key,kind,first_segment_id,created_utc) VALUES(?,?,?,?)",
            (asset_key, kind, segment_id, utc_now()),
        )
        self.conn.commit()
        return cur.rowcount == 1

    def rows(self):
        return [dict(r) for r in self.conn.execute(
            """SELECT s.*, a.status AS ai_status, a.attempts AS ai_attempts,
                      a.tiles AS ai_tiles, a.detections AS ai_detections,
                      a.last_error AS ai_last_error
               FROM segments s JOIN ai_segments a USING(segment_id)
               ORDER BY s.segment_id"""
        )]

    @staticmethod
    def _counts(conn, table):
        return {r[0]: int(r[1]) for r in conn.execute(_STATUS_COUNT_SQL[table])}

    def summary(self):
        total = int(self.conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0])
        osm = self._counts(self.conn, "segments")
        ai = self._counts(self.conn, "ai_segments")
        osm_done = int(osm.get("done", 0))
        ai_done = int(ai.get("done", 0))
        assets = int(self.conn.execute("SELECT COUNT(*) FROM seen_assets WHERE kind != 'ai_substation'").fetchone()[0])
        ai_assets = int(self.conn.execute("SELECT COUNT(*) FROM seen_assets WHERE kind = 'ai_substation'").fetchone()[0])
        tiles = int(self.conn.execute("SELECT COALESCE(SUM(tiles),0) FROM ai_segments WHERE status='done'").fetchone()[0] or 0)
        return {
            "total": total,
            "osm_done": osm_done,
            "osm_pending": int(osm.get("pending", 0)),
            "osm_retry": int(osm.get("retry", 0)),
            "osm_failed": int(osm.get("failed", 0)),
            "osm_progress_pct": round(100.0 * osm_done / total, 2) if total else 0.0,
            "ai_done": ai_done,
            "ai_pending": int(ai.get("pending", 0)),
            "ai_retry": int(ai.get("retry", 0)),
            "ai_failed": int(ai.get("failed", 0)),
            "ai_progress_pct": round(100.0 * ai_done / total, 2) if total else 0.0,
            "assets": assets,
            "ai_detections": ai_assets,
            "ai_tiles": tiles,
        }

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from .species_catalog import MASTER_SPECIES


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class Database:
    """Thread-safe-by-connection SQLite repository for all FinFrame projects."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    deployment_id TEXT NOT NULL DEFAULT '',
                    site TEXT NOT NULL DEFAULT '',
                    observer TEXT NOT NULL DEFAULT '',
                    survey_date TEXT NOT NULL DEFAULT '',
                    depth TEXT NOT NULL DEFAULT '',
                    notes TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS videos (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    media_type TEXT NOT NULL DEFAULT 'video',
                    duration REAL NOT NULL DEFAULT 0,
                    width INTEGER NOT NULL DEFAULT 0,
                    height INTEGER NOT NULL DEFAULT 0,
                    fps REAL NOT NULL DEFAULT 25,
                    frame_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    UNIQUE(project_id, path)
                );

                CREATE TABLE IF NOT EXISTS species (
                    id TEXT PRIMARY KEY,
                    common_name TEXT NOT NULL,
                    scientific_name TEXT NOT NULL DEFAULT '',
                    code TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    color TEXT NOT NULL DEFAULT '#ff8465',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS frames (
                    id TEXT PRIMARY KEY,
                    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
                    frame_number INTEGER NOT NULL,
                    time_seconds REAL NOT NULL,
                    reviewed INTEGER NOT NULL DEFAULT 0,
                    note TEXT NOT NULL DEFAULT '',
                    image_path TEXT NOT NULL DEFAULT '',
                    training_selected INTEGER NOT NULL DEFAULT 0,
                    training_reason TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    UNIQUE(video_id, frame_number)
                );

                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL UNIQUE,
                    weights_path TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('candidate','active','rejected','failed')),
                    map50_95 REAL,
                    verified_examples INTEGER NOT NULL DEFAULT 0,
                    training_run_id TEXT,
                    created_at TEXT NOT NULL,
                    activated_at TEXT,
                    notes TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS annotations (
                    id TEXT PRIMARY KEY,
                    frame_id TEXT NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
                    species_id TEXT NOT NULL REFERENCES species(id),
                    track_id TEXT NOT NULL,
                    x REAL NOT NULL CHECK(x >= 0 AND x <= 1),
                    y REAL NOT NULL CHECK(y >= 0 AND y <= 1),
                    width REAL NOT NULL CHECK(width > 0 AND width <= 1),
                    height REAL NOT NULL CHECK(height > 0 AND height <= 1),
                    life_stage TEXT NOT NULL DEFAULT 'Unknown',
                    activity TEXT NOT NULL DEFAULT 'Passing',
                    uncertain INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL CHECK(status IN ('pending','verified','rejected')),
                    source TEXT NOT NULL CHECK(source IN ('manual','ai','ai_verified','ai_corrected','tracker','tracker_verified','tracker_corrected')),
                    confidence REAL,
                    model_id TEXT REFERENCES models(id),
                    created_by TEXT NOT NULL DEFAULT '',
                    modified INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS training_runs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed','skipped')),
                    trigger_reason TEXT NOT NULL,
                    verified_examples INTEGER NOT NULL,
                    dataset_path TEXT NOT NULL DEFAULT '',
                    base_model TEXT NOT NULL DEFAULT '',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    requested_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_frames_video_number ON frames(video_id, frame_number);
                CREATE INDEX IF NOT EXISTS idx_annotations_frame_status ON annotations(frame_id, status);
                CREATE INDEX IF NOT EXISTS idx_annotations_species_status ON annotations(species_id, status);
                """
            )
            video_columns = {row[1] for row in db.execute("PRAGMA table_info(videos)")}
            if "media_type" not in video_columns:
                db.execute("ALTER TABLE videos ADD COLUMN media_type TEXT NOT NULL DEFAULT 'video'")
            frame_columns = {row[1] for row in db.execute("PRAGMA table_info(frames)")}
            training_selection_added = "training_selected" not in frame_columns
            if "image_path" not in frame_columns:
                db.execute("ALTER TABLE frames ADD COLUMN image_path TEXT NOT NULL DEFAULT ''")
            if "training_selected" not in frame_columns:
                db.execute("ALTER TABLE frames ADD COLUMN training_selected INTEGER NOT NULL DEFAULT 0")
            if "training_reason" not in frame_columns:
                db.execute("ALTER TABLE frames ADD COLUMN training_reason TEXT NOT NULL DEFAULT ''")
            if training_selection_added:
                db.execute(
                    """UPDATE frames SET reviewed=1,updated_at=?
                       WHERE EXISTS (SELECT 1 FROM annotations a WHERE a.frame_id=frames.id AND a.status='verified')""",
                    (utc_now(),),
                )
                selected = db.execute(
                    """UPDATE frames SET training_selected=1,training_reason='legacy_verified'
                       WHERE EXISTS (
                           SELECT 1 FROM annotations a WHERE a.frame_id=frames.id AND a.status='verified'
                           AND (a.source='manual' OR a.source LIKE '%_corrected')
                       )"""
                ).rowcount
                if selected:
                    self._bump_training_revision(db)
            for common, scientific, code, color in MASTER_SPECIES:
                db.execute(
                    "INSERT OR IGNORE INTO species(id, common_name, scientific_name, code, color, created_at) VALUES(?,?,?,?,?,?)",
                    (new_id("sp"), common, scientific, code, color, utc_now()),
                )

    @staticmethod
    def _rows(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
        return [dict(row) for row in rows]

    @staticmethod
    def _bump_dataset_revision(db: sqlite3.Connection) -> None:
        db.execute(
            """INSERT INTO settings(key,value) VALUES('verified_dataset_revision','1')
               ON CONFLICT(key) DO UPDATE SET value=CAST(CAST(value AS INTEGER)+1 AS TEXT)"""
        )

    @staticmethod
    def _bump_training_revision(db: sqlite3.Connection) -> None:
        db.execute(
            """INSERT INTO settings(key,value) VALUES('training_dataset_revision','1')
               ON CONFLICT(key) DO UPDATE SET value=CAST(CAST(value AS INTEGER)+1 AS TEXT)"""
        )

    @classmethod
    def _invalidate_frame_review(cls, db: sqlite3.Connection, frame_id: str) -> None:
        frame = db.execute(
            "SELECT reviewed,training_selected FROM frames WHERE id=?",
            (frame_id,),
        ).fetchone()
        if frame is None or not frame["reviewed"]:
            return
        db.execute(
            "UPDATE frames SET reviewed=0,training_selected=0,training_reason='',updated_at=? WHERE id=?",
            (utc_now(), frame_id),
        )

    def create_project(self, name: str, **metadata: Any) -> dict[str, Any]:
        project_id, now = new_id("project"), utc_now()
        with self.connect() as db:
            db.execute(
                """INSERT INTO projects(id,name,deployment_id,site,observer,survey_date,depth,notes,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    project_id,
                    name.strip() or "Untitled survey",
                    metadata.get("deployment_id", ""),
                    metadata.get("site", ""),
                    metadata.get("observer", ""),
                    metadata.get("survey_date", ""),
                    str(metadata.get("depth", "")),
                    metadata.get("notes", ""),
                    now,
                    now,
                ),
            )
        return self.get_project(project_id)

    def get_project(self, project_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown project {project_id}")
        return dict(row)

    def list_projects(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            return self._rows(db.execute("SELECT * FROM projects ORDER BY updated_at DESC"))

    def update_project(self, project_id: str, **metadata: Any) -> None:
        allowed = {"name", "deployment_id", "site", "observer", "survey_date", "depth", "notes"}
        values = {key: value for key, value in metadata.items() if key in allowed}
        if not values:
            return
        values["updated_at"] = utc_now()
        assignments = ",".join(f"{key}=?" for key in values)
        with self.connect() as db:
            db.execute(f"UPDATE projects SET {assignments} WHERE id=?", (*values.values(), project_id))

    def add_video(
        self,
        project_id: str,
        path: str | Path,
        *,
        duration: float,
        width: int,
        height: int,
        fps: float,
        frame_count: int,
        media_type: str = "video",
    ) -> dict[str, Any]:
        if media_type not in {"video", "image"}:
            raise ValueError("media_type must be video or image")
        resolved = str(Path(path).expanduser().resolve())
        with self.connect() as db:
            existing = db.execute("SELECT * FROM videos WHERE project_id=? AND path=?", (project_id, resolved)).fetchone()
            if existing:
                return dict(existing)
            video_id = new_id("video")
            db.execute(
                """INSERT INTO videos(id,project_id,path,file_name,media_type,duration,width,height,fps,frame_count,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (video_id, project_id, resolved, Path(resolved).name, media_type, duration, width, height, fps, frame_count, utc_now()),
            )
        return self.get_video(video_id)

    def get_video(self, video_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown video {video_id}")
        return dict(row)

    def list_videos(self, project_id: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as db:
            if project_id:
                rows = db.execute("SELECT * FROM videos WHERE project_id=? ORDER BY created_at", (project_id,))
            else:
                rows = db.execute("SELECT * FROM videos ORDER BY created_at")
            return self._rows(rows)

    def relink_video(self, video_id: str, path: str | Path, *, duration: float, width: int, height: int, fps: float, frame_count: int) -> dict[str, Any]:
        resolved = str(Path(path).expanduser().resolve())
        with self.connect() as db:
            db.execute(
                """UPDATE videos SET path=?,file_name=?,duration=?,width=?,height=?,fps=?,frame_count=? WHERE id=?""",
                (resolved, Path(resolved).name, duration, width, height, fps, frame_count, video_id),
            )
        return self.get_video(video_id)

    def list_species(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            return self._rows(db.execute("SELECT * FROM species ORDER BY common_name COLLATE NOCASE"))

    def get_species(self, species_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM species WHERE id=?", (species_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown species {species_id}")
        return dict(row)

    def species_by_code(self, code: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM species WHERE code=? COLLATE NOCASE", (code,)).fetchone()
        return dict(row) if row else None

    def add_species(self, common_name: str, scientific_name: str, code: str, color: str) -> dict[str, Any]:
        species_id = new_id("sp")
        with self.connect() as db:
            db.execute(
                "INSERT INTO species(id,common_name,scientific_name,code,color,created_at) VALUES(?,?,?,?,?,?)",
                (species_id, common_name.strip(), scientific_name.strip(), code.strip().upper(), color, utc_now()),
            )
        return self.get_species(species_id)

    def ensure_frame(self, video_id: str, frame_number: int, time_seconds: float) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM frames WHERE video_id=? AND frame_number=?", (video_id, frame_number)).fetchone()
            if row:
                return dict(row)
            frame_id = new_id("frame")
            db.execute(
                "INSERT INTO frames(id,video_id,frame_number,time_seconds,updated_at) VALUES(?,?,?,?,?)",
                (frame_id, video_id, frame_number, time_seconds, utc_now()),
            )
        return self.get_frame(video_id, frame_number)

    def get_frame(self, video_id: str, frame_number: int) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM frames WHERE video_id=? AND frame_number=?", (video_id, frame_number)).fetchone()
        if row is None:
            raise KeyError(f"Unknown frame {video_id}:{frame_number}")
        return dict(row)

    def frames_for_video(self, video_id: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            return self._rows(db.execute(
                "SELECT * FROM frames WHERE video_id=? ORDER BY frame_number",
                (video_id,),
            ))

    def add_annotation(
        self,
        *,
        video_id: str,
        frame_number: int,
        time_seconds: float,
        species_id: str,
        track_id: str,
        box: tuple[float, float, float, float],
        status: str = "verified",
        source: str = "manual",
        confidence: float | None = None,
        model_id: str | None = None,
        created_by: str = "",
        life_stage: str = "Unknown",
        activity: str = "Passing",
        uncertain: bool = False,
    ) -> dict[str, Any]:
        if status not in {"pending", "verified", "rejected"}:
            raise ValueError("Invalid annotation status")
        x, y, width, height = box
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1 and x + width <= 1.000001 and y + height <= 1.000001):
            raise ValueError("Bounding box must be normalised inside the frame")
        frame = self.ensure_frame(video_id, frame_number, time_seconds)
        annotation_id, now = new_id("ann"), utc_now()
        with self.connect() as db:
            self._invalidate_frame_review(db, frame["id"])
            db.execute(
                """INSERT INTO annotations(
                       id,frame_id,species_id,track_id,x,y,width,height,life_stage,activity,uncertain,status,source,
                       confidence,model_id,created_by,modified,created_at,updated_at
                   ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    annotation_id, frame["id"], species_id, track_id, x, y, width, height, life_stage, activity,
                    int(uncertain), status, source, confidence, model_id, created_by, 0, now, now,
                ),
            )
            if status == "verified":
                self._bump_dataset_revision(db)
        return self.get_annotation(annotation_id)

    def add_pending_tracker_annotations(
        self,
        video_id: str,
        frame_number: int,
        time_seconds: float,
        proposals: Iterable[dict[str, Any]],
        *,
        created_by: str = "",
    ) -> int:
        """Insert one playback frame's tracker proposals in a single transaction."""
        frame = self.ensure_frame(video_id, frame_number, time_seconds)
        candidates = list(proposals)
        if not candidates:
            return 0
        now = utc_now()
        rows = []
        with self.connect() as db:
            existing_tracks = {
                str(row[0])
                for row in db.execute("SELECT track_id FROM annotations WHERE frame_id=?", (frame["id"],))
            }
            for proposal in candidates:
                track_id = str(proposal["track_id"])
                if track_id in existing_tracks:
                    continue
                x, y, width, height = map(float, proposal["box"])
                if not (
                    0 <= x <= 1
                    and 0 <= y <= 1
                    and 0 < width <= 1
                    and 0 < height <= 1
                    and x + width <= 1.000001
                    and y + height <= 1.000001
                ):
                    continue
                rows.append((
                    new_id("ann"), frame["id"], proposal["species_id"], track_id,
                    x, y, width, height,
                    proposal.get("life_stage", "Unknown"), proposal.get("activity", "Unknown"),
                    int(bool(proposal.get("uncertain", False))), "pending", "tracker",
                    proposal.get("confidence"), proposal.get("model_id"), created_by, 0, now, now,
                ))
                existing_tracks.add(track_id)
            if rows:
                self._invalidate_frame_review(db, frame["id"])
                db.executemany(
                    """INSERT INTO annotations(
                           id,frame_id,species_id,track_id,x,y,width,height,life_stage,activity,uncertain,status,source,
                           confidence,model_id,created_by,modified,created_at,updated_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    rows,
                )
        return len(rows)

    def get_annotation(self, annotation_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute(
                """SELECT a.*,s.common_name,s.scientific_name,s.code,s.color,f.video_id,f.frame_number,f.time_seconds
                   FROM annotations a JOIN species s ON s.id=a.species_id JOIN frames f ON f.id=a.frame_id
                   WHERE a.id=?""",
                (annotation_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown annotation {annotation_id}")
        return dict(row)

    def update_annotation(self, annotation_id: str, **changes: Any) -> dict[str, Any]:
        mapping = {"species_id", "track_id", "x", "y", "width", "height", "life_stage", "activity", "uncertain"}
        current = self.get_annotation(annotation_id)
        values = {}
        for key, value in changes.items():
            if key not in mapping:
                continue
            comparable = int(bool(value)) if key == "uncertain" else value
            if current[key] != comparable:
                values[key] = comparable
        if not values:
            return current
        values["modified"] = 1
        values["updated_at"] = utc_now()
        assignments = ",".join(f"{key}=?" for key in values)
        with self.connect() as db:
            self._invalidate_frame_review(db, current["frame_id"])
            db.execute(f"UPDATE annotations SET {assignments} WHERE id=?", (*values.values(), annotation_id))
            if current["status"] == "verified":
                self._bump_dataset_revision(db)
        return self.get_annotation(annotation_id)

    def review_annotation(self, annotation_id: str, decision: str) -> dict[str, Any]:
        if decision not in {"approve", "reject"}:
            raise ValueError("Decision must be approve or reject")
        annotation = self.get_annotation(annotation_id)
        if decision == "reject":
            status, source = "rejected", annotation["source"]
        else:
            status = "verified"
            if annotation["source"] in {"ai", "tracker"}:
                source = f"{annotation['source']}_{'corrected' if annotation['modified'] else 'verified'}"
            else:
                source = annotation["source"]
        with self.connect() as db:
            self._invalidate_frame_review(db, annotation["frame_id"])
            db.execute("UPDATE annotations SET status=?,source=?,updated_at=? WHERE id=?", (status, source, utc_now(), annotation_id))
            if annotation["status"] != status and "verified" in {annotation["status"], status}:
                self._bump_dataset_revision(db)
        return self.get_annotation(annotation_id)

    def delete_annotation(self, annotation_id: str) -> None:
        annotation = self.get_annotation(annotation_id)
        with self.connect() as db:
            self._invalidate_frame_review(db, annotation["frame_id"])
            db.execute("DELETE FROM annotations WHERE id=?", (annotation_id,))
            if annotation["status"] == "verified":
                self._bump_dataset_revision(db)

    def video_annotation_stats(self, video_id: str) -> dict[str, int]:
        with self.connect() as db:
            row = db.execute(
                """SELECT COUNT(*) AS total,
                          SUM(CASE WHEN a.status='verified' THEN 1 ELSE 0 END) AS verified,
                          SUM(CASE WHEN a.status='pending' THEN 1 ELSE 0 END) AS pending,
                          SUM(CASE WHEN a.status='rejected' THEN 1 ELSE 0 END) AS rejected,
                          COUNT(DISTINCT a.frame_id) AS frames
                   FROM annotations a JOIN frames f ON f.id=a.frame_id
                   WHERE f.video_id=?""",
                (video_id,),
            ).fetchone()
        return {key: int(row[key] or 0) for key in ("total", "verified", "pending", "rejected", "frames")}

    def clear_video_annotations(self, video_id: str) -> dict[str, int]:
        """Delete every box for one video and invalidate only frames that contained boxes."""
        with self.connect() as db:
            row = db.execute(
                """SELECT COUNT(*) AS total,
                          SUM(CASE WHEN a.status='verified' THEN 1 ELSE 0 END) AS verified,
                          SUM(CASE WHEN a.status='pending' THEN 1 ELSE 0 END) AS pending,
                          SUM(CASE WHEN a.status='rejected' THEN 1 ELSE 0 END) AS rejected,
                          COUNT(DISTINCT a.frame_id) AS frames,
                          COUNT(DISTINCT CASE WHEN f.training_selected=1 THEN f.id END) AS training_frames
                   FROM annotations a JOIN frames f ON f.id=a.frame_id
                   WHERE f.video_id=?""",
                (video_id,),
            ).fetchone()
            result = {
                key: int(row[key] or 0)
                for key in ("total", "verified", "pending", "rejected", "frames", "training_frames")
            }
            if not result["total"]:
                return result
            db.execute(
                """UPDATE frames SET reviewed=0,training_selected=0,training_reason='',updated_at=?
                   WHERE video_id=? AND EXISTS (
                       SELECT 1 FROM annotations a WHERE a.frame_id=frames.id
                   )""",
                (utc_now(), video_id),
            )
            db.execute(
                "DELETE FROM annotations WHERE frame_id IN (SELECT id FROM frames WHERE video_id=?)",
                (video_id,),
            )
            if result["verified"]:
                self._bump_dataset_revision(db)
            if result["training_frames"]:
                self._bump_training_revision(db)
        return result

    def delete_pending_proposals(
        self,
        video_id: str,
        *,
        source: str,
        frame_number: int | None = None,
        model_only: bool = False,
    ) -> int:
        query = "DELETE FROM annotations WHERE status='pending' AND source=?"
        if model_only:
            query += " AND model_id IS NOT NULL"
        query += """ AND frame_id IN (
                       SELECT id FROM frames WHERE video_id=?"""
        params: list[Any] = [source, video_id]
        if frame_number is not None:
            query += " AND frame_number=?"
            params.append(frame_number)
        query += ")"
        with self.connect() as db:
            cursor = db.execute(query, params)
            return cursor.rowcount

    def update_frame(
        self,
        video_id: str,
        frame_number: int,
        *,
        reviewed: bool | None = None,
        note: str | None = None,
        image_path: str | Path | None = None,
        training_selected: bool | None = None,
        training_reason: str | None = None,
    ) -> None:
        values: dict[str, Any] = {}
        if reviewed is not None:
            values["reviewed"] = int(reviewed)
        if note is not None:
            values["note"] = note
        if image_path is not None:
            values["image_path"] = str(Path(image_path).expanduser().resolve()) if str(image_path) else ""
        if training_selected is not None:
            values["training_selected"] = int(training_selected)
        if training_reason is not None:
            values["training_reason"] = training_reason
        if not values:
            return
        values["updated_at"] = utc_now()
        assignments = ",".join(f"{key}=?" for key in values)
        with self.connect() as db:
            db.execute(
                f"UPDATE frames SET {assignments} WHERE video_id=? AND frame_number=?",
                (*values.values(), video_id, frame_number),
            )

    def set_frame_reviewed(
        self,
        video_id: str,
        frame_number: int,
        reviewed: bool,
        *,
        sample_interval_seconds: float | None = None,
    ) -> dict[str, Any]:
        video = self.get_video(video_id)
        time_seconds = frame_number / max(float(video["fps"]), 0.001) if video.get("media_type") == "video" else 0.0
        frame = self.ensure_frame(video_id, frame_number, time_seconds)
        interval = float(
            sample_interval_seconds
            if sample_interval_seconds is not None
            else self.get_setting("training_sample_interval_seconds", 1.0)
        )
        with self.connect() as db:
            current = db.execute("SELECT * FROM frames WHERE id=?", (frame["id"],)).fetchone()
            selected, reason = 0, ""
            if reviewed:
                pending = db.execute(
                    "SELECT COUNT(*) FROM annotations WHERE frame_id=? AND status='pending'",
                    (frame["id"],),
                ).fetchone()[0]
                if pending:
                    raise ValueError("Approve, correct or reject every pending box before marking the frame complete")
                sources = [row[0] for row in db.execute(
                    "SELECT source FROM annotations WHERE frame_id=? AND status='verified'",
                    (frame["id"],),
                )]
                strong_label = any(source == "manual" or source.endswith("_corrected") for source in sources)
                if strong_label:
                    selected, reason = 1, "manual_or_corrected"
                else:
                    closest = db.execute(
                        """SELECT MIN(ABS(time_seconds-?)) FROM frames
                           WHERE video_id=? AND id!=? AND reviewed=1 AND training_selected=1""",
                        (time_seconds, video_id, frame["id"]),
                    ).fetchone()[0]
                    if closest is None or float(closest) >= max(0.0, interval):
                        selected, reason = 1, "temporal_sample"
                    else:
                        reason = "near_duplicate"
            changed = (
                int(current["reviewed"]) != int(reviewed)
                or int(current["training_selected"]) != selected
                or current["training_reason"] != reason
            )
            if changed:
                old_eligible = bool(current["reviewed"] and current["training_selected"])
                new_eligible = bool(reviewed and selected)
                db.execute(
                    """UPDATE frames SET reviewed=?,training_selected=?,training_reason=?,updated_at=?
                       WHERE id=?""",
                    (int(reviewed), selected, reason, utc_now(), frame["id"]),
                )
                if old_eligible or new_eligible:
                    self._bump_training_revision(db)
        return self.get_frame(video_id, frame_number)

    def pending_annotations_in_range(self, video_id: str, start_frame: int, end_frame: int) -> list[dict[str, Any]]:
        first, last = sorted((int(start_frame), int(end_frame)))
        with self.connect() as db:
            return self._rows(db.execute(
                """SELECT a.*,f.frame_number FROM annotations a JOIN frames f ON f.id=a.frame_id
                   WHERE f.video_id=? AND f.frame_number BETWEEN ? AND ? AND a.status='pending'
                   ORDER BY f.frame_number,a.created_at""",
                (video_id, first, last),
            ))

    def annotated_frame_numbers_in_range(self, video_id: str, start_frame: int, end_frame: int) -> list[int]:
        first, last = sorted((int(start_frame), int(end_frame)))
        with self.connect() as db:
            return [int(row[0]) for row in db.execute(
                """SELECT DISTINCT f.frame_number FROM frames f JOIN annotations a ON a.frame_id=f.id
                   WHERE f.video_id=? AND f.frame_number BETWEEN ? AND ? ORDER BY f.frame_number""",
                (video_id, first, last),
            )]

    def annotations_for_frame(self, video_id: str, frame_number: int, *, include_rejected: bool = False) -> list[dict[str, Any]]:
        query = """SELECT a.*,s.common_name,s.scientific_name,s.code,s.color,
                          f.video_id,f.frame_number,f.time_seconds
                   FROM frames f JOIN annotations a ON a.frame_id=f.id JOIN species s ON s.id=a.species_id
                   WHERE f.video_id=? AND f.frame_number=?"""
        params: list[Any] = [video_id, frame_number]
        if not include_rejected:
            query += " AND a.status != 'rejected'"
        query += " ORDER BY a.created_at"
        with self.connect() as db:
            return self._rows(db.execute(query, params))

    def next_pending_frame(self, video_id: str, after_frame: int = -1) -> int | None:
        with self.connect() as db:
            row = db.execute(
                """SELECT MIN(f.frame_number) FROM frames f JOIN annotations a ON a.frame_id=f.id
                   WHERE f.video_id=? AND a.status='pending' AND f.frame_number>?""",
                (video_id, after_frame),
            ).fetchone()
            frame_number = row[0]
            if frame_number is None:
                frame_number = db.execute(
                    """SELECT MIN(f.frame_number) FROM frames f JOIN annotations a ON a.frame_id=f.id
                       WHERE f.video_id=? AND a.status='pending'""",
                    (video_id,),
                ).fetchone()[0]
        return int(frame_number) if frame_number is not None else None

    def next_track_id(self, video_id: str, species_id: str) -> str:
        species = self.get_species(species_id)
        with self.connect() as db:
            tracks = [row[0] for row in db.execute(
                """SELECT a.track_id FROM annotations a JOIN frames f ON f.id=a.frame_id
                   WHERE f.video_id=? AND a.species_id=?""",
                (video_id, species_id),
            )]
        highest = 0
        for track in tracks:
            try:
                highest = max(highest, int(str(track).rsplit("-", 1)[-1]))
            except ValueError:
                continue
        return f"{species['code']}-{highest + 1:03d}"

    def verified_annotations(
        self,
        video_id: str | None = None,
        *,
        reviewed_only: bool = True,
        training_only: bool = False,
    ) -> list[dict[str, Any]]:
        query = """SELECT a.*,s.common_name,s.scientific_name,s.code,s.color,
                          f.video_id,f.frame_number,f.time_seconds,f.reviewed,f.note,f.image_path,
                          f.training_selected,f.training_reason,
                          v.path AS video_path,v.file_name,v.media_type,v.width AS video_width,v.height AS video_height,v.fps,
                          p.id AS project_id,p.name AS project_name,p.deployment_id,p.site,p.observer
                   FROM annotations a
                   JOIN species s ON s.id=a.species_id
                   JOIN frames f ON f.id=a.frame_id
                   JOIN videos v ON v.id=f.video_id
                   JOIN projects p ON p.id=v.project_id
                   WHERE a.status='verified'"""
        params: list[Any] = []
        if reviewed_only:
            query += " AND f.reviewed=1"
        if training_only:
            query += " AND f.training_selected=1"
        if video_id:
            query += " AND f.video_id=?"
            params.append(video_id)
        query += " ORDER BY v.id,f.frame_number,a.created_at"
        with self.connect() as db:
            return self._rows(db.execute(query, params))

    def training_frames(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            return self._rows(db.execute(
                """SELECT f.*,v.path AS video_path,v.file_name,v.media_type,
                          v.width AS video_width,v.height AS video_height,v.fps,
                          v.project_id,p.deployment_id,p.name AS project_name,p.site,p.observer
                   FROM frames f JOIN videos v ON v.id=f.video_id JOIN projects p ON p.id=v.project_id
                   WHERE f.reviewed=1 AND f.training_selected=1
                   ORDER BY v.id,f.frame_number"""
            ))

    def annotated_frames(self, *, verified_only: bool = True) -> list[dict[str, Any]]:
        condition = "a.status='verified'" if verified_only else "a.status!='rejected'"
        with self.connect() as db:
            return self._rows(db.execute(
                f"""SELECT DISTINCT f.*,v.path AS video_path,v.file_name,v.media_type,v.width AS video_width,v.height AS video_height,v.fps,
                            v.project_id,p.deployment_id,p.name AS project_name
                     FROM frames f JOIN annotations a ON a.frame_id=f.id JOIN videos v ON v.id=f.video_id JOIN projects p ON p.id=v.project_id
                     WHERE {condition} ORDER BY v.id,f.frame_number"""
            ))

    def frame_counts(self, video_id: str, frame_number: int) -> list[dict[str, Any]]:
        with self.connect() as db:
            return self._rows(db.execute(
                """SELECT s.id AS species_id,s.common_name,s.scientific_name,s.code,s.color,COUNT(*) AS count
                   FROM frames f JOIN annotations a ON a.frame_id=f.id JOIN species s ON s.id=a.species_id
                   WHERE f.video_id=? AND f.frame_number=? AND a.status='verified'
                   GROUP BY s.id ORDER BY count DESC,s.common_name""",
                (video_id, frame_number),
            ))

    def maxn_summary(self, video_id: str, *, reviewed_only: bool = True) -> list[dict[str, Any]]:
        review_filter = "AND f.reviewed=1" if reviewed_only else ""
        with self.connect() as db:
            return self._rows(db.execute(
                f"""WITH counts AS (
                       SELECT f.id AS frame_id,f.frame_number,f.time_seconds,a.species_id,COUNT(*) AS fish_count
                       FROM frames f JOIN annotations a ON a.frame_id=f.id
                       WHERE f.video_id=? {review_filter} AND a.status='verified'
                       GROUP BY f.id,a.species_id
                   ), ranked AS (
                       SELECT counts.*,ROW_NUMBER() OVER(PARTITION BY species_id ORDER BY fish_count DESC,frame_number ASC) AS rank
                       FROM counts
                   )
                   SELECT s.id AS species_id,s.common_name,s.scientific_name,s.code,s.color,
                          ranked.fish_count AS maxn,ranked.frame_number,ranked.time_seconds
                   FROM ranked JOIN species s ON s.id=ranked.species_id WHERE ranked.rank=1
                   ORDER BY maxn DESC,s.common_name""",
                (video_id,),
            ))

    def training_stats(self) -> dict[str, int]:
        with self.connect() as db:
            row = db.execute(
                """SELECT COUNT(a.id) AS examples,COUNT(DISTINCT a.species_id) AS classes,
                          COUNT(DISTINCT f.video_id) AS videos,COUNT(DISTINCT f.id) AS training_frames
                   FROM frames f LEFT JOIN annotations a ON a.frame_id=f.id AND a.status='verified'
                   WHERE f.reviewed=1 AND f.training_selected=1"""
            ).fetchone()
            pending = db.execute("SELECT COUNT(*) FROM annotations WHERE status='pending'").fetchone()[0]
            verified_total = db.execute(
                """SELECT COUNT(*) FROM annotations a JOIN frames f ON f.id=a.frame_id
                   WHERE a.status='verified' AND f.reviewed=1"""
            ).fetchone()[0]
            reviewed_frames = db.execute("SELECT COUNT(*) FROM frames WHERE reviewed=1").fetchone()[0]
        revision = int(self.get_setting("training_dataset_revision", 0) or 0)
        verified_revision = int(self.get_setting("verified_dataset_revision", 0) or 0)
        return {
            "examples": int(row["examples"]),
            "classes": int(row["classes"]),
            "videos": int(row["videos"]),
            "frames": int(row["training_frames"]),
            "verified_total": int(verified_total),
            "reviewed_frames": int(reviewed_frames),
            "pending": int(pending),
            "revision": revision,
            "verified_revision": verified_revision,
        }

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self.connect() as db:
            row = db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return row["value"]

    def set_setting(self, key: str, value: Any) -> None:
        encoded = json.dumps(value)
        with self.connect() as db:
            db.execute(
                "INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, encoded),
            )

    def create_training_run(self, reason: str, verified_examples: int, base_model: str) -> dict[str, Any]:
        run_id = new_id("train")
        with self.connect() as db:
            db.execute(
                """INSERT INTO training_runs(id,status,trigger_reason,verified_examples,base_model,requested_at)
                   VALUES(?,?,?,?,?,?)""",
                (run_id, "queued", reason, verified_examples, base_model, utc_now()),
            )
        return self.get_training_run(run_id)

    def get_training_run(self, run_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM training_runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return dict(row)

    def update_training_run(self, run_id: str, **changes: Any) -> None:
        allowed = {"status", "dataset_path", "metrics_json", "error", "started_at", "completed_at"}
        values = {key: value for key, value in changes.items() if key in allowed}
        if not values:
            return
        assignments = ",".join(f"{key}=?" for key in values)
        with self.connect() as db:
            db.execute(f"UPDATE training_runs SET {assignments} WHERE id=?", (*values.values(), run_id))

    def latest_training_run(self) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM training_runs ORDER BY requested_at DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    def register_model(self, weights_path: str, map50_95: float | None, examples: int, run_id: str, *, status: str = "candidate", notes: str = "") -> dict[str, Any]:
        with self.connect() as db:
            version = db.execute("SELECT COALESCE(MAX(version),0)+1 FROM models").fetchone()[0]
            model_id = new_id("model")
            db.execute(
                """INSERT INTO models(id,version,weights_path,status,map50_95,verified_examples,training_run_id,created_at,notes)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (model_id, version, weights_path, status, map50_95, examples, run_id, utc_now(), notes),
            )
        return self.get_model(model_id)

    def next_model_version(self) -> int:
        with self.connect() as db:
            return int(db.execute("SELECT COALESCE(MAX(version),0)+1 FROM models").fetchone()[0])

    def get_model(self, model_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
        if row is None:
            raise KeyError(model_id)
        return dict(row)

    def activate_model(self, model_id: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE models SET status='candidate',activated_at=NULL WHERE status='active'")
            db.execute("UPDATE models SET status='active',activated_at=? WHERE id=?", (utc_now(), model_id))

    def reject_model(self, model_id: str, notes: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE models SET status='rejected',notes=? WHERE id=?", (notes, model_id))

    def active_model(self) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM models WHERE status='active' ORDER BY version DESC LIMIT 1").fetchone()
        return dict(row) if row else None

    def project_snapshot(self, project_id: str) -> dict[str, Any]:
        project = self.get_project(project_id)
        videos = self.list_videos(project_id)
        snapshot: dict[str, Any] = {"schemaVersion": "2.0", "project": project, "species": self.list_species(), "videos": []}
        with self.connect() as db:
            for video in videos:
                frames = self._rows(db.execute("SELECT * FROM frames WHERE video_id=? ORDER BY frame_number", (video["id"],)))
                for frame in frames:
                    frame["annotations"] = self._rows(db.execute("SELECT * FROM annotations WHERE frame_id=? ORDER BY created_at", (frame["id"],)))
                snapshot["videos"].append({**video, "frames": frames})
        return snapshot

    def import_project_snapshot(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        if snapshot.get("schemaVersion") != "2.0" or not isinstance(snapshot.get("project"), dict) or not isinstance(snapshot.get("videos"), list):
            raise ValueError("This is not a FinFrame Desktop 2.0 project backup")
        source_project = snapshot["project"]
        existing_names = {item["name"] for item in self.list_projects()}
        name = source_project.get("name") or "Imported survey"
        if name in existing_names:
            name = f"{name} (imported)"
        project = self.create_project(
            name,
            deployment_id=source_project.get("deployment_id", ""),
            site=source_project.get("site", ""),
            observer=source_project.get("observer", ""),
            survey_date=source_project.get("survey_date", ""),
            depth=source_project.get("depth", ""),
            notes=source_project.get("notes", ""),
        )
        species_map: dict[str, str] = {}
        for source_species in snapshot.get("species", []):
            target = self.species_by_code(source_species.get("code", ""))
            if target is None:
                target = self.add_species(
                    source_species.get("common_name", source_species.get("code", "Unknown species")),
                    source_species.get("scientific_name", ""),
                    source_species.get("code", new_id("TAXON")[:12]),
                    source_species.get("color", "#ff8465"),
                )
            species_map[source_species.get("id", target["id"])] = target["id"]
        for source_video in snapshot["videos"]:
            video = self.add_video(
                project["id"],
                source_video["path"],
                duration=float(source_video.get("duration", 0)),
                width=int(source_video.get("width", 0)),
                height=int(source_video.get("height", 0)),
                fps=float(source_video.get("fps", 25)),
                frame_count=int(source_video.get("frame_count", 0)),
                media_type=source_video.get("media_type", "video"),
            )
            for source_frame in source_video.get("frames", []):
                frame_number = int(source_frame["frame_number"])
                time_seconds = float(source_frame.get("time_seconds", frame_number / max(float(video["fps"]), 0.001)))
                self.ensure_frame(video["id"], frame_number, time_seconds)
                for source_annotation in source_frame.get("annotations", []):
                    target_species_id = species_map.get(source_annotation.get("species_id"))
                    if target_species_id is None:
                        continue
                    self.add_annotation(
                        video_id=video["id"],
                        frame_number=frame_number,
                        time_seconds=time_seconds,
                        species_id=target_species_id,
                        track_id=source_annotation.get("track_id", "UNKNOWN-001"),
                        box=(float(source_annotation["x"]), float(source_annotation["y"]), float(source_annotation["width"]), float(source_annotation["height"])),
                        status=source_annotation.get("status", "verified"),
                        source=source_annotation.get("source", "manual"),
                        confidence=source_annotation.get("confidence"),
                        created_by=source_annotation.get("created_by", source_project.get("observer", "")),
                        life_stage=source_annotation.get("life_stage", "Unknown"),
                        activity=source_annotation.get("activity", "Passing"),
                        uncertain=bool(source_annotation.get("uncertain", False)),
                    )
                self.update_frame(
                    video["id"],
                    frame_number,
                    note=source_frame.get("note", ""),
                    image_path=source_frame.get("image_path", ""),
                )
                if source_frame.get("reviewed", False):
                    try:
                        self.set_frame_reviewed(video["id"], frame_number, True)
                    except ValueError:
                        self.set_frame_reviewed(video["id"], frame_number, False)
        return project

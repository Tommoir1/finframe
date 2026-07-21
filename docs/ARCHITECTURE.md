# Desktop architecture

```text
PySide6 student interface
        |
        +-- OpenCV video/frame reader
        +-- Default box-seeded CSRT/KCF/MIL propagation
        +-- SQLite project and annotation repository
        +-- Verified-only MaxN queries
        +-- COCO/YOLO/CSV dataset services
        +-- Ultralytics detector training
        +-- ByteTrack / BoT-SORT inference
```

## Persistent data

SQLite is the source of truth for projects, videos, species, frames, annotations, training runs, model versions and application settings. Source videos remain external files and are addressed by absolute path.

All database methods use short independent connections with foreign keys, WAL mode and a busy timeout. This supports UI and background-training threads within one workstation process.

## Human review boundary

Inference writes only `pending` annotations. Verified-only SQL queries are used by MaxN, training statistics and dataset builders. This makes the approval boundary structural rather than dependent on a UI filter.

Any modification to a verified label advances `verified_dataset_revision`. The training coordinator compares that revision with the last completed training snapshot, allowing corrections and deletions—not only new boxes—to trigger retraining.

## Tracking identity boundary

Drawing or correcting a box seeds the strongest available OpenCV CPU tracker. Sequential playback writes its propagated geometry as pending `tracker` proposals. Seeking, rejection, deletion or the explicit stop action ends propagation.

Track IDs represent uninterrupted visibility only. If the last visible box touches the image boundary and then disappears, the seeded tracker retires that identity immediately. A boundary identity allocator applies the same rule to ByteTrack and BoT-SORT output, preventing reuse of an internal tracker ID after an exit. Missing detections away from an edge may retain an identity briefly as an in-frame occlusion. There is deliberately no cross-exit re-identification.

## Training lifecycle

1. Check minimum examples, species diversity, revision threshold and cooldown.
2. Record a queued training run.
3. Extract labelled frames for every verified annotation.
4. Create video-grouped train/validation splits.
5. Fine-tune from the active model or configured base model.
6. Register immutable candidate weights and metrics.
7. Activate only when the evaluation gate passes.
8. Retain the previous active model otherwise.

## Institutional deployment

The included repository is a single-host desktop database. A multi-computer deployment should keep the same service interfaces but replace direct SQLite access with an authenticated API backed by PostgreSQL and managed object storage. Direct SQLite access over SMB/NFS is not recommended.

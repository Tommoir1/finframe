# Desktop architecture

```text
PySide6 student interface
        |
        +-- Background OpenCV image/video reader and responsive 0.5×–6× playback
        +-- Default box-seeded CSRT/KCF/MIL propagation
        +-- SQLite project and annotation repository
        +-- Complete-frame MaxN queries
        +-- COCO/YOLO/CSV dataset services
        +-- Portable contribution bundle exporter/importer
        +-- Ultralytics detector training
        +-- ByteTrack / BoT-SORT inference
```

## Persistent data

SQLite is the source of truth for projects, image/video media, species, frames, annotations, training runs, model versions and application settings. Original source media remain external files. Frames arriving in student contribution bundles are copied into managed storage under the receiving FinFrame data directory.

All database methods use short independent connections with foreign keys, WAL mode and a busy timeout. This supports UI and background-training threads within one workstation process.

Video decoding and seeded tracker updates run in a dedicated playback thread. Tracker proposals for one frame are written in a single SQLite transaction, video-frame UI refreshes omit the expensive MaxN query, and MaxN is refreshed when playback pauses. High-speed playback may skip display frames only when no seeded tracker is active; seeded tracking always consumes sequential frames.

## Human review boundary

Inference writes only `pending` annotations. Final MaxN and observation exports query only verified boxes whose frames are complete. Training builders add a second database filter for selected keyframes. This makes both review boundaries structural rather than dependent on a UI filter.

Any edit to a complete frame invalidates its completion and training selection. Re-completing the edited frame, or explicitly marking a selected frame incomplete, advances `training_dataset_revision`. This prevents training from starting halfway through a correction. The coordinator compares the revision with the last completed snapshot, so finalized corrections and explicit removals trigger retraining.

## Tracking identity boundary

Drawing or correcting a box seeds the strongest available OpenCV CPU tracker. Sequential playback writes its propagated geometry as pending `tracker` proposals. Seeking, rejection, deletion or the explicit stop action ends propagation.

Track IDs represent uninterrupted visibility only. If the last visible box touches the image boundary and then disappears, the seeded tracker retires that identity immediately. A boundary identity allocator applies the same rule to ByteTrack and BoT-SORT output, preventing reuse of an internal tracker ID after an exit. Missing detections away from an edge may retain an identity briefly as an in-frame occlusion. There is deliberately no cross-exit re-identification.

## Training lifecycle

1. Check minimum examples, species diversity, revision threshold and cooldown.
2. Record a queued training run.
3. Extract complete manual/corrected frames and temporally sampled complete tracker/AI frames, including selected negatives.
4. Create video-grouped train/validation splits.
5. Fine-tune from the active model or configured base model.
6. Register immutable candidate weights and metrics.
7. Activate only when the evaluation gate passes.
8. Retain the previous active model otherwise.

## Institutional deployment

Students on separate computers should export `.finframe.zip` contributions and the instructor should batch-import them into the training workstation. The included repository remains a single-host desktop database. A simultaneous multi-computer deployment should keep the same service interfaces but replace direct SQLite access with an authenticated API backed by PostgreSQL and managed object storage. Direct SQLite access over SMB/NFS is not recommended.

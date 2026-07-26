# Desktop architecture

```text
PySide6 student interface
        |
        +-- Background OpenCV image/video reader and responsive 0.5×–6× playback
        +-- Optional background MobileSAM / SAM 3 point-prompt segmentation
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

Video decoding runs in a dedicated playback thread. Video-frame UI refreshes omit the expensive MaxN query, and MaxN is refreshed when playback pauses. High-speed playback may skip display frames to remain responsive. Detector-driven ByteTrack and BoT-SORT operate separately from ordinary playback.

## Human review boundary

Detector and tracker inference writes only `pending` annotations. A SAM result remains an ephemeral canvas preview while the student adds positive and negative points; explicit acceptance stores the mask and derived box as a human-reviewed `manual` annotation, displayed as `SAM Manual` in the interface. Final MaxN and observation exports query only verified boxes whose frames are complete. Training builders add a second database filter for selected keyframes. This makes both review boundaries structural rather than dependent on a UI filter.

Any edit to a complete frame invalidates its completion and training selection. Re-completing the edited frame, or explicitly marking a selected frame incomplete, advances `training_dataset_revision`. The training tab reports how many dataset changes have accumulated since the previous run, and the next explicit **Train now** request includes finalized corrections and removals.

## Tracking identity boundary

Track IDs represent uninterrupted visibility only. A boundary identity allocator prevents ByteTrack and BoT-SORT from reusing an internal tracker ID after a fish exits the frame. Missing detections away from an edge may retain an identity briefly as an in-frame occlusion. There is deliberately no cross-exit re-identification.

## Training lifecycle

1. Wait for an explicit **Train now** request, then check minimum examples and species diversity.
2. Record a queued training run.
3. Extract complete manual/corrected frames and temporally sampled complete tracker/AI frames, including selected negatives.
4. Create video-grouped train/validation splits.
5. Fine-tune from the active model or configured base model.
6. Register immutable candidate weights and metrics.
7. Activate only when the evaluation gate passes.
8. Retain the previous active model otherwise.

## Institutional deployment

Students on separate computers should export `.finframe.zip` contributions and the instructor should batch-import them into the training workstation. The included repository remains a single-host desktop database. A simultaneous multi-computer deployment should keep the same service interfaces but replace direct SQLite access with an authenticated API backed by PostgreSQL and managed object storage. Direct SQLite access over SMB/NFS is not recommended.

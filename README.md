# FinFrame Desktop

FinFrame is a Python desktop replacement for the single-camera EventMeasure MaxN workflow. Students annotate fish once for ecological analysis; the same verified bounding boxes accumulate automatically into an object-detection dataset.

The application is local-first. Video remains on the workstation, while projects, taxonomy, annotations, review decisions, training runs and model versions are stored in one SQLite database.

## Student workflow

1. Choose **Annotate video**, **Annotate images**, or **Open existing project** at startup.
2. Create a survey project and record deployment, site and observer metadata.
3. Add one or more source videos or still images.
4. Select a species and draw a box around every visible fish on an observation frame or image.
5. For video, optionally choose 0.5× through 6× playback and press Play. Every drawn box seeds a CPU tracker by default.
6. FinFrame immediately calculates verified per-frame counts and per-species MaxN.
7. Manual boxes enter the shared training dataset as verified labels.
8. Propagated, AI or detector-tracker boxes enter as pending proposals and are visibly dashed.
9. A student must approve, correct or reject every proposal.
10. Only approved or corrected proposals enter MaxN, exports and retraining.

All projects in the database contribute to training. Opening another image or video does not discard earlier annotations.

For a teaching cohort, each student exports one `.finframe.zip` contribution. It contains the project metadata, every annotation decision and JPEG copies of annotated frames, but not complete source videos. An instructor can batch-import many contributions into one FinFrame installation; verified labels retain observer attribution, pending proposals remain pending, duplicate bundles are rejected, and the embedded frames remain available for training without relinking the students' source files.

## Desktop features

- Native PySide6 desktop interface; no browser or separate local web server
- First-class still-image annotation and multi-image import
- OpenCV video playback from 0.5× to 6×, timeline seeking, frame stepping and five-second jumps
- Bounding-box drawing, selection, movement and resizing
- Default box-seeded propagation during playback, with corrected boxes re-seeding the tracker
- Shared species taxonomy, scientific names, stable codes and track IDs
- Life stage, activity, uncertainty and student/observer attribution
- Verified per-frame counts and live per-species MaxN
- Audited `pending`, `verified` and `rejected` annotation states
- AI suggestions on the current frame
- Species suggestions for newly drawn boxes once an active detector exists; these remain pending until reviewed
- Whole-video ByteTrack or BoT-SORT proposals using the active detector
- COCO and YOLO exports across every verified project
- Optional extraction of native-resolution labelled JPEG frames
- Project JSON backups and label-only exports with frame manifests
- Portable `.finframe.zip` student contributions and batch instructor import
- Duplicate-contribution protection in the combined training database
- Automatic detector retraining and model versioning

## Approval and dataset safety

AI output is never treated as truth. A detector or tracker proposal is stored as `pending` and is excluded from:

- MaxN and per-frame ecological counts
- COCO and YOLO labels
- detector retraining
- released observation data

Approving an unchanged proposal records `ai_verified` or `tracker_verified`. Changing its class, geometry or metadata before approval records `ai_corrected` or `tracker_corrected`. Rejected proposals remain outside the dataset.

Track IDs describe one uninterrupted period of visibility, not an inferred biological individual. When a tracked box reaches the image boundary and disappears, FinFrame permanently retires that identity. A fish entering later always receives a new identity. FinFrame does not attempt re-identification or suggest that a returning fish is the same individual. Brief missed detections away from the image boundary may retain the existing identity as an in-frame occlusion.

## Frequent retraining

The default policy checks for retraining after every **10 verified dataset changes**, with a two-minute cooldown. A dataset change includes:

- a new manual annotation
- an approved or corrected AI/tracker proposal
- a verified class or geometry correction
- deletion of an incorrect verified box

Training starts after at least 20 verified boxes across at least two species. Every run rebuilds the detector dataset from all verified annotations across all projects, images and videos. Pending predictions cannot reinforce themselves.

Each candidate is evaluated on held-out data. The current model remains active unless the candidate meets the configured mAP50-95 improvement gate. When at least two videos exist, complete videos are held out; a one-video temporal split is marked as preliminary.

## Installation

Python 3.10 or newer is required.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m finframe
```

The first training run uses `yolo11n.pt` by default. Ultralytics may download those base weights if they are not already present. Later runs warm-start from the active FinFrame detector.

## Data location

By default, Windows data is stored under `%LOCALAPPDATA%\FinFrame`:

```text
FinFrame/
  finframe.sqlite3
  contributions/
  models/
  training/
  exports/
```

Choose another managed location with:

```powershell
$env:FINFRAME_DATA_DIR = "D:\FinFrameData"
python -m finframe
```

Original images and videos retain their filesystem paths. Imported contribution frames are copied under `contributions/`, inside the same FinFrame data directory as the database. Moving an original source requires relinking it, but imported contribution bundles remain self-contained for training. Do not place SQLite directly on an unreliable network share. Concurrent students on different computers should use contribution bundles or a centrally deployed database/API rather than sharing the SQLite file.

## Exports

COCO and YOLO exports include only verified annotations. Extracted frame images are optional. When images are omitted, `metadata/frame_manifest.csv` records the source path, frame number, timestamp and dimensions for later extraction.

The **Export contribution** action serves a different purpose: it always embeds the annotated frames plus all verified, pending and rejected decisions needed to reproduce the student's review state. The receiving instructor can select multiple contribution files in one import operation.

Training, validation and test data must be split by complete deployment or video. Randomly splitting neighbouring frames causes serious data leakage.

See [docs/DATASET_FORMAT.md](docs/DATASET_FORMAT.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Tests

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

## Windows application build

```powershell
.\scripts\build_windows.ps1
```

The script creates an isolated CPU-only build environment so CUDA libraries
installed elsewhere on the workstation are not accidentally bundled. The
portable application is created under `dist\FinFrame`. Maintainers who need a
machine-specific NVIDIA build can pass `-UseCuda` after installing the intended
CUDA-enabled PyTorch version in `.build-venv`.

## Licensing note

The desktop application uses Ultralytics for detector training and tracking integration. Review Ultralytics' AGPL and enterprise licensing options before distributing FinFrame outside an environment compatible with that licence. Model weights and source videos are excluded from Git by default.

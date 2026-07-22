# FinFrame Desktop

FinFrame is a Python desktop replacement for the single-camera EventMeasure MaxN workflow. Students annotate fish once for ecological analysis; completed observations also accumulate automatically into a curated object-detection dataset.

The application is local-first. Video remains on the workstation, while projects, taxonomy, annotations, review decisions, training runs and model versions are stored in one SQLite database.

## Student workflow

1. Choose **Annotate video**, **Annotate images**, or **Open existing project** at startup.
2. Create a survey project and record deployment, site and observer metadata.
3. Add one or more source videos, choose multiple still-image files, or import an image folder including its subfolders.
4. Select a species and draw a box around every visible fish on an observation frame or image.
5. Mark the frame complete only after every visible fish is boxed. Only complete frames contribute to final MaxN.
6. For video, optionally choose 0.5× through 6× playback and press Play. Every drawn box seeds a CPU tracker by default. Playback decoding and tracking run in the background, so speed and navigation controls remain responsive.
7. Propagated, AI or detector-tracker boxes enter as pending proposals and are visibly dashed.
8. Correct proposals while watching, then use **Approve watched segment** after checking that no fish were missed.
9. FinFrame calculates per-species MaxN from all complete frames and keeps incomplete work visibly excluded.
10. FinFrame selects complete, diverse keyframes for detector training; pending predictions can never train the model.

All projects in the database contribute to training. Opening another image or video does not discard earlier annotations.

For a teaching cohort, each student exports one `.finframe.zip` contribution. It contains project metadata, every annotation decision and JPEG copies of annotated or selected negative keyframes, but not complete source videos. An instructor can batch-import many contributions into one FinFrame installation; labels retain observer attribution, pending proposals remain pending, duplicate bundles are rejected, and embedded frames remain available for training without relinking the students' source files.

## Desktop features

- Native PySide6 desktop interface; no browser or separate local web server
- First-class still-image annotation with multi-file and recursive folder import; Image arrows move through the imported photo set
- Responsive background video playback from 0.5× to 6×, timeline seeking, frame stepping and five-second jumps
- Bounding-box drawing, selection, movement and resizing
- Default box-seeded propagation during playback, with corrected boxes re-seeding the tracker
- Shared species taxonomy with search-as-you-type selection, scientific names, stable codes and track IDs
- Life stage, activity, uncertainty and student/observer attribution
- Complete-frame counts and per-species MaxN
- Audited `pending`, `verified` and `rejected` annotation states
- AI suggestions on the current frame
- Species suggestions for newly drawn boxes once an active detector exists; these remain pending until reviewed
- Whole-video ByteTrack or BoT-SORT proposals using the active detector
- COCO and YOLO exports across completed observations in every project
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

Approval operates at two levels. Box approval means that one proposed label is correct. Frame completion means that the student has checked the whole image and boxed every visible fish. A frame cannot be completed while any proposal on it is pending. Editing a complete frame automatically makes it incomplete again so the changed observation must be reviewed.

All complete frames can support MaxN, but using every tracked video frame for training would overweight near-identical images. FinFrame therefore always selects complete manual/corrected frames and samples unchanged tracker/AI frames at most once per second by default. Complete sampled frames with no fish are retained as negative training images. One selected frame containing five fish is one training image with five bounding-box labels; a tracker following those fish through 100 frames does not become 500 independent training examples.

Playback speed is a target. Without active seeded boxes, FinFrame can skip display frames at high speed. While CPU box propagation is active it processes frames sequentially to preserve tracking accuracy, so older hardware may run below the selected 6× target; the interface remains responsive and proposals are still retained.

Track IDs describe one uninterrupted period of visibility, not an inferred biological individual. When a tracked box reaches the image boundary and disappears, FinFrame permanently retires that identity. A fish entering later always receives a new identity. FinFrame does not attempt re-identification or suggest that a returning fish is the same individual. Brief missed detections away from the image boundary may retain the existing identity as an in-frame occlusion.

## Frequent retraining

The default policy checks for retraining after every **10 selected-keyframe changes**, with a two-minute cooldown. A training-dataset change includes:

- completing a useful manual or corrected frame
- selecting a temporally spaced approved tracker/AI frame
- selecting a reviewed negative frame
- changing or deleting a label on a previously selected frame

Training starts after at least 20 boxes across at least two species in selected keyframes. Every run rebuilds the detector dataset from all complete selected keyframes across all projects, images and videos. Pending predictions and incomplete frames cannot reinforce the model.

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

COCO and YOLO observation exports include only verified annotations on complete frames. Extracted frame images are optional. When images are omitted, `metadata/frame_manifest.csv` records the source path, frame number, timestamp, completion state, training selection and dimensions for later extraction.

The **Export contribution** action serves a different purpose: it embeds annotated frames, selected negative keyframes and all verified, pending and rejected decisions needed to reproduce the student's review state. The receiving instructor can select multiple contribution files in one import operation.

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

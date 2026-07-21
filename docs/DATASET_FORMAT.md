# FinFrame dataset contract

FinFrame Desktop stores normalised bounding boxes and produces ecological observations and computer-vision labels from the same student-reviewed records.

## Annotation state

Every annotation has a `status`:

- `pending`: AI or tracker proposal awaiting a student decision
- `verified`: manual, approved or corrected student label
- `rejected`: proposal excluded by a student

Only `verified` annotations contribute to MaxN, exports and detector training.

The `source` audit field is one of:

- `manual`
- `ai`
- `ai_verified`
- `ai_corrected`
- `tracker`
- `tracker_verified`
- `tracker_corrected`

`ai` and `tracker` are pending sources. The suffixed forms record the student's decision.

## Coordinates

SQLite and project backups store `x`, `y`, `width` and `height` between zero and one relative to the uncropped source frame. `x` and `y` are the top-left corner.

COCO exports convert these to pixel `[x, y, width, height]`. YOLO exports use normalised `class centre_x centre_y width height` lines.

## Dataset archives

COCO:

```text
images/                         # optional
annotations/instances.json
metadata/frame_manifest.csv
metadata/verified_annotations.json
README.txt
```

YOLO:

```text
images/                         # optional
labels/
data.yaml
classes.txt
metadata/frame_manifest.csv
metadata/verified_annotations.json
README.txt
```

The frame manifest always records the source video path, project, deployment, frame number, timestamp and dimensions. Label-only archives therefore remain extractable later when the original videos are available.

## Training snapshots

Automatic training rebuilds a YOLO dataset from every verified annotation in the database. With two or more videos, whole videos form the validation group. With only one video, FinFrame creates a temporal holdout and records that the metric is preliminary.

Each training run stores:

- trigger reason
- verified example count
- dataset path
- base model
- status and error information
- validation metrics
- activation decision

Candidate models do not overwrite earlier weights. An activated model receives a new version and previous weights remain auditable.

## Data leakage

Do not randomly distribute neighbouring frames among train, validation and test sets. Use complete deployments/videos as the grouping unit and keep a final test set that is never used for automatic activation decisions.

# FinFrame dataset contract

FinFrame Desktop stores normalised bounding boxes and produces ecological observations and computer-vision labels from the same student-reviewed records.

## Annotation state

Every annotation has a `status`:

- `pending`: AI or tracker proposal awaiting a student decision
- `verified`: manual, approved or corrected student label
- `rejected`: proposal excluded by a student

Box status alone is not sufficient. A frame must also be marked `reviewed`, meaning the student confirmed that every visible fish was boxed. Only verified boxes on reviewed frames contribute to final MaxN and released observation exports.

Reviewed frames have a separate `training_selected` flag. FinFrame always selects manual/corrected reviewed frames, temporally samples unchanged tracker/AI frames, and can select reviewed zero-fish frames as negatives. This prevents neighbouring video frames from overwhelming a training set with near-duplicates.

The `source` audit field is one of:

- `manual`
- `ai`
- `ai_verified`
- `ai_corrected`
- `tracker`
- `tracker_verified`
- `tracker_corrected`

`ai` and `tracker` are pending sources. The suffixed forms record the student's decision.

`track_id` identifies one uninterrupted visible track segment; it is not an assertion that two appearances belong to the same biological individual. An identity ends when a fish leaves the image boundary. A later entrance receives a new ID, with no re-identification across exits.

## Species taxonomy

FinFrame's bundled catalogue contains the 99 species-column headers from the first worksheet named `Master Sheet` in `species_list.xlsx` (columns H–DB). Source spelling and abbreviations are preserved so historical EventMeasure labels remain searchable. Deterministic, collision-checked codes beginning with `MS` are used for track IDs and exports. User-created taxonomy records remain alongside the master catalogue.

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

The frame manifest always records the source media path, project, deployment, frame number, timestamp and dimensions. Label-only archives therefore remain extractable later when the original images or videos are available.

## Student contribution bundles

The portable cohort format is one `.finframe.zip` per student project:

```text
project.finframe.json          # taxonomy, media metadata and every review decision
frames/<media-id>/*.jpg        # annotated frames and selected zero-fish keyframes
README.txt
```

Complete videos are deliberately omitted. On import, embedded frames are copied under the receiving installation's `contributions/` directory and their paths are registered in SQLite. Complete selected frames can therefore train without the student's original media. Pending and rejected records are retained for audit/review but remain excluded from training. A SHA-256 bundle fingerprint prevents accidental duplicate imports.

## Training snapshots

Automatic training rebuilds a YOLO dataset from every complete selected keyframe in the database, including still images, selected negative frames and embedded contribution frames. A selected frame with no fish has an empty YOLO label file. With two or more independent media sources, complete sources form the validation group. With only one video, FinFrame creates a temporal holdout and records that the metric is preliminary.

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

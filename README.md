# FinFrame

FinFrame is a local-first browser prototype for annotating fish in underwater video. It combines the familiar EventMeasure-style abundance workflow with bounding boxes that can be reused as object-detection labels.

## What works

- Local MP4/MOV/WebM playback with scrubbing, speed control, five-second jumps and frame stepping
- Bounding-box drawing, selection, movement and resizing
- Species taxonomy, stable track IDs, life stage, activity and uncertainty flags
- Clone boxes from the preceding annotated frame
- Live per-species counts on every annotated frame, MaxN-frame badges, running MaxN, final MaxN, mean count and first-arrival calculation
- Deployment metadata, frame notes and review status
- Review table with flagged-annotation and MaxN-frame filters
- Autosave in browser storage
- Full project JSON import/export
- One-click extraction of the exact labelled video frames at native resolution
- Complete COCO and YOLO dataset ZIPs containing JPEG images, labels, taxonomy, project metadata and per-frame counts
- Ecology observation CSV and dedicated per-frame count/MaxN CSV exports
- Responsive interface and a built-in sample survey

## Run it

This is a dependency-free static app. From this folder, run:

```powershell
python -m http.server 4173
```

Then open `http://127.0.0.1:4173`.

## Important scope note

This first version covers the single-camera annotation and MaxN workflow. True stereo 3D length/range measurement requires camera calibration, frame synchronisation and photogrammetric reconstruction, so it should be treated as a separate validated module rather than inferred from a 2D box.

The original source video must be open when a COCO or YOLO dataset is exported. FinFrame seeks to every labelled frame, extracts a high-quality JPEG at the video's native dimensions and packages the image with its matching labels. The export is performed locally in the browser; neither footage nor labels are uploaded.

## Machine-learning compatibility

FinFrame treats model compatibility as a core data-contract requirement:

- Bounding boxes are stored internally as normalised `x`, `y`, `width`, `height` values relative to the uncropped source frame.
- COCO exports convert those boxes to pixel `x`, `y`, `width`, `height` coordinates and assign stable category/image/annotation IDs.
- YOLO exports use normalised centre `x`, centre `y`, width and height coordinates with a deterministic class list.
- Every exported image is linked to its source video, frame number, timestamp and deployment ID.
- Track IDs, life stage, activity, uncertainty and review state are preserved in the project and COCO attributes.
- The complete editable project is bundled with each dataset so labels remain auditable.

Do not randomly split nearby frames from one video between training and validation. They are highly correlated and will inflate validation performance. Combine multiple deployment exports and split whole deployments or videos into train, validation and test groups.

See [docs/DATASET_FORMAT.md](docs/DATASET_FORMAT.md) for the full schema and training guidance.
The machine-readable project contract is in [schemas/finframe-project.schema.json](schemas/finframe-project.schema.json).

## Suggested production architecture

Keep this interaction model, then add a service layer for user accounts, project locking, cloud/object storage, dataset versioning and audit logs. For model-assisted annotation, ingest detector/tracker proposals as editable boxes and preserve both the original prediction and the student's reviewed label.

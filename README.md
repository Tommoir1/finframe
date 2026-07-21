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
- Optional one-click extraction of the exact labelled video frames at native resolution
- COCO and YOLO dataset ZIPs containing labels, frame manifests, taxonomy, project metadata and per-frame counts, with JPEG images included only when requested
- On-device Class Assist that learns from verified crops and suggests species for newly drawn boxes
- Optional local detector tracking with ByteTrack or BoT-SORT; imported tracks remain review proposals
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

The original source video is not required for label-only COCO or YOLO exports. Every dataset includes a frame manifest with source filename, frame number, timestamp and dimensions. If **Include extracted frame images** is selected, FinFrame seeks to each labelled frame and packages a high-quality JPEG at the video's native dimensions. Extraction is entirely local; neither footage nor labels are uploaded.

## Class Assist

Class Assist is an optional local learning loop for the stage after a student draws a bounding box. It extracts a compact colour, texture and shape feature vector from verified fish crops and uses distance-weighted nearest neighbours to suggest the species of new boxes.

- Learning begins from manually labelled or corrected boxes while the source video is open.
- Existing verified annotations can be added with **Learn from existing boxes**.
- Suggestions remain pending until a student accepts the guess or chooses a different species.
- Pending guesses are excluded from MaxN, observation CSVs, COCO labels and YOLO labels.
- Accepted and corrected predictions become new verified training examples, allowing the assistant to improve during the annotation session.

This lightweight classifier is intended for immediate, private, in-browser assistance. It is not a replacement for training a full detector such as YOLO on the exported dataset; the export formats remain the path to a production model that can detect fish without a student first drawing a box.

## Automatic tracking

FinFrame includes an optional local service in [tracking_service/](tracking_service/) that combines custom fish-detector weights with ByteTrack or BoT-SORT. The browser sends the open video only to `127.0.0.1`; the service removes its temporary copy after inference.

All tracker output is imported as unverified proposals. Proposed boxes and track IDs do not contribute to MaxN or released datasets until accepted or corrected. Students can accept individual boxes or all proposals on the current frame.

ByteTrack is the recommended fast baseline after a fish detector has been trained. It associates detection boxes but cannot propagate a lone manually drawn box by itself. For that early annotation workflow, a future SAM 2 provider is preferable because it accepts a box prompt and propagates the object through video. See [tracking_service/README.md](tracking_service/README.md) for setup and tracker-selection guidance.

## Machine-learning compatibility

FinFrame treats model compatibility as a core data-contract requirement:

- Bounding boxes are stored internally as normalised `x`, `y`, `width`, `height` values relative to the uncropped source frame.
- COCO exports convert those boxes to pixel `x`, `y`, `width`, `height` coordinates and assign stable category/image/annotation IDs.
- YOLO exports use normalised centre `x`, centre `y`, width and height coordinates with a deterministic class list.
- Every exported image is linked to its source video, frame number, timestamp and deployment ID.
- Track IDs, life stage, activity, uncertainty and review state are preserved in the project and COCO attributes.
- The complete editable project is bundled with each dataset so labels remain auditable.
- Unverified model suggestions are retained for auditing but excluded from released labels and abundance metrics.

Do not randomly split nearby frames from one video between training and validation. They are highly correlated and will inflate validation performance. Combine multiple deployment exports and split whole deployments or videos into train, validation and test groups.

See [docs/DATASET_FORMAT.md](docs/DATASET_FORMAT.md) for the full schema and training guidance.
The machine-readable project contract is in [schemas/finframe-project.schema.json](schemas/finframe-project.schema.json).

## Suggested production architecture

Keep this interaction model, then add a service layer for user accounts, project locking, cloud/object storage, dataset versioning and audit logs. For model-assisted annotation, ingest detector/tracker proposals as editable boxes and preserve both the original prediction and the student's reviewed label.

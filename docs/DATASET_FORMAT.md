# FinFrame dataset contract

FinFrame produces ecological observations and computer-vision labels from the same reviewed bounding boxes. This document defines the fields that must remain stable as the application evolves.

## Coordinate system

The project file stores each box as:

```json
{
  "x": 0.17,
  "y": 0.24,
  "w": 0.15,
  "h": 0.14
}
```

`x` and `y` are the top-left corner. All four values are normalised against the full, uncropped source frame and must remain between zero and one. The exported JPEG has the same width and height recorded in `project.video`.

COCO export converts the values to `[x_pixels, y_pixels, width_pixels, height_pixels]`. YOLO export converts them to `class_id centre_x centre_y width height`, with all geometry normalised.

## Dataset ZIPs

### COCO

```text
images/
  <video>_frame_00000310.jpg  # optional
annotations/
  instances.json
metadata/
  project.finframe.json
  per_frame_counts.csv
  frame_manifest.csv
README.txt
```

The annotation JSON uses standard COCO `images`, `annotations` and `categories` arrays. The following extra fields are intentional and may be ignored by consumers that only support core COCO:

- Images: `video_file`, `frame_number`, `timestamp_seconds`, `deployment_id`
- Annotations: `track_id`, and `attributes` containing `stage`, `activity` and `uncertain`

### YOLO

```text
images/
  <video>_frame_00000310.jpg  # optional
labels/
  <video>_frame_00000310.txt
data.yaml
classes.txt
metadata/
  project.finframe.json
  per_frame_counts.csv
  frame_manifest.csv
README.txt
```

Image and label basenames match exactly. Class indices are zero-based and follow the order in `classes.txt` and `data.yaml`.

Frame images are optional in both formats. `frame_manifest.csv` always identifies the source video, timestamp, frame number, dimensions and intended image filename so image extraction can happen later in a separate pipeline.

## Model-assisted labels

Annotations include `verified` and `labelSource` fields. `labelSource` is one of:

- `manual`: assigned by the annotator without a model suggestion
- `model`: pending Class Assist suggestion
- `model_verified`: model suggestion accepted by the annotator
- `corrected`: model suggestion changed by the annotator
- `tracker`: pending detector/tracker proposal
- `tracker_verified`: detector/tracker proposal accepted by the annotator
- `tracker_corrected`: detector/tracker proposal assigned a different species by the annotator

Only annotations where `verified` is not `false` contribute to MaxN or appear in COCO, YOLO and observation CSV exports. Pending predictions remain in `project.finframe.json` for audit and model-evaluation purposes.

The optional `featureVector`, `modelSuggestedSpeciesId`, `modelConfidence`, `modelVersion` and `learningExampleId` fields support the on-device continual-learning loop. `learningExampleId` deduplicates a verified annotation in the browser-level Class Assist library and allows corrections or deletions to update that example rather than reinforcing stale labels. The feature vector is not a substitute for the source crop and is not written into COCO or YOLO labels.

Class Assist uses verified feature examples from every project opened in the same browser profile. Taxa are matched across project-specific IDs using scientific name, species code and common name. The shared library is intentionally separate from project JSON, remains local to that browser profile and does not contain source frames. Importing a project seeds any verified feature vectors it contains; pending predictions are never added.

Tracked proposals additionally contain `trackingSource` (`bytetrack` or `botsort`) and `trackingRunId`. Stable tracker IDs are exported as `track_id` after verification. COCO attributes retain the tracker source for audit.

## Per-frame abundance

`per_frame_counts.csv` contains one row for every labelled species at every saved observation frame. Important columns are:

- `count_in_frame`: number of boxes for that species in the frame
- `running_maxn`: highest count encountered at or before the frame
- `final_maxn`: maximum count in any saved frame for the species
- `is_final_maxn_frame`: whether this frame reaches the final MaxN

Frames with a saved note or review state but no boxes remain part of the project audit trail. Dataset image exports include frames containing at least one bounding box.

## Train/validation/test splitting

Do not randomly split extracted frames. Consecutive frames are nearly duplicates, so a frame-level random split leaks visual information and produces misleading evaluation scores.

Use `deployment_id` or source video as the grouping unit:

1. Combine exports from multiple independent deployments.
2. Allocate complete deployments to training, validation or test.
3. Check that rare species occur in the evaluation groups.
4. Record the split manifest and project-file hashes with each model run.

## Quality recommendations

- Review every uncertain annotation before releasing a training version.
- Define whether partially visible and heavily occluded fish should be boxed.
- Keep taxonomic codes stable across projects.
- Retain negative frames separately when training detectors; frames containing no fish are useful but are not generated automatically by the labelled-frame export.
- Version released datasets instead of overwriting them.
- Preserve the original project JSON so annotations can be traced back to observer, deployment, frame and source video.

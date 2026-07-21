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
  <video>_frame_00000310.jpg
annotations/
  instances.json
metadata/
  project.finframe.json
  per_frame_counts.csv
README.txt
```

The annotation JSON uses standard COCO `images`, `annotations` and `categories` arrays. The following extra fields are intentional and may be ignored by consumers that only support core COCO:

- Images: `video_file`, `frame_number`, `timestamp_seconds`, `deployment_id`
- Annotations: `track_id`, and `attributes` containing `stage`, `activity` and `uncertain`

### YOLO

```text
images/
  <video>_frame_00000310.jpg
labels/
  <video>_frame_00000310.txt
data.yaml
classes.txt
metadata/
  project.finframe.json
  per_frame_counts.csv
README.txt
```

Image and label basenames match exactly. Class indices are zero-based and follow the order in `classes.txt` and `data.yaml`.

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

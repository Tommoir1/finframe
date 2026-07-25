from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from .database import Database, new_id, utc_now
from .sam_assist import mask_area_from_rle


class DatasetError(RuntimeError):
    pass


def _safe_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "_" for character in value.strip())
    return cleaned.strip("_").lower() or "finframe"


def _frame_name(annotation: dict[str, Any]) -> str:
    return f"{_safe_name(Path(annotation['file_name']).stem)}_frame_{annotation['frame_number']:08d}.jpg"


def _csv_bytes(rows: list[list[Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _verified_records(db: Database, video_id: str | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotations = db.verified_annotations(video_id)
    if not annotations:
        raise DatasetError("There are no verified annotations on complete frames to export")
    frames: dict[tuple[str, int], dict[str, Any]] = {}
    for annotation in annotations:
        key = (annotation["video_id"], annotation["frame_number"])
        frames.setdefault(key, {
            "video_id": annotation["video_id"],
            "video_path": annotation["video_path"],
            "file_name": annotation["file_name"],
            "media_type": annotation.get("media_type", "video"),
            "image_path": annotation.get("image_path", ""),
            "frame_number": annotation["frame_number"],
            "time_seconds": annotation["time_seconds"],
            "width": annotation["video_width"],
            "height": annotation["video_height"],
            "fps": annotation["fps"],
            "project_id": annotation["project_id"],
            "project_name": annotation["project_name"],
            "deployment_id": annotation["deployment_id"],
            "site": annotation["site"],
            "observer": annotation["observer"],
            "reviewed": annotation["reviewed"],
            "training_selected": annotation.get("training_selected", 0),
            "training_reason": annotation.get("training_reason", ""),
            "note": annotation["note"],
        })
    return annotations, list(frames.values())


def _read_frame(
    video_path: str,
    frame_number: int,
    *,
    media_type: str = "video",
    image_path: str = "",
) -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise DatasetError("OpenCV is required to extract video frames") from exc
    still_path = Path(image_path) if image_path else None
    if still_path and still_path.is_file():
        image = cv2.imread(str(still_path))
        if image is None:
            raise DatasetError(f"Could not read extracted frame: {still_path}")
        return image
    if media_type == "image":
        image = cv2.imread(video_path)
        if image is None:
            raise DatasetError(f"Could not read source image: {video_path}")
        return image

    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            raise DatasetError(f"Could not open source video: {video_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, image = capture.read()
        if not ok:
            raise DatasetError(f"Could not read frame {frame_number} from {video_path}")
        return image
    finally:
        capture.release()


def export_dataset(db: Database, destination: str | Path, *, fmt: str, include_images: bool, video_id: str | None = None) -> Path:
    if fmt not in {"coco", "yolo"}:
        raise ValueError("fmt must be coco or yolo")
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    annotations, frames = _verified_records(db, video_id)
    species = sorted({item["species_id"]: item for item in annotations}.values(), key=lambda item: item["code"])
    category_index = {item["species_id"]: index for index, item in enumerate(species)}
    frame_index = {(frame["video_id"], frame["frame_number"]): index + 1 for index, frame in enumerate(frames)}
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        grouped[(annotation["video_id"], annotation["frame_number"])].append(annotation)

    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        if include_images:
            for frame in frames:
                image = _read_frame(
                    frame["video_path"],
                    frame["frame_number"],
                    media_type=frame["media_type"],
                    image_path=frame["image_path"],
                )
                import cv2
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise DatasetError(f"Could not encode frame {frame['frame_number']}")
                archive.writestr(f"images/{_frame_name(frame)}", encoded.tobytes())

        if fmt == "coco":
            coco_images = [{
                "id": frame_index[(frame["video_id"], frame["frame_number"])],
                "file_name": f"images/{_frame_name(frame)}",
                "width": frame["width"],
                "height": frame["height"],
                "video_file": frame["file_name"],
                "media_type": frame["media_type"],
                "frame_number": frame["frame_number"],
                "timestamp_seconds": frame["time_seconds"],
                "deployment_id": frame["deployment_id"],
            } for frame in frames]
            coco_annotations = []
            for index, annotation in enumerate(annotations, start=1):
                width, height = annotation["video_width"], annotation["video_height"]
                bbox = [annotation["x"] * width, annotation["y"] * height, annotation["width"] * width, annotation["height"] * height]
                segmentation: dict[str, Any] | list[Any] = []
                if annotation.get("mask_rle"):
                    try:
                        segmentation = json.loads(annotation["mask_rle"])
                    except (TypeError, json.JSONDecodeError):
                        segmentation = []
                mask_area = mask_area_from_rle(str(annotation.get("mask_rle") or ""))
                coco_annotations.append({
                    "id": index,
                    "image_id": frame_index[(annotation["video_id"], annotation["frame_number"])],
                    "category_id": category_index[annotation["species_id"]] + 1,
                    "bbox": [round(value, 2) for value in bbox],
                    "segmentation": segmentation,
                    "area": mask_area or round(bbox[2] * bbox[3], 2),
                    "iscrowd": 0,
                    "track_id": annotation["track_id"],
                    "attributes": {
                        "life_stage": annotation["life_stage"],
                        "activity": annotation["activity"],
                        "uncertain": bool(annotation["uncertain"]),
                        "label_source": annotation["source"],
                        "created_by": annotation["created_by"],
                    },
                })
            categories = [{
                "id": index + 1,
                "name": item["common_name"],
                "scientific_name": item["scientific_name"],
                "code": item["code"],
                "supercategory": "fish",
            } for index, item in enumerate(species)]
            archive.writestr("annotations/instances.json", json.dumps({
                "info": {"description": "FinFrame completed observations", "version": "2.0", "created": utc_now()},
                "images": coco_images,
                "annotations": coco_annotations,
                "categories": categories,
            }, indent=2))
        else:
            for frame in frames:
                lines = []
                for annotation in grouped[(frame["video_id"], frame["frame_number"])]:
                    centre_x = annotation["x"] + annotation["width"] / 2
                    centre_y = annotation["y"] + annotation["height"] / 2
                    lines.append(f"{category_index[annotation['species_id']]} {centre_x:.6f} {centre_y:.6f} {annotation['width']:.6f} {annotation['height']:.6f}")
                archive.writestr(f"labels/{Path(_frame_name(frame)).with_suffix('.txt').name}", "\n".join(lines))
            names = "\n".join(f"  {index}: {json.dumps(item['common_name'])}" for index, item in enumerate(species))
            archive.writestr("data.yaml", f"path: .\ntrain: images\nval: images\nnames:\n{names}\n")
            archive.writestr("classes.txt", "\n".join(item["common_name"] for item in species))

        manifest = [["image_file", "source_media", "media_type", "source_path", "project", "deployment_id", "frame_number", "time_seconds", "width", "height", "reviewed", "training_keyframe", "training_reason", "image_included"]]
        for frame in frames:
            manifest.append([_frame_name(frame), frame["file_name"], frame["media_type"], frame["video_path"], frame["project_name"], frame["deployment_id"], frame["frame_number"], frame["time_seconds"], frame["width"], frame["height"], bool(frame["reviewed"]), bool(frame["training_selected"]), frame["training_reason"], include_images])
        archive.writestr("metadata/frame_manifest.csv", _csv_bytes(manifest))
        counts: dict[tuple[str, int, str], int] = defaultdict(int)
        for annotation in annotations:
            counts[(annotation["video_id"], annotation["frame_number"], annotation["species_id"])] += 1
        maximums: dict[tuple[str, str], int] = defaultdict(int)
        for (count_video, _frame_number, species_id), count in counts.items():
            maximums[(count_video, species_id)] = max(maximums[(count_video, species_id)], count)
        species_by_video: dict[str, set[str]] = defaultdict(set)
        for annotation in annotations:
            species_by_video[annotation["video_id"]].add(annotation["species_id"])
        species_lookup = {item["species_id"]: item for item in annotations}
        frame_count_rows = [["project", "deployment_id", "media", "frame_number", "time_seconds", "species_code", "common_name", "count_in_frame", "final_maxn", "is_maxn_frame", "reviewed"]]
        for frame in frames:
            for species_id in sorted(species_by_video[frame["video_id"]], key=lambda item: species_lookup[item]["code"]):
                species_record = species_lookup[species_id]
                count = counts[(frame["video_id"], frame["frame_number"], species_id)]
                maximum = maximums[(frame["video_id"], species_id)]
                frame_count_rows.append([frame["project_name"], frame["deployment_id"], frame["file_name"], frame["frame_number"], frame["time_seconds"], species_record["code"], species_record["common_name"], count, maximum, bool(count and count == maximum), bool(frame["reviewed"])])
        archive.writestr("metadata/per_frame_counts.csv", _csv_bytes(frame_count_rows))
        observation_rows = [["project", "deployment_id", "site", "observer", "media", "frame_number", "time_seconds", "species_code", "common_name", "scientific_name", "count_in_frame", "maxn", "is_maxn_frame", "track_id", "life_stage", "activity", "uncertain", "label_source", "created_by", "x", "y", "width", "height", "note"]]
        for annotation in annotations:
            count = counts[(annotation["video_id"], annotation["frame_number"], annotation["species_id"])]
            maximum = maximums[(annotation["video_id"], annotation["species_id"])]
            observation_rows.append([
                annotation["project_name"], annotation["deployment_id"], annotation["site"], annotation["observer"], annotation["file_name"],
                annotation["frame_number"], annotation["time_seconds"], annotation["code"], annotation["common_name"], annotation["scientific_name"],
                count, maximum, bool(count == maximum), annotation["track_id"], annotation["life_stage"], annotation["activity"], bool(annotation["uncertain"]),
                annotation["source"], annotation["created_by"], annotation["x"], annotation["y"], annotation["width"], annotation["height"], annotation["note"],
            ])
        archive.writestr("metadata/observations.csv", _csv_bytes(observation_rows))
        archive.writestr("metadata/verified_annotations.json", json.dumps(annotations, indent=2))
        archive.writestr("README.txt", (
            "FinFrame 2 completed-observation dataset\n\n"
            f"Format: {fmt.upper()}\nFrame images included: {'yes' if include_images else 'no'}\n"
            "Only verified boxes on student-completed frames are included. Pending proposals and incomplete frames are excluded.\n"
            "Split training, validation and test data by complete deployment/video to avoid leakage.\n"
        ))
    return destination


def export_project_backup(db: Database, project_id: str, destination: str | Path) -> Path:
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(db.project_snapshot(project_id), indent=2), encoding="utf-8")
    return destination


def export_contribution_bundle(db: Database, project_id: str, destination: str | Path) -> Path:
    """Create one transferable archive containing labels and annotated source frames."""
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot = db.project_snapshot(project_id)
    if not any(
        frame.get("annotations") or frame.get("training_selected")
        for video in snapshot["videos"]
        for frame in video.get("frames", [])
    ):
        raise DatasetError("Annotate at least one frame or complete a zero-fish keyframe before exporting a contribution")
    embedded_frames = 0
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for video in snapshot["videos"]:
            for frame in video.get("frames", []):
                if not frame.get("annotations") and not frame.get("training_selected"):
                    continue
                image = _read_frame(
                    video["path"],
                    int(frame["frame_number"]),
                    media_type=video.get("media_type", "video"),
                    image_path=frame.get("image_path", ""),
                )
                import cv2
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise DatasetError(f"Could not encode {video['file_name']} frame {frame['frame_number']}")
                portable_path = f"frames/{_safe_name(video['id'])}/{int(frame['frame_number']):08d}.jpg"
                archive.writestr(portable_path, encoded.tobytes())
                frame["portable_image"] = portable_path
                frame["image_path"] = ""
                embedded_frames += 1
        snapshot["contribution"] = {
            "created_at": utc_now(),
            "embedded_frames": embedded_frames,
            "includes_full_videos": False,
        }
        archive.writestr("project.finframe.json", json.dumps(snapshot, indent=2))
        archive.writestr(
            "README.txt",
            "FinFrame student contribution bundle\n\n"
            "Contains project metadata, all annotation decisions, and JPEG copies of annotated and selected negative frames.\n"
            "It does not contain complete source videos. Import it into FinFrame to add completed observations and selected keyframes to the shared database.\n",
        )
    return destination


def import_contribution_bundle(db: Database, source: str | Path, data_dir: str | Path) -> dict[str, Any]:
    """Import a student archive and retain its embedded frames for future training."""
    source = Path(source).expanduser().resolve()
    hasher = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    imported_hashes = db.get_setting("imported_contribution_hashes", [])
    if digest in imported_hashes:
        raise DatasetError("This contribution bundle has already been imported")
    contribution_root = Path(data_dir).expanduser().resolve() / "contributions" / new_id("bundle")
    contribution_root.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(source, "r") as archive:
            try:
                snapshot = json.loads(archive.read("project.finframe.json"))
            except KeyError as exc:
                raise DatasetError("The archive has no FinFrame project manifest") from exc
            embedded_frames = 0
            for video in snapshot.get("videos", []):
                for frame in video.get("frames", []):
                    portable_path = frame.get("portable_image")
                    if not portable_path:
                        continue
                    try:
                        frame_bytes = archive.read(portable_path)
                    except KeyError as exc:
                        raise DatasetError(f"Contribution frame is missing: {portable_path}") from exc
                    target = contribution_root / f"{_safe_name(video.get('id', 'media'))}_{int(frame['frame_number']):08d}.jpg"
                    target.write_bytes(frame_bytes)
                    frame["image_path"] = str(target)
                    embedded_frames += 1
        project = db.import_project_snapshot(snapshot)
        db.set_setting("imported_contribution_hashes", [*imported_hashes, digest])
        return {"project": project, "embedded_frames": embedded_frames, "storage": contribution_root}
    except Exception:
        for child in contribution_root.glob("*"):
            if child.is_file():
                child.unlink()
        contribution_root.rmdir()
        raise


def build_yolo_training_dataset(db: Database, destination: str | Path) -> dict[str, Any]:
    """Build a detector dataset from reviewed, automatically sampled keyframes."""
    destination = Path(destination).expanduser().resolve()
    frames = db.training_frames()
    annotations = db.verified_annotations(training_only=True)
    if not frames:
        raise DatasetError("No complete training keyframes are available")
    if not annotations:
        raise DatasetError("Training requires at least one complete keyframe containing a verified fish")
    species = sorted({item["species_id"]: item for item in annotations}.values(), key=lambda item: item["code"])
    class_index = {item["species_id"]: index for index, item in enumerate(species)}
    by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for annotation in annotations:
        by_frame[(annotation["video_id"], annotation["frame_number"])].append(annotation)

    video_ids = sorted({frame["video_id"] for frame in frames})
    if len(video_ids) >= 2:
        validation_count = max(1, round(len(video_ids) * 0.2))
        validation_videos = set(video_ids[-validation_count:])
        def split_for(frame: dict[str, Any], index: int) -> str:
            return "val" if frame["video_id"] in validation_videos else "train"
        split_strategy = "complete videos"
    else:
        validation_start = max(1, int(len(frames) * 0.8))
        def split_for(frame: dict[str, Any], index: int) -> str:
            return "val" if index >= validation_start else "train"
        split_strategy = "temporal holdout within one video"

    split_counts = {"train": 0, "val": 0}
    for index, frame in enumerate(frames):
        split = split_for(frame, index)
        image_dir, label_dir = destination / "images" / split, destination / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        image = _read_frame(
            frame["video_path"],
            frame["frame_number"],
            media_type=frame["media_type"],
            image_path=frame["image_path"],
        )
        import cv2
        image_path = image_dir / _frame_name(frame)
        if not cv2.imwrite(str(image_path), image):
            raise DatasetError(f"Could not write training image {image_path}")
        label_lines = []
        for annotation in by_frame[(frame["video_id"], frame["frame_number"])]:
            centre_x = annotation["x"] + annotation["width"] / 2
            centre_y = annotation["y"] + annotation["height"] / 2
            label_lines.append(f"{class_index[annotation['species_id']]} {centre_x:.6f} {centre_y:.6f} {annotation['width']:.6f} {annotation['height']:.6f}")
        (label_dir / image_path.with_suffix(".txt").name).write_text("\n".join(label_lines), encoding="utf-8")
        split_counts[split] += 1

    if not split_counts["train"] or not split_counts["val"]:
        raise DatasetError("At least two independently labelled frames are required for detector training")
    names = "\n".join(f"  {index}: {json.dumps(item['code'])}" for index, item in enumerate(species))
    yaml_path = destination / "data.yaml"
    yaml_path.write_text(f"path: {json.dumps(str(destination))}\ntrain: images/train\nval: images/val\nnames:\n{names}\n", encoding="utf-8")
    (destination / "split_manifest.json").write_text(json.dumps({
        "strategy": split_strategy,
        "train_frames": split_counts["train"],
        "validation_frames": split_counts["val"],
        "videos": video_ids,
        "warning": "Validation is preliminary until at least two independent videos are available." if len(video_ids) == 1 else "",
    }, indent=2), encoding="utf-8")
    return {
        "path": destination,
        "yaml": yaml_path,
        "examples": len(annotations),
        "frames": len(frames),
        "classes": len(species),
        "split": split_counts,
        "species": species,
    }

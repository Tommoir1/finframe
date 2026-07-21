# Optional local tracking service

This service runs a custom fish detector with ByteTrack or BoT-SORT and returns normalised tracked-box proposals to FinFrame. It is optional: manual annotation, MaxN, Class Assist and data exports do not depend on it.

## Why ByteTrack is optional

ByteTrack associates boxes produced by a detector on successive frames. It cannot follow a manually drawn fish unless a detector also finds that fish on later frames. Use it after training a detector from FinFrame's COCO or YOLO exports.

For the earlier annotation stage—one manual box followed forward through video—a promptable video model such as SAM 2 is a better provider. That integration is intentionally kept separate because it requires large checkpoints and typically a CUDA GPU.

## Setup

Create a Python environment and install the optional dependencies:

```powershell
python -m pip install -r tracking_service/requirements.txt
```

Set the detector weights trained on your FinFrame species classes:

```powershell
$env:FINFRAME_MODEL_PATH = "C:\models\fish_detector\best.pt"
python -m uvicorn tracking_service.app:app --host 127.0.0.1 --port 8765
```

Run the FinFrame web app on `http://127.0.0.1:4173`, open the same source video, and choose **Auto-track**.

Video is sent only to the loopback service on the same computer and is removed from its temporary directory after inference. Tracker proposals are imported with `verified: false`; they do not affect MaxN or released datasets until reviewed.

## Tracker choice

- **ByteTrack:** fast, simple and a strong first baseline when detector output is stable.
- **BoT-SORT:** worth testing when camera motion disrupts association. Its appearance ReID is optional and requires separate configuration and domain validation.
- **SAM 2:** preferable for propagating one or more student-prompted objects before a detector is available.

Evaluate trackers on held-out underwater footage using IDF1/HOTA, ID switches and annotation correction time. Human-scene benchmark rankings do not guarantee performance on schooling fish.

## Licensing

This optional service uses the `ultralytics` package, currently distributed under AGPL-3.0 with separate enterprise licensing available. Review its licence before distributing or hosting a combined product. FinFrame does not download or bundle model weights.

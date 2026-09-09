# Aerial Person Studio 1.0

A Windows desktop application for the complete **person-only aerial dataset** workflow used in this project:

**Raw aerial images/videos → frame extraction → quality filtering → duplicate filtering → source-safe split → CVAT packages → local YOLO pre-labeling → visual box review/editing → image deletion/background handling → YOLO/VisDrone/COCO/CVAT XML export + reports.**

The application is path-independent: it does not contain any hard-coded `D:\...` project path and can be moved to another Windows folder.

## Fastest start on Windows

1. Extract this folder somewhere with a reasonably short path, for example:
   `D:\AerialPersonStudio`
2. Double-click **START_HERE.bat**.
3. The setup script uses Python 3.12 and creates a short, stable runtime under `%LOCALAPPDATA%\AerialPersonStudio\venv-py312`, installs dependencies, and starts the GUI. This deliberately avoids Windows long-path failures with PyTorch and remains valid if the application folder is moved.
4. Put your trained model at:
   `models\prelabel\yolo26s_visdrone_best.pt`
   or browse to any `.pt` file from the Auto Label tab.

The first setup needs internet access to download Python packages. Later runs use **RUN_STUDIO.bat**.

## Continue the dataset you already prepared

You do **not** need to rebuild your current raw data.

1. Start Aerial Person Studio.
2. Click **Open Workspace**.
3. Select your existing project root, e.g. the folder that directly contains:
   `raw`, `work`, `models`, `outputs`, `run_pipeline.py`, etc.
4. In Dashboard click **Import Existing CVAT ZIPs into Editor**.
5. The app reads the current files under:
   - `work\cvat_upload\train_images.zip`
   - `work\cvat_upload\val_images.zip`
   - `work\cvat_upload\test_images.zip`
6. It creates a separate editable copy under `work\studio\images\...`. Original ZIPs are not modified.
7. Go to Auto Label, select your YOLO model, then run pre-labeling.
8. Go to Annotate & Review to correct boxes and delete unwanted images.

For the current cleaned dataset, if the ZIPs contain 89 Train / 20 Val / 15 Test images, Studio will import exactly those current ZIP contents.

## New dataset workflow

### Prepare Data

Use **Add Raw Images** and **Add Raw Videos**, then configure:

- video sampling interval
- minimum image size
- blur filtering
- near-black startup-frame filtering
- perceptual-hash near-duplicate removal
- source-aware Train / Validation / Test split

Then click **Run Complete Preparation Pipeline**.

The preparation pipeline creates:

- `work\standardized\`
- `work\manifest.csv`
- `work\cvat_upload\train_images.zip`
- `work\cvat_upload\val_images.zip`
- `work\cvat_upload\test_images.zip`
- `work\studio\images\train|val|test`
- preparation rejection report

Frames from the same video/source are kept together during splitting to reduce leakage.

## Auto Label

The Auto Label tab supports:

- any Ultralytics `.pt` detector
- automatic single-class → `person` mapping
- confidence threshold
- IoU threshold
- image size
- maximum detections
- CPU / GPU device selection
- optional **Full-frame + tiled inference** for tiny aerial persons

If the model stores its only class as `{0: 'item'}`, the app intentionally maps that class to the canonical final label `person`.

Model predictions are pre-labels, not guaranteed ground truth. Review them before final export.

## Annotation editor

The built-in editor supports:

- selecting boxes
- dragging boxes to move them
- resizing boxes with corner/edge handles
- drawing new person boxes
- deleting selected boxes
- marking an image as background (zero boxes)
- deleting an entire image (moved to `work\studio\trash\...`)
- mouse-wheel zoom
- Previous / Next navigation
- keyboard shortcuts:
  - `Delete`: delete selected box
  - `N`: next image
  - `P`: previous image
  - `Ctrl+S`: save boxes

All editing affects the editable Studio dataset only. This protects the original prepared packages.

## Export formats

The Export tab can create all of the following from the reviewed dataset:

### YOLO
`outputs\yolo\`

Single class:
`0: person`

### VisDrone-style master format
`outputs\visdrone\`

Annotation rows:
`bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,truncation,occlusion`

For this custom person-only dataset:
- `object_category = 1`
- `score = 1`
- default truncation = 0
- default occlusion = 0

### COCO
`outputs\coco\train.json`, `val.json`, `test.json`

### CVAT for images 1.1 XML
`outputs\cvat_xml\train.xml`, `val.xml`, `test.xml`

### Reports
`outputs\reports\dataset_report.csv`
`outputs\reports\dataset_summary.json`

## Windows / Python

The supported setup path is Python **3.12 64-bit**. The runtime is intentionally stored under `%LOCALAPPDATA%\AerialPersonStudio\venv-py312` instead of inside the project folder. This avoids PyTorch `WinError 206` long-path failures and prevents a moved/renamed project folder from breaking `pip`.

The application itself can run on CPU. For CUDA inference, install a CUDA-enabled PyTorch build compatible with your NVIDIA driver/GPU in the Studio runtime; then choose `auto` or `0` in Auto Label.

## Build a Windows executable folder

After a successful normal setup, double-click:

`BUILD_WINDOWS_EXE.bat`

PyInstaller will attempt to build:

`dist\AerialPersonStudio\AerialPersonStudio.exe`

Because PyTorch, Ultralytics and Qt are large frameworks, the portable build can be large. The normal `RUN_STUDIO.bat` mode is usually easier to maintain and update.

## Safety / recoverability

- Original CVAT ZIP imports are not edited.
- Deleted editor images are moved to `work\studio\trash` rather than immediately destroyed.
- Exports are regenerated into `outputs`.
- The app uses relative workspace data paths for new projects, improving portability when a project folder is moved.

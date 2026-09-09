from __future__ import annotations
import os
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import QThread, Qt, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QSpinBox, QSplitter,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget
)

from ..autolabel import run_autolabel
from ..config import AppConfig, AutoLabelConfig, PrepareConfig
from ..dataset import StudioDataset
from ..exporters import export_all, export_coco, export_cvat_xml, export_report, export_visdrone, export_yolo
from ..prepare import prepare_workspace
from .annotator import AnnotationView
from .theme import DARK_QSS
from .worker import FunctionWorker


class MainWindow(QMainWindow):
    def __init__(self, app_root: Path):
        super().__init__()
        self.app_root = Path(app_root).resolve()
        self.config_path = self.app_root / "studio_config.json"
        self.config = AppConfig.load(self.config_path)
        self.workspace = (self.app_root / "workspace").resolve()
        self.ds = StudioDataset(self.workspace)
        self.worker_thread = None
        self.worker = None
        self._worker_on_done = None
        self.current_split = "train"
        self.current_name = None
        self._loading_image = False

        self.setWindowTitle("Aerial Person Studio")
        self.resize(1500, 940)
        self.setMinimumSize(1180, 760)
        self.setStyleSheet(DARK_QSS)
        self._build_ui()
        self._install_shortcuts()
        self.refresh_everything()

    # ---------- helpers ----------
    def card(self):
        f = QFrame(); f.setObjectName("Card")
        return f

    def secondary(self, text):
        b = QPushButton(text); b.setProperty("secondary", True)
        return b

    def danger(self, text):
        b = QPushButton(text); b.setProperty("danger", True)
        return b

    def log(self, text: str):
        self.log_view.append(str(text))

    def open_folder(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(path)  # noqa
        elif sys.platform == "darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')

    def _build_ui(self):
        central = QWidget(); root = QVBoxLayout(central); root.setContentsMargins(18, 16, 18, 18); root.setSpacing(12)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        title = QLabel("Aerial Person Studio"); title.setObjectName("Title")
        subtitle = QLabel("Prepare • Auto-label • Review • Correct • Export — person-only aerial dataset workflow")
        subtitle.setObjectName("Subtitle")
        titles.addWidget(title); titles.addWidget(subtitle)
        header.addLayout(titles, 1)
        self.workspace_edit = QLineEdit(str(self.workspace)); self.workspace_edit.setMinimumWidth(500)
        browse = self.secondary("Open Workspace")
        browse.clicked.connect(self.choose_workspace)
        header.addWidget(QLabel("Workspace")); header.addWidget(self.workspace_edit); header.addWidget(browse)
        root.addLayout(header)

        self.tabs = QTabWidget()
        self.dashboard_tab = self._build_dashboard_tab()
        self.prepare_tab = self._build_prepare_tab()
        self.auto_tab = self._build_auto_tab()
        self.annotate_tab = self._build_annotate_tab()
        self.export_tab = self._build_export_tab()
        self.logs_tab = self._build_logs_tab()
        self.tabs.addTab(self.dashboard_tab, "Dashboard")
        self.tabs.addTab(self.prepare_tab, "Prepare Data")
        self.tabs.addTab(self.auto_tab, "Auto Label")
        self.tabs.addTab(self.annotate_tab, "Annotate & Review")
        self.tabs.addTab(self.export_tab, "Export")
        self.tabs.addTab(self.logs_tab, "Logs")
        root.addWidget(self.tabs, 1)

        status = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0); self.progress.setMaximumWidth(420)
        status.addWidget(self.status_label, 1); status.addWidget(self.progress)
        root.addLayout(status)
        self.setCentralWidget(central)

    def _build_dashboard_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        grid = QGridLayout(); grid.setSpacing(12)
        self.metric_images = QLabel("0"); self.metric_boxes = QLabel("0"); self.metric_train = QLabel("0"); self.metric_valtest = QLabel("0")
        for x in [self.metric_images, self.metric_boxes, self.metric_train, self.metric_valtest]: x.setObjectName("Metric")
        metrics = [("Total images", self.metric_images), ("Total person boxes", self.metric_boxes), ("Train images", self.metric_train), ("Val + Test", self.metric_valtest)]
        for i,(name,label) in enumerate(metrics):
            c=self.card(); cl=QVBoxLayout(c); cl.addWidget(QLabel(name)); cl.addWidget(label); grid.addWidget(c,0,i)
        lay.addLayout(grid)

        c=self.card(); cl=QVBoxLayout(c)
        cl.addWidget(QLabel("Start or resume a project"))
        info=QLabel("Use an existing workspace produced by the earlier pipeline, or create a new workspace. The app never hard-codes a drive/path.")
        info.setWordWrap(True); info.setObjectName("Subtitle"); cl.addWidget(info)
        buttons=QHBoxLayout()
        imp=QPushButton("Import Existing CVAT ZIPs into Editor")
        imp.clicked.connect(self.import_existing_zips)
        openraw=self.secondary("Open Raw Folder"); openraw.clicked.connect(lambda:self.open_folder(self.ds.paths.raw))
        openwork=self.secondary("Open Work Folder"); openwork.clicked.connect(lambda:self.open_folder(self.ds.paths.work))
        refresh=self.secondary("Refresh Status"); refresh.clicked.connect(self.refresh_everything)
        buttons.addWidget(imp); buttons.addWidget(openraw); buttons.addWidget(openwork); buttons.addWidget(refresh); buttons.addStretch()
        cl.addLayout(buttons); lay.addWidget(c)

        c2=self.card(); c2l=QVBoxLayout(c2)
        c2l.addWidget(QLabel("Current workflow"))
        flow=QLabel("1) Add raw images/videos → 2) Prepare & split → 3) Run your trained YOLO model → 4) Review/edit/delete boxes or images → 5) Export YOLO / VisDrone / COCO / CVAT XML + reports")
        flow.setWordWrap(True); flow.setObjectName("Subtitle"); c2l.addWidget(flow); lay.addWidget(c2)
        lay.addStretch()
        return w

    def _build_prepare_tab(self):
        w=QWidget(); lay=QVBoxLayout(w)
        top=QHBoxLayout()
        addimg=QPushButton("Add Raw Images"); addimg.clicked.connect(self.add_raw_images)
        addvid=QPushButton("Add Raw Videos"); addvid.clicked.connect(self.add_raw_videos)
        raw=self.secondary("Open Raw Media"); raw.clicked.connect(lambda:self.open_folder(self.ds.paths.raw))
        top.addWidget(addimg); top.addWidget(addvid); top.addWidget(raw); top.addStretch(); lay.addLayout(top)

        group=QGroupBox("Preparation settings"); form=QFormLayout(group)
        p=self.config.prepare
        self.frame_interval=QDoubleSpinBox(); self.frame_interval.setRange(.1,60); self.frame_interval.setValue(p.video_frame_interval_seconds); self.frame_interval.setSuffix(" s")
        self.min_w=QSpinBox(); self.min_w.setRange(32,10000); self.min_w.setValue(p.min_width)
        self.min_h=QSpinBox(); self.min_h.setRange(32,10000); self.min_h.setValue(p.min_height)
        self.blur_enabled=QCheckBox("Reject blurry frames"); self.blur_enabled.setChecked(p.blur_filter_enabled)
        self.blur_threshold=QDoubleSpinBox(); self.blur_threshold.setRange(0,10000); self.blur_threshold.setValue(p.blur_threshold)
        self.black_enabled=QCheckBox("Reject near-black/empty startup frames"); self.black_enabled.setChecked(p.black_filter_enabled)
        self.black_mean=QDoubleSpinBox(); self.black_mean.setRange(0,255); self.black_mean.setValue(p.black_mean_threshold)
        self.dup_enabled=QCheckBox("Remove near-duplicate frames"); self.dup_enabled.setChecked(p.duplicate_filter_enabled)
        self.phash=QSpinBox(); self.phash.setRange(0,30); self.phash.setValue(p.phash_hamming_threshold)
        self.train_ratio=QDoubleSpinBox(); self.train_ratio.setRange(.1,.98); self.train_ratio.setSingleStep(.05); self.train_ratio.setValue(p.train_ratio)
        self.val_ratio=QDoubleSpinBox(); self.val_ratio.setRange(.01,.49); self.val_ratio.setSingleStep(.05); self.val_ratio.setValue(p.val_ratio)
        self.test_ratio=QDoubleSpinBox(); self.test_ratio.setRange(.01,.49); self.test_ratio.setSingleStep(.05); self.test_ratio.setValue(p.test_ratio)
        form.addRow("Video sampling interval", self.frame_interval); form.addRow("Minimum width", self.min_w); form.addRow("Minimum height", self.min_h)
        form.addRow(self.blur_enabled); form.addRow("Blur threshold", self.blur_threshold); form.addRow(self.black_enabled); form.addRow("Black mean threshold", self.black_mean)
        form.addRow(self.dup_enabled); form.addRow("pHash Hamming threshold", self.phash); form.addRow("Train ratio", self.train_ratio); form.addRow("Validation ratio", self.val_ratio); form.addRow("Test ratio", self.test_ratio)
        lay.addWidget(group)
        note=QLabel("Split is source-aware: frames from the same video/source stay together to reduce leakage. Empty but valid background images can remain; near-black startup frames can be filtered automatically.")
        note.setWordWrap(True); note.setObjectName("Subtitle"); lay.addWidget(note)
        run=QPushButton("Run Complete Preparation Pipeline"); run.clicked.connect(self.run_prepare); lay.addWidget(run)
        lay.addStretch(); return w

    def _build_auto_tab(self):
        w=QWidget(); lay=QVBoxLayout(w)
        group=QGroupBox("Model-assisted person pre-labeling"); form=QFormLayout(group)
        a=self.config.autolabel
        model_row=QHBoxLayout(); self.model_edit=QLineEdit(a.model_path); browse=self.secondary("Browse .pt"); browse.clicked.connect(self.choose_model); model_row.addWidget(self.model_edit,1); model_row.addWidget(browse)
        form.addRow("YOLO model", model_row)
        self.auto_split=QComboBox(); self.auto_split.addItems(["train","val","test"])
        self.conf=QDoubleSpinBox(); self.conf.setRange(.001,.999); self.conf.setDecimals(3); self.conf.setSingleStep(.01); self.conf.setValue(a.confidence)
        self.iou=QDoubleSpinBox(); self.iou.setRange(.1,.95); self.iou.setDecimals(2); self.iou.setValue(a.iou)
        self.imgsz=QSpinBox(); self.imgsz.setRange(320,4096); self.imgsz.setSingleStep(32); self.imgsz.setValue(a.imgsz)
        self.max_det=QSpinBox(); self.max_det.setRange(10,10000); self.max_det.setValue(a.max_det)
        self.device=QComboBox(); self.device.setEditable(True); self.device.addItems(["auto","cpu","0"]); self.device.setCurrentText(a.device)
        self.tiled=QCheckBox("Full-frame + tiled inference for tiny aerial persons"); self.tiled.setChecked(a.tiled_inference)
        self.tile_size=QSpinBox(); self.tile_size.setRange(320,2048); self.tile_size.setSingleStep(32); self.tile_size.setValue(a.tile_size)
        self.tile_overlap=QDoubleSpinBox(); self.tile_overlap.setRange(.0,.75); self.tile_overlap.setDecimals(2); self.tile_overlap.setSingleStep(.05); self.tile_overlap.setValue(a.tile_overlap)
        form.addRow("Split", self.auto_split); form.addRow("Confidence", self.conf); form.addRow("IoU", self.iou); form.addRow("Inference image size", self.imgsz); form.addRow("Maximum detections", self.max_det); form.addRow("Device", self.device); form.addRow(self.tiled); form.addRow("Tile size", self.tile_size); form.addRow("Tile overlap", self.tile_overlap)
        lay.addWidget(group)
        note=QLabel("If the model has exactly one class (even if stored as 'item'), the app maps it to the canonical label 'person'. Predictions remain editable and should be human-reviewed before final export.")
        note.setWordWrap(True); note.setObjectName("Subtitle"); lay.addWidget(note)
        buttons=QHBoxLayout(); run=QPushButton("Run Auto Label on Selected Split"); run.clicked.connect(self.run_auto_label); clear=self.danger("Clear Split Annotations"); clear.clicked.connect(self.clear_split_annotations); buttons.addWidget(run); buttons.addWidget(clear); buttons.addStretch(); lay.addLayout(buttons)
        lay.addStretch(); return w

    def _build_annotate_tab(self):
        w=QWidget(); outer=QVBoxLayout(w)
        toolbar=QHBoxLayout()
        self.ann_split=QComboBox(); self.ann_split.addItems(["train","val","test"]); self.ann_split.currentTextChanged.connect(self.change_annotation_split)
        prev=self.secondary("◀ Previous"); prev.clicked.connect(lambda:self.step_image(-1))
        nxt=self.secondary("Next ▶"); nxt.clicked.connect(lambda:self.step_image(1))
        self.draw_btn=QPushButton("Draw Person Box"); self.draw_btn.setCheckable(True); self.draw_btn.toggled.connect(self.annotation_view_set_draw_mode)
        save=QPushButton("Save Boxes"); save.clicked.connect(self.save_current_boxes)
        delete_box=self.danger("Delete Selected Box"); delete_box.clicked.connect(self.delete_selected_box)
        background=self.secondary("Mark as Background (0 boxes)"); background.clicked.connect(self.mark_background)
        delete_img=self.danger("Delete Image → Trash"); delete_img.clicked.connect(self.delete_current_image)
        toolbar.addWidget(QLabel("Split")); toolbar.addWidget(self.ann_split); toolbar.addWidget(prev); toolbar.addWidget(nxt); toolbar.addWidget(self.draw_btn); toolbar.addWidget(save); toolbar.addWidget(delete_box); toolbar.addWidget(background); toolbar.addWidget(delete_img); toolbar.addStretch()
        outer.addLayout(toolbar)

        splitter=QSplitter(); self.image_list=QListWidget(); self.image_list.setMinimumWidth(250); self.image_list.currentTextChanged.connect(self.load_selected_image)
        self.annotation_view=AnnotationView(); self.annotation_view.boxesChanged.connect(self.update_annotation_info)
        right=QWidget(); rl=QVBoxLayout(right); self.image_info=QLabel("No image selected"); self.image_info.setObjectName("Subtitle"); rl.addWidget(self.image_info); rl.addWidget(self.annotation_view,1)
        splitter.addWidget(self.image_list); splitter.addWidget(right); splitter.setStretchFactor(1,1); outer.addWidget(splitter,1)
        helptext=QLabel("Mouse: click box to select • drag box to move • drag yellow handles to resize • Draw Person Box to add • mouse wheel to zoom • Delete key removes selected boxes • N/P navigate.")
        helptext.setObjectName("Subtitle"); helptext.setWordWrap(True); outer.addWidget(helptext)
        return w

    def _build_export_tab(self):
        w=QWidget(); lay=QVBoxLayout(w)
        note=QLabel("Exports always use the current editable Studio dataset, including image deletions and your latest box corrections.")
        note.setWordWrap(True); note.setObjectName("Subtitle"); lay.addWidget(note)
        grid=QGridLayout();
        actions=[("Export YOLO Dataset",lambda:self.do_export("yolo")),("Export VisDrone Master",lambda:self.do_export("visdrone")),("Export COCO",lambda:self.do_export("coco")),("Export CVAT XML 1.1",lambda:self.do_export("cvat")),("Generate Reports",lambda:self.do_export("report")),("Export Everything",lambda:self.do_export("all"))]
        for i,(text,fn) in enumerate(actions): b=QPushButton(text); b.clicked.connect(fn); grid.addWidget(b,i//2,i%2)
        lay.addLayout(grid)
        openout=self.secondary("Open Outputs Folder"); openout.clicked.connect(lambda:self.open_folder(self.ds.paths.outputs)); lay.addWidget(openout)
        self.export_summary=QTextEdit(); self.export_summary.setReadOnly(True); lay.addWidget(self.export_summary,1)
        return w

    def _build_logs_tab(self):
        w=QWidget(); lay=QVBoxLayout(w); self.log_view=QTextEdit(); self.log_view.setReadOnly(True); lay.addWidget(self.log_view); return w

    def _install_shortcuts(self):
        act=QAction(self); act.setShortcut(QKeySequence(Qt.Key.Key_Delete)); act.triggered.connect(self.delete_selected_box); self.addAction(act)
        n=QAction(self); n.setShortcut(QKeySequence("N")); n.triggered.connect(lambda:self.step_image(1)); self.addAction(n)
        p=QAction(self); p.setShortcut(QKeySequence("P")); p.triggered.connect(lambda:self.step_image(-1)); self.addAction(p)
        s=QAction(self); s.setShortcut(QKeySequence("Ctrl+S")); s.triggered.connect(self.save_current_boxes); self.addAction(s)

    # ---------- workspace ----------
    def choose_workspace(self):
        folder=QFileDialog.getExistingDirectory(self,"Select workspace",str(self.workspace))
        if folder:
            self.save_current_boxes(silent=True)
            self.workspace=Path(folder).resolve(); self.workspace_edit.setText(str(self.workspace)); self.ds=StudioDataset(self.workspace); self.refresh_everything(); self.log(f"Workspace: {self.workspace}")

    def refresh_everything(self):
        self.ds=StudioDataset(self.workspace)
        st=self.ds.stats(); self.metric_images.setText(str(st["total_images"])); self.metric_boxes.setText(str(st["total_boxes"])); self.metric_train.setText(str(st["train"]["images"])); self.metric_valtest.setText(str(st["val"]["images"]+st["test"]["images"]))
        self.refresh_image_list(keep_selection=True)
        self.status_label.setText(f"Workspace ready • {st['total_images']} images • {st['total_boxes']} boxes")

    def import_existing_zips(self):
        if not any((self.ds.paths.cvat_upload/f"{s}_images.zip").exists() for s in ("train","val","test")):
            QMessageBox.warning(self,"No packages","No train_images.zip / val_images.zip / test_images.zip found under work/cvat_upload."); return
        if QMessageBox.question(self,"Import prepared packages","This will rebuild the editable Studio copy from the current CVAT ZIPs. Original ZIPs are not modified. Continue?") != QMessageBox.StandardButton.Yes: return
        self.run_background("Importing prepared CVAT ZIPs", lambda prog,log:self.ds.import_existing_cvat_zips(True, lambda s,i,n,name:prog(i,n,f"{s}: {name}")), self._after_import)

    def _after_import(self, result):
        self.log(f"Imported {result} prepared images into editable Studio dataset."); self.refresh_everything(); self.tabs.setCurrentWidget(self.annotate_tab)

    # ---------- raw/prepare ----------
    def add_raw_images(self):
        files,_=QFileDialog.getOpenFileNames(self,"Add aerial images",str(Path.home()),"Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)")
        if not files:return
        self.ds.paths.raw_images.mkdir(parents=True,exist_ok=True)
        for f in files: shutil.copy2(f,self.ds.paths.raw_images/Path(f).name)
        self.log(f"Added {len(files)} raw images.")

    def add_raw_videos(self):
        files,_=QFileDialog.getOpenFileNames(self,"Add aerial videos",str(Path.home()),"Videos (*.mp4 *.avi *.mov *.mkv *.m4v *.webm)")
        if not files:return
        self.ds.paths.raw_videos.mkdir(parents=True,exist_ok=True)
        for f in files: shutil.copy2(f,self.ds.paths.raw_videos/Path(f).name)
        self.log(f"Added {len(files)} raw videos.")

    def prepare_cfg(self):
        ratios=self.train_ratio.value()+self.val_ratio.value()+self.test_ratio.value()
        if abs(ratios-1.0)>.02: raise ValueError(f"Train/Val/Test ratios must sum to ~1.0; current total={ratios:.3f}")
        return PrepareConfig(video_frame_interval_seconds=self.frame_interval.value(),min_width=self.min_w.value(),min_height=self.min_h.value(),blur_filter_enabled=self.blur_enabled.isChecked(),blur_threshold=self.blur_threshold.value(),black_filter_enabled=self.black_enabled.isChecked(),black_mean_threshold=self.black_mean.value(),duplicate_filter_enabled=self.dup_enabled.isChecked(),phash_hamming_threshold=self.phash.value(),train_ratio=self.train_ratio.value(),val_ratio=self.val_ratio.value(),test_ratio=self.test_ratio.value())

    def run_prepare(self):
        try: cfg=self.prepare_cfg()
        except Exception as e: QMessageBox.warning(self,"Invalid settings",str(e)); return
        if QMessageBox.question(self,"Rebuild prepared data","Preparation rebuilds work/candidates, work/standardized, CVAT ZIPs and the editable Studio dataset. Raw media is preserved. Continue?") != QMessageBox.StandardButton.Yes:return
        self.config.prepare=cfg; self.config.save(self.config_path)
        self.run_background("Preparing dataset",lambda prog,log:prepare_workspace(self.workspace,cfg,lambda stage,i,n,name:prog(i,n,f"{stage}: {name}"),log),self._after_prepare)

    def _after_prepare(self,result): self.log(f"Preparation result: {result}"); self.refresh_everything(); QMessageBox.information(self,"Preparation complete",str(result))

    # ---------- auto label ----------
    def choose_model(self):
        start=str(self.app_root/"models"/"prelabel")
        f,_=QFileDialog.getOpenFileName(self,"Select trained YOLO model",start,"PyTorch model (*.pt)")
        if f:self.model_edit.setText(f)

    def auto_cfg(self):
        return AutoLabelConfig(model_path=self.model_edit.text().strip(),confidence=self.conf.value(),iou=self.iou.value(),imgsz=self.imgsz.value(),max_det=self.max_det.value(),device=self.device.currentText().strip() or "auto",tiled_inference=self.tiled.isChecked(),tile_size=self.tile_size.value(),tile_overlap=self.tile_overlap.value())

    def run_auto_label(self):
        cfg=self.auto_cfg(); split=self.auto_split.currentText(); self.config.autolabel=cfg; self.config.save(self.config_path)
        if not self.ds.records(split): QMessageBox.warning(self,"No images",f"No Studio images for {split}. Import existing CVAT ZIPs or run Prepare first."); return
        if QMessageBox.question(self,"Run model",f"Run model-assisted pre-labeling on all {len(self.ds.records(split))} {split} images? Existing annotations in this split will be replaced.") != QMessageBox.StandardButton.Yes:return
        self.run_background(f"Auto-labeling {split}",lambda prog,log:run_autolabel(self.workspace,split,cfg,lambda i,n,name:prog(i,n,name),log,True),lambda result:self._after_auto(split,result))

    def _after_auto(self,split,result):
        self.log(f"Auto-label complete for {split}: {result}"); self.refresh_everything(); self.ann_split.setCurrentText(split); self.tabs.setCurrentWidget(self.annotate_tab); QMessageBox.information(self,"Auto label complete",f"{split}: {result}\nReview and correct predictions before export.")

    def clear_split_annotations(self):
        split=self.auto_split.currentText()
        if QMessageBox.question(self,"Clear annotations",f"Delete all annotations for {split}? Images remain.") == QMessageBox.StandardButton.Yes:
            self.ds.clear_annotations(split); self.refresh_everything()

    # ---------- annotation ----------
    def refresh_image_list(self, keep_selection=False):
        if not hasattr(self,"image_list"):return
        old=self.image_list.currentItem().text() if keep_selection and self.image_list.currentItem() else None
        split=self.ann_split.currentText() if hasattr(self,"ann_split") else "train"
        self.current_split=split
        self.image_list.blockSignals(True); self.image_list.clear()
        for rec in self.ds.records(split): self.image_list.addItem(rec["name"])
        self.image_list.blockSignals(False)
        if self.image_list.count():
            matches=self.image_list.findItems(old,Qt.MatchFlag.MatchExactly) if old else []
            self.image_list.setCurrentItem(matches[0]) if matches else self.image_list.setCurrentRow(0)

    def change_annotation_split(self,split):
        self.save_current_boxes(silent=True); self.current_split=split; self.current_name=None; self.refresh_image_list(False)

    def load_selected_image(self,name):
        if not name:return
        if self.current_name and not self._loading_image:self.save_current_boxes(silent=True)
        self._loading_image=True
        try:
            rec=self.ds.record(self.current_split,name)
            if not rec:return
            self.annotation_view.load_image(self.ds.image_path(rec),self.ds.boxes(self.current_split,name)); self.current_name=name; self.update_annotation_info()
        except Exception as e: QMessageBox.critical(self,"Image error",str(e))
        finally:self._loading_image=False

    def update_annotation_info(self):
        if not self.current_name:return
        rec=self.ds.record(self.current_split,self.current_name); count=len(self.annotation_view.box_items())
        self.image_info.setText(f"{self.current_split.upper()} • {self.current_name} • {rec['width']}×{rec['height']} • {count} person box(es)")

    def annotation_view_set_draw_mode(self,enabled): self.annotation_view.set_draw_mode(enabled); self.draw_btn.setText("Drawing… click-drag on image" if enabled else "Draw Person Box")

    def save_current_boxes(self,silent=False):
        if not self.current_name:return
        self.ds.set_boxes(self.current_split,self.current_name,self.annotation_view.current_boxes()); self.update_annotation_info()
        if not silent:self.status_label.setText(f"Saved {self.current_split}/{self.current_name}")

    def delete_selected_box(self):
        if not hasattr(self,"annotation_view"):return
        if self.annotation_view.delete_selected_boxes(): self.save_current_boxes(silent=True)

    def mark_background(self):
        if not self.current_name:return
        if QMessageBox.question(self,"Mark as background","Remove every box from this image and keep the image as a valid negative/background sample?") == QMessageBox.StandardButton.Yes:
            rec=self.ds.record(self.current_split,self.current_name); self.annotation_view.load_image(self.ds.image_path(rec),[]); self.ds.set_boxes(self.current_split,self.current_name,[]); self.update_annotation_info()

    def delete_current_image(self):
        if not self.current_name:return
        name=self.current_name
        if QMessageBox.question(self,"Delete image",f"Move {name} to work/studio/trash/{self.current_split}/ and remove it from the editable dataset?") != QMessageBox.StandardButton.Yes:return
        self.ds.delete_image(self.current_split,name); self.current_name=None; self.refresh_everything()

    def step_image(self,delta):
        if not hasattr(self,"image_list") or not self.image_list.count():return
        row=max(0,min(self.image_list.count()-1,self.image_list.currentRow()+delta)); self.image_list.setCurrentRow(row)

    # ---------- export ----------
    def do_export(self,kind):
        self.save_current_boxes(silent=True)
        try:
            if kind=="yolo": result=export_yolo(self.workspace)
            elif kind=="visdrone": result=export_visdrone(self.workspace)
            elif kind=="coco": result=export_coco(self.workspace)
            elif kind=="cvat": result=export_cvat_xml(self.workspace)
            elif kind=="report": result=export_report(self.workspace)
            else: result=export_all(self.workspace)
            self.export_summary.append(f"{kind.upper()}\n{result}\n"); self.log(f"Exported {kind}: {result}")
        except Exception as e: QMessageBox.critical(self,"Export failed",str(e))

    # ---------- background worker ----------
    def run_background(self,title,fn,on_done):
        if self.worker_thread and self.worker_thread.isRunning():
            QMessageBox.warning(self,"Busy","Another operation is still running.")
            return

        self.status_label.setText(title)
        self.progress.setValue(0)
        self._worker_on_done = on_done

        # IMPORTANT:
        # The QThread and worker are separate QObjects. UI completion/error
        # handlers are connected directly to QObject slots on MainWindow so
        # PySide always queues them back to the GUI thread. The previous v1.0
        # used a Python lambda for the finished callback; depending on the
        # PySide version that callback could execute in the worker thread and
        # then touch QWidget/QObject instances, producing:
        #   QObject::setParent: Cannot set parent, new parent is in a different thread
        thread = QThread()
        worker = FunctionWorker(fn)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress, Qt.ConnectionType.QueuedConnection)
        worker.log.connect(self.log, Qt.ConnectionType.QueuedConnection)
        worker.finished.connect(self._worker_finished_slot, Qt.ConnectionType.QueuedConnection)
        worker.failed.connect(self._worker_failed_slot, Qt.ConnectionType.QueuedConnection)

        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._worker_thread_finished, Qt.ConnectionType.QueuedConnection)
        thread.finished.connect(thread.deleteLater)

        self.worker_thread = thread
        self.worker = worker
        thread.start()

    @Slot(int, int, str)
    def _on_progress(self,i,n,text):
        pct=int(100*i/max(1,n))
        self.progress.setValue(pct)
        self.status_label.setText(f"{text}  ({i}/{n})")

    @Slot(object)
    def _worker_finished_slot(self,result):
        # Guaranteed to execute on the MainWindow/GUI thread.
        self.progress.setValue(100)
        self.status_label.setText("Complete")
        callback = self._worker_on_done
        if callback is not None:
            try:
                callback(result)
            except Exception as exc:
                self.log(f"Completion callback error: {exc}")
                QMessageBox.critical(self, "Completion error", str(exc))

    @Slot(str)
    def _worker_failed_slot(self,trace):
        # Guaranteed to execute on the MainWindow/GUI thread.
        self.log(trace)
        self.progress.setValue(0)
        self.status_label.setText("Operation failed")
        QMessageBox.critical(self,"Operation failed",trace[-5000:])

    @Slot()
    def _worker_thread_finished(self):
        self.worker = None
        self.worker_thread = None
        self._worker_on_done = None

    def closeEvent(self,event):
        self.save_current_boxes(silent=True); event.accept()

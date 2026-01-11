"""
DICOM Explorer - Image Browser with Annotation Matching

A GUI tool for browsing DICOM images and matching them with CSV annotations.
Supports multiple CSV files, flexible column mapping, and various filtering options.
"""

import sys
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

# Import pydicom and register decompression handlers
import pydicom
try:
    # pylibjpeg registers itself as a pydicom handler on import
    import pylibjpeg  # noqa: F401
    try:
        from pylibjpeg import libjpeg  # noqa: F401 - JPEG Lossless support
    except ImportError:
        pass
    try:
        from pylibjpeg import openjpeg  # noqa: F401 - JPEG 2000 support
    except ImportError:
        pass
except ImportError:
    pass
try:
    import gdcm  # noqa: F401 - python-gdcm for additional decompression support
except ImportError:
    pass

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QComboBox, QSpinBox,
    QCheckBox, QGroupBox, QFileDialog, QProgressBar, QTextEdit,
    QTabWidget, QFormLayout, QFrame, QSplitter, QMessageBox,
    QStatusBar, QMenuBar, QMenu, QScrollArea, QSizePolicy,
    QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QTreeWidget, QTreeWidgetItem
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QFont, QIcon, QAction, QPalette, QColor, QPixmap, QImage, QPainter, QPen


# Custom widgets that always ignore scroll wheel events
class NoScrollComboBox(QComboBox):
    """ComboBox that always ignores scroll wheel events."""
    def wheelEvent(self, event):
        event.ignore()


class ClickableLabel(QLabel):
    """QLabel that emits a signal on double-click."""
    doubleClicked = pyqtSignal()
    
    def mouseDoubleClickEvent(self, event):
        self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)


@dataclass
class DicomFileInfo:
    """Information about a DICOM file."""
    path: Path
    filename: str
    sop_instance_uid: Optional[str] = None
    patient_id: Optional[str] = None
    study_id: Optional[str] = None
    series_id: Optional[str] = None
    modality: Optional[str] = None
    rows: Optional[int] = None
    cols: Optional[int] = None
    annotations: List[Dict[str, Any]] = field(default_factory=list)
    matched: bool = False


class ScanThread(QThread):
    """Background thread for scanning DICOM files."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    log = pyqtSignal(str)
    
    # Directories to skip during scanning
    SKIP_DIRS = {'.venv', 'venv', '.git', '__pycache__', 'node_modules', '.tox', 
                 '.pytest_cache', '.mypy_cache', 'dist', 'build', 'egg-info',
                 'site-packages', '.conda', 'anaconda', 'miniconda'}
    
    def __init__(self, folder: Path):
        super().__init__()
        self.folder = folder
    
    def should_skip_dir(self, path: Path) -> bool:
        """Check if directory should be skipped."""
        parts = path.parts
        return any(part in self.SKIP_DIRS for part in parts)
    
    def scan_for_files(self, pattern: str):
        """Scan for files matching pattern, skipping unwanted directories."""
        for f in self.folder.rglob(pattern):
            if not self.should_skip_dir(f):
                yield f
    
    def run(self):
        try:
            files = []
            
            # First, look for common DICOM extensions
            dicom_extensions = ['*.dcm', '*.DCM', '*.dicom', '*.DICOM', '*.ima', '*.IMA']
            dicom_candidates = []
            
            for ext in dicom_extensions:
                for f in self.scan_for_files(ext):
                    dicom_candidates.append(f)
            
            # Also look for files without extension (common in DICOM)
            for f in self.scan_for_files("*"):
                if f.is_file() and f.suffix == '' and f not in dicom_candidates:
                    dicom_candidates.append(f)
            
            # Remove duplicates
            dicom_candidates = list(set(dicom_candidates))
            
            self.log.emit(f"Found {len(dicom_candidates)} candidate files to scan...")
            
            if len(dicom_candidates) == 0:
                self.log.emit("No DICOM candidates found. Scanning all files...")
                # Fall back to scanning all files (but skip unwanted dirs)
                dicom_candidates = [f for f in self.scan_for_files("*") if f.is_file()]
                # Limit to avoid scanning too many files
                if len(dicom_candidates) > 10000:
                    self.log.emit(f"Too many files ({len(dicom_candidates)}), limiting to 10000")
                    dicom_candidates = dicom_candidates[:10000]
            
            total = len(dicom_candidates)
            self.log.emit(f"Scanning {total} files...")
            
            for i, file_path in enumerate(dicom_candidates):
                if i % 50 == 0:
                    pct = int((i / total) * 100) if total > 0 else 0
                    self.progress.emit(pct, f"Scanning {i}/{total}: {file_path.name}")
                
                try:
                    # Try to read as DICOM
                    ds = pydicom.dcmread(file_path, stop_before_pixels=True, force=True)
                    
                    # Check if it's a valid DICOM - must have at least one of these
                    is_valid = (
                        hasattr(ds, 'SOPInstanceUID') or 
                        hasattr(ds, 'Modality') or
                        hasattr(ds, 'PatientID') or
                        hasattr(ds, 'Rows')
                    )
                    
                    if is_valid:
                        info = DicomFileInfo(
                            path=file_path,
                            filename=file_path.stem,
                            sop_instance_uid=str(ds.SOPInstanceUID) if hasattr(ds, 'SOPInstanceUID') else None,
                            patient_id=str(ds.PatientID) if hasattr(ds, 'PatientID') else None,
                            study_id=str(ds.StudyInstanceUID) if hasattr(ds, 'StudyInstanceUID') else None,
                            series_id=str(ds.SeriesInstanceUID) if hasattr(ds, 'SeriesInstanceUID') else None,
                            modality=str(ds.Modality) if hasattr(ds, 'Modality') else None,
                            rows=int(ds.Rows) if hasattr(ds, 'Rows') else None,
                            cols=int(ds.Columns) if hasattr(ds, 'Columns') else None,
                        )
                        files.append(info)
                except Exception:
                    # Not a DICOM file, skip
                    pass
            
            self.log.emit(f"Found {len(files)} valid DICOM files")
            self.progress.emit(100, f"Scan complete - found {len(files)} DICOM files")
            self.finished.emit(files)
            
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")


class FilterThread(QThread):
    """Background thread for filtering DICOM files by CSV data."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(list, int)  # (dicom_files, matched_count)
    error = pyqtSignal(str)
    
    def __init__(self, dicom_files: list, combined_csv, id_col: str, filter_col: str, filter_val: str):
        super().__init__()
        self.dicom_files = dicom_files
        self.combined_csv = combined_csv
        self.id_col = id_col
        self.filter_col = filter_col
        self.filter_val = filter_val
    
    def run(self):
        try:
            # Find image IDs that match the filter
            self.progress.emit(0, "Building filter index...")
            matching_ids = set()
            
            csv_rows = list(self.combined_csv.iterrows())
            total_csv = len(csv_rows)
            
            for i, (_, row) in enumerate(csv_rows):
                if i % 100 == 0:
                    pct = int((i / total_csv) * 30) if total_csv > 0 else 0
                    self.progress.emit(pct, f"Indexing CSV rows: {i}/{total_csv}")
                
                if self.filter_col in row and str(row[self.filter_col]) == self.filter_val:
                    image_id = str(row[self.id_col]).strip()
                    matching_ids.add(image_id)
                    matching_ids.add(image_id.lower())
                    # Also add without extension
                    matching_ids.add(Path(image_id).stem)
                    matching_ids.add(Path(image_id).stem.lower())
            
            self.progress.emit(30, f"Found {len(matching_ids)} matching IDs in CSV")
            
            # Mark matching DICOM files
            matched_count = 0
            total_dicom = len(self.dicom_files)
            
            for i, dicom_file in enumerate(self.dicom_files):
                if i % 50 == 0:
                    pct = 30 + int((i / total_dicom) * 70) if total_dicom > 0 else 30
                    self.progress.emit(pct, f"Matching images: {i}/{total_dicom}")
                
                dicom_file.matched = False
                dicom_file.annotations = []
                
                # Check if this file matches any ID
                if (dicom_file.filename in matching_ids or 
                    dicom_file.filename.lower() in matching_ids or
                    any(mid in dicom_file.filename or dicom_file.filename in mid 
                        for mid in matching_ids)):
                    dicom_file.matched = True
                    matched_count += 1
                    # Add annotation info
                    for _, row in self.combined_csv.iterrows():
                        csv_id = str(row[self.id_col]).strip()
                        if (csv_id in dicom_file.filename or 
                            dicom_file.filename in csv_id or
                            Path(csv_id).stem == dicom_file.filename):
                            dicom_file.annotations.append(row.to_dict())
            
            self.progress.emit(100, f"Filter complete - {matched_count} matches found")
            self.finished.emit(self.dicom_files, matched_count)
            
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")


class DicomExplorerGUI(QMainWindow):
    """Main GUI window for DICOM Explorer."""
    
    def __init__(self):
        super().__init__()
        self.dicom_files: List[DicomFileInfo] = []
        self.csv_data: Dict[str, pd.DataFrame] = {}  # filename -> dataframe
        self.combined_csv: Optional[pd.DataFrame] = None
        self.scan_thread = None
        self.filter_thread = None
        self.init_ui()
        self.setup_responsive_size()
    
    def setup_responsive_size(self):
        """Setup responsive window size based on screen."""
        screen = QApplication.primaryScreen()
        if screen:
            screen_geometry = screen.availableGeometry()
            width = int(screen_geometry.width() * 0.8)
            height = int(screen_geometry.height() * 0.85)
            width = max(1000, min(width, 1800))
            height = max(700, min(height, 1200))
            self.resize(width, height)
            x = (screen_geometry.width() - width) // 2
            y = (screen_geometry.height() - height) // 2
            self.move(x, y)
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("DICOM Explorer")
        self.setMinimumSize(1000, 700)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Create menu bar
        self.create_menu_bar()
        
        # ===== STEP 1: Select DICOM Folder =====
        step1_group = QGroupBox("Step 1: Select DICOM Folder")
        step1_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        step1_layout = QHBoxLayout(step1_group)
        
        self.dicom_path = QLineEdit()
        self.dicom_path.setPlaceholderText("Click 'Browse' to select a folder containing DICOM images...")
        self.dicom_path.setReadOnly(True)
        self.dicom_path.setMinimumHeight(35)
        step1_layout.addWidget(self.dicom_path, 1)
        
        browse_btn = QPushButton("📁 Browse...")
        browse_btn.setMinimumHeight(35)
        browse_btn.setMinimumWidth(120)
        browse_btn.clicked.connect(self.browse_dicom_folder)
        browse_btn.setToolTip("Select a folder containing DICOM images (.dcm, .dicom files)")
        step1_layout.addWidget(browse_btn)
        
        self.scan_btn = QPushButton("🔍 Scan")
        self.scan_btn.setMinimumHeight(35)
        self.scan_btn.setMinimumWidth(100)
        self.scan_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        self.scan_btn.clicked.connect(self.scan_and_load)
        self.scan_btn.setToolTip("Scan the folder for DICOM images")
        step1_layout.addWidget(self.scan_btn)
        
        main_layout.addWidget(step1_group)
        
        # ===== STEP 2 (Optional): Load CSV Annotations =====
        step2_group = QGroupBox("Step 2 (Optional): Filter by CSV Data")
        step2_group.setStyleSheet("QGroupBox { font-size: 14px; font-weight: bold; }")
        step2_layout = QVBoxLayout(step2_group)
        
        csv_row = QHBoxLayout()
        self.csv_list = QLineEdit()
        self.csv_list.setPlaceholderText("No CSV file loaded (optional - skip if you just want to browse images)")
        self.csv_list.setReadOnly(True)
        self.csv_list.setMinimumHeight(35)
        csv_row.addWidget(self.csv_list, 1)
        
        add_csv_btn = QPushButton("📄 Add CSV...")
        add_csv_btn.setMinimumHeight(35)
        add_csv_btn.clicked.connect(self.add_csv_file)
        add_csv_btn.setToolTip("Load a CSV file containing image annotations/labels")
        csv_row.addWidget(add_csv_btn)
        
        clear_csv_btn = QPushButton("Clear")
        clear_csv_btn.setMinimumHeight(35)
        clear_csv_btn.clicked.connect(self.clear_csv_files)
        csv_row.addWidget(clear_csv_btn)
        
        step2_layout.addLayout(csv_row)
        
        # CSV Mapping (only shown when CSV is loaded)
        self.csv_mapping_frame = QFrame()
        mapping_layout = QVBoxLayout(self.csv_mapping_frame)
        mapping_layout.setContentsMargins(0, 10, 0, 0)
        
        # Row 1: Link CSV to images
        link_row = QHBoxLayout()
        link_row.addWidget(QLabel("1️⃣ Link CSV to images using column:"))
        self.id_column = NoScrollComboBox()
        self.id_column.addItem("-- Select column with image filename --")
        self.id_column.setMinimumWidth(200)
        self.id_column.setToolTip("Which CSV column contains the DICOM filename?")
        link_row.addWidget(self.id_column)
        link_row.addStretch()
        mapping_layout.addLayout(link_row)
        
        # Row 2: Filter by column and value
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("2️⃣ Find images where"))
        
        self.filter_column = NoScrollComboBox()
        self.filter_column.addItem("-- Select column --")
        self.filter_column.setMinimumWidth(150)
        self.filter_column.currentTextChanged.connect(self.on_filter_column_changed)
        self.filter_column.setToolTip("Which column do you want to filter by? (e.g., 'diagnosis', 'label')")
        filter_row.addWidget(self.filter_column)
        
        filter_row.addWidget(QLabel("equals"))
        
        self.filter_value = NoScrollComboBox()
        self.filter_value.addItem("-- Select value --")
        self.filter_value.setMinimumWidth(150)
        self.filter_value.setToolTip("What value are you looking for? (e.g., 'TB', 'Pneumonia')")
        filter_row.addWidget(self.filter_value)
        
        filter_row.addStretch()
        
        self.filter_btn = QPushButton("🔍 Filter Images")
        self.filter_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        self.filter_btn.clicked.connect(self.do_filter_by_csv)
        self.filter_btn.setToolTip("Show only images matching this filter")
        filter_row.addWidget(self.filter_btn)
        
        self.clear_filter_btn = QPushButton("Clear Filter")
        self.clear_filter_btn.clicked.connect(self.clear_csv_filter)
        self.clear_filter_btn.setToolTip("Show all images again")
        filter_row.addWidget(self.clear_filter_btn)
        
        mapping_layout.addLayout(filter_row)
        
        # Row 3: Bounding Box columns (optional)
        bbox_row = QHBoxLayout()
        self.show_bbox = QCheckBox("3️⃣ Show bounding boxes from columns:")
        self.show_bbox.setChecked(False)
        self.show_bbox.setToolTip("Enable to draw bounding boxes on the image preview")
        self.show_bbox.toggled.connect(self.on_bbox_toggled)
        bbox_row.addWidget(self.show_bbox)
        
        bbox_row.addWidget(QLabel("X1:"))
        self.bbox_x1_col = NoScrollComboBox()
        self.bbox_x1_col.addItem("-- Select --")
        self.bbox_x1_col.setMinimumWidth(100)
        self.bbox_x1_col.setToolTip("Column for X1 (left) coordinate")
        self.bbox_x1_col.currentTextChanged.connect(self.refresh_preview)
        bbox_row.addWidget(self.bbox_x1_col)
        
        bbox_row.addWidget(QLabel("Y1:"))
        self.bbox_y1_col = NoScrollComboBox()
        self.bbox_y1_col.addItem("-- Select --")
        self.bbox_y1_col.setMinimumWidth(100)
        self.bbox_y1_col.setToolTip("Column for Y1 (top) coordinate")
        self.bbox_y1_col.currentTextChanged.connect(self.refresh_preview)
        bbox_row.addWidget(self.bbox_y1_col)
        
        bbox_row.addWidget(QLabel("X2:"))
        self.bbox_x2_col = NoScrollComboBox()
        self.bbox_x2_col.addItem("-- Select --")
        self.bbox_x2_col.setMinimumWidth(100)
        self.bbox_x2_col.setToolTip("Column for X2 (right) or Width")
        self.bbox_x2_col.currentTextChanged.connect(self.refresh_preview)
        bbox_row.addWidget(self.bbox_x2_col)
        
        bbox_row.addWidget(QLabel("Y2:"))
        self.bbox_y2_col = NoScrollComboBox()
        self.bbox_y2_col.addItem("-- Select --")
        self.bbox_y2_col.setMinimumWidth(100)
        self.bbox_y2_col.setToolTip("Column for Y2 (bottom) or Height")
        self.bbox_y2_col.currentTextChanged.connect(self.refresh_preview)
        bbox_row.addWidget(self.bbox_y2_col)
        
        bbox_row.addStretch()
        mapping_layout.addLayout(bbox_row)
        
        # Row 4: Bbox format and label column
        bbox_options_row = QHBoxLayout()
        bbox_options_row.addSpacing(30)  # Indent to align with checkbox above
        
        bbox_options_row.addWidget(QLabel("Format:"))
        self.annotation_type = NoScrollComboBox()
        self.annotation_type.addItems(["x1,y1,x2,y2 (corners)", "x,y,w,h (position + size)"])
        self.annotation_type.setToolTip("How are the coordinates formatted in your CSV?")
        self.annotation_type.currentTextChanged.connect(self.refresh_preview)
        bbox_options_row.addWidget(self.annotation_type)
        
        bbox_options_row.addWidget(QLabel("Label column:"))
        self.label_column = NoScrollComboBox()
        self.label_column.addItem("-- None --")
        self.label_column.setMinimumWidth(120)
        self.label_column.setToolTip("Column containing label text to show on boxes (optional)")
        self.label_column.currentTextChanged.connect(self.refresh_preview)
        bbox_options_row.addWidget(self.label_column)
        
        self.show_labels = QCheckBox("Show labels")
        self.show_labels.setChecked(True)
        self.show_labels.toggled.connect(self.refresh_preview)
        bbox_options_row.addWidget(self.show_labels)
        
        bbox_options_row.addStretch()
        mapping_layout.addLayout(bbox_options_row)
        
        # Hidden match method - use partial match by default
        self.match_method = NoScrollComboBox()
        self.match_method.addItems(["Partial match (contains)"])
        
        self.csv_mapping_frame.setVisible(False)
        step2_layout.addWidget(self.csv_mapping_frame)
        
        main_layout.addWidget(step2_group)
        
        # ===== Main Content: Image List + Preview =====
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left side: Image list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Search/filter bar
        filter_row = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("🔍 Search images...")
        self.search_input.setMinimumHeight(30)
        self.search_input.textChanged.connect(self.apply_filters)
        filter_row.addWidget(self.search_input, 1)
        
        self.match_filter = NoScrollComboBox()
        self.match_filter.addItems(["All Images", "Matched Only", "Unmatched Only"])
        self.match_filter.currentTextChanged.connect(self.apply_filters)
        self.match_filter.setToolTip("Filter images by match status")
        filter_row.addWidget(self.match_filter)
        
        left_layout.addLayout(filter_row)
        
        # Image count
        self.count_label = QLabel("No images loaded")
        self.count_label.setStyleSheet("color: #666; font-size: 12px;")
        left_layout.addWidget(self.count_label)
        
        # Image list
        self.image_list = QListWidget()
        self.image_list.setAlternatingRowColors(True)
        self.image_list.itemClicked.connect(self.on_image_selected)
        self.image_list.itemDoubleClicked.connect(self.on_image_double_clicked)
        self.image_list.setStyleSheet("""
            QListWidget::item { padding: 8px; font-size: 13px; }
            QListWidget::item:selected { background-color: #2196F3; color: white; }
        """)
        left_layout.addWidget(self.image_list, 1)
        
        content_splitter.addWidget(left_widget)
        
        # Right side: Preview + Details
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Preview
        preview_frame = QFrame()
        preview_frame.setStyleSheet("QFrame { background-color: #2a2a2a; border-radius: 5px; }")
        preview_layout = QVBoxLayout(preview_frame)
        
        self.preview_label = ClickableLabel("Select an image to preview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("color: #888; font-size: 14px;")
        self.preview_label.setMinimumSize(400, 300)
        self.preview_label.setToolTip("Double-click to open in image viewer")
        self.preview_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.preview_label.doubleClicked.connect(self.open_current_in_viewer)
        preview_layout.addWidget(self.preview_label)
        
        right_layout.addWidget(preview_frame, 2)
        
        # Details
        details_group = QGroupBox("Image Details")
        details_layout = QVBoxLayout(details_group)
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(150)
        self.details_text.setPlaceholderText("Image metadata and annotations will appear here")
        details_layout.addWidget(self.details_text)
        
        right_layout.addWidget(details_group, 1)
        
        content_splitter.addWidget(right_widget)
        content_splitter.setSizes([350, 550])
        
        main_layout.addWidget(content_splitter, 1)
        
        # ===== Bottom: Status & Export =====
        bottom_layout = QHBoxLayout()
        
        self.stats_label = QLabel("")
        self.stats_label.setStyleSheet("color: #666;")
        bottom_layout.addWidget(self.stats_label)
        
        bottom_layout.addStretch()
        
        self.export_btn = QPushButton("📤 Export List to CSV")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_filtered)
        self.export_btn.setToolTip("Export the current filtered image list to a CSV file")
        bottom_layout.addWidget(self.export_btn)
        
        main_layout.addLayout(bottom_layout)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Status bar with helpful message
        self.statusBar().showMessage("👆 Start by selecting a folder with DICOM images, then click 'Scan'")
        
        # Hidden widgets for compatibility
        self.label_filter = NoScrollComboBox()
        self.label_filter.addItem("All Labels")
        self.bbox_frame = QFrame()
        self.image_info_label = QLabel()
        self.details_table = None  # Not used in simplified UI
        
        # Initialize filter widgets if not already created
        if not hasattr(self, 'filter_column'):
            self.filter_column = NoScrollComboBox()
            self.filter_column.addItem("-- Select column --")
        if not hasattr(self, 'filter_value'):
            self.filter_value = NoScrollComboBox()
            self.filter_value.addItem("-- Select value --")
        
        # Apply clean stylesheet
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; }
            QGroupBox { 
                font-weight: bold; 
                border: 2px solid #ccc; 
                border-radius: 8px; 
                margin-top: 12px; 
                padding-top: 12px;
                background-color: white;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 15px; 
                padding: 0 8px;
                color: #333;
            }
            QLineEdit { 
                padding: 8px; 
                border: 1px solid #ccc; 
                border-radius: 4px; 
                background-color: white; 
            }
            QComboBox { 
                padding: 6px; 
                border: 1px solid #ccc; 
                border-radius: 4px; 
                background-color: white; 
            }
            QPushButton { 
                padding: 8px 16px; 
                border: 1px solid #bbb; 
                border-radius: 4px; 
                background-color: #f5f5f5; 
            }
            QPushButton:hover { background-color: #e8e8e8; }
        """)
    
    def create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_folder_action = QAction("Open DICOM Folder...", self)
        open_folder_action.setShortcut("Ctrl+O")
        open_folder_action.triggered.connect(self.browse_dicom_folder)
        file_menu.addAction(open_folder_action)
        
        add_csv_action = QAction("Add CSV File...", self)
        add_csv_action.setShortcut("Ctrl+Shift+O")
        add_csv_action.triggered.connect(self.add_csv_file)
        file_menu.addAction(add_csv_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("Export Filtered Images...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_filtered)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        refresh_action = QAction("Refresh", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self.apply_filters)
        view_menu.addAction(refresh_action)
        
        clear_action = QAction("Clear All", self)
        clear_action.triggered.connect(self.clear_all)
        view_menu.addAction(clear_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_input_section(self) -> QGroupBox:
        """Create input paths section."""
        group = QGroupBox("Data Sources")
        layout = QVBoxLayout(group)
        
        # DICOM folder row
        dicom_layout = QHBoxLayout()
        dicom_layout.addWidget(QLabel("📁 DICOM Folder:"))
        self.dicom_path = QLineEdit()
        self.dicom_path.setPlaceholderText("Select folder containing DICOM images...")
        self.dicom_path.setReadOnly(True)
        dicom_layout.addWidget(self.dicom_path, 1)
        
        browse_dicom_btn = QPushButton("Browse")
        browse_dicom_btn.clicked.connect(self.browse_dicom_folder)
        dicom_layout.addWidget(browse_dicom_btn)
        
        layout.addLayout(dicom_layout)
        
        # CSV files row
        csv_layout = QHBoxLayout()
        csv_layout.addWidget(QLabel("📄 CSV Files:"))
        self.csv_list = QLineEdit()
        self.csv_list.setPlaceholderText("No CSV files added...")
        self.csv_list.setReadOnly(True)
        csv_layout.addWidget(self.csv_list, 1)
        
        add_csv_btn = QPushButton("Add CSV")
        add_csv_btn.clicked.connect(self.add_csv_file)
        csv_layout.addWidget(add_csv_btn)
        
        clear_csv_btn = QPushButton("Clear")
        clear_csv_btn.clicked.connect(self.clear_csv_files)
        csv_layout.addWidget(clear_csv_btn)
        
        layout.addLayout(csv_layout)
        
        # Scan button row
        scan_layout = QHBoxLayout()
        scan_layout.addStretch()
        
        self.scan_btn = QPushButton("🔍 Scan && Load")
        self.scan_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 10px 25px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.scan_btn.clicked.connect(self.scan_and_load)
        scan_layout.addWidget(self.scan_btn)
        
        scan_layout.addStretch()
        layout.addLayout(scan_layout)
        
        return group
    
    def create_mapping_section(self) -> QGroupBox:
        """Create column mapping section."""
        group = QGroupBox("Column Mapping")
        layout = QFormLayout(group)
        
        # ID Column selection
        self.id_column = NoScrollComboBox()
        self.id_column.addItem("-- Select ID Column --")
        self.id_column.currentTextChanged.connect(self.on_mapping_changed)
        layout.addRow("ID Column:", self.id_column)
        
        # ID Match Method
        self.match_method = NoScrollComboBox()
        self.match_method.addItems([
            "Exact Match (filename)",
            "Exact Match (SOPInstanceUID)",
            "Contains (partial match)",
            "Starts With",
            "Ends With"
        ])
        self.match_method.currentTextChanged.connect(self.on_mapping_changed)
        layout.addRow("Match Method:", self.match_method)
        
        # Label Column selection
        self.label_column = NoScrollComboBox()
        self.label_column.addItem("-- No Label Column --")
        self.label_column.currentTextChanged.connect(self.on_mapping_changed)
        layout.addRow("Label Column:", self.label_column)
        
        # Annotation Type
        self.annotation_type = NoScrollComboBox()
        self.annotation_type.addItems([
            "Classification (labels only)",
            "Bounding Box (x,y,w,h)",
            "Bounding Box (x1,y1,x2,y2)",
            "Segmentation Mask",
            "Mixed / Other"
        ])
        layout.addRow("Annotation Type:", self.annotation_type)
        
        # Bounding box columns (shown conditionally)
        self.bbox_frame = QFrame()
        bbox_layout = QFormLayout(self.bbox_frame)
        bbox_layout.setContentsMargins(0, 5, 0, 0)
        
        self.bbox_x1_col = NoScrollComboBox()
        self.bbox_x1_col.addItem("-- Select --")
        bbox_layout.addRow("X1/X Column:", self.bbox_x1_col)
        
        self.bbox_y1_col = NoScrollComboBox()
        self.bbox_y1_col.addItem("-- Select --")
        bbox_layout.addRow("Y1/Y Column:", self.bbox_y1_col)
        
        self.bbox_x2_col = NoScrollComboBox()
        self.bbox_x2_col.addItem("-- Select --")
        bbox_layout.addRow("X2/W Column:", self.bbox_x2_col)
        
        self.bbox_y2_col = NoScrollComboBox()
        self.bbox_y2_col.addItem("-- Select --")
        bbox_layout.addRow("Y2/H Column:", self.bbox_y2_col)
        
        self.bbox_frame.setVisible(False)
        layout.addRow(self.bbox_frame)
        
        # Connect annotation type change
        self.annotation_type.currentTextChanged.connect(self.on_annotation_type_changed)
        
        return group
    
    def create_filter_section(self) -> QGroupBox:
        """Create filter section."""
        group = QGroupBox("Filters")
        layout = QVBoxLayout(group)
        
        # Search bar
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("🔍"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search by image ID...")
        self.search_input.textChanged.connect(self.apply_filters)
        search_layout.addWidget(self.search_input)
        layout.addLayout(search_layout)
        
        # Filter dropdowns
        filter_layout = QHBoxLayout()
        
        # Label filter
        filter_layout.addWidget(QLabel("Label:"))
        self.label_filter = NoScrollComboBox()
        self.label_filter.addItem("All Labels")
        self.label_filter.currentTextChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.label_filter, 1)
        
        # Match status filter
        filter_layout.addWidget(QLabel("Show:"))
        self.match_filter = NoScrollComboBox()
        self.match_filter.addItems(["All Images", "Matched Only", "Unmatched Only", "With Annotations"])
        self.match_filter.currentTextChanged.connect(self.apply_filters)
        filter_layout.addWidget(self.match_filter, 1)
        
        layout.addLayout(filter_layout)
        
        return group
    
    def create_image_list_section(self) -> QGroupBox:
        """Create image list section."""
        group = QGroupBox("Images")
        layout = QVBoxLayout(group)
        
        self.image_list = QListWidget()
        self.image_list.setAlternatingRowColors(True)
        self.image_list.itemClicked.connect(self.on_image_selected)
        self.image_list.itemDoubleClicked.connect(self.on_image_double_clicked)
        layout.addWidget(self.image_list)
        
        # Count label
        self.count_label = QLabel("0 images")
        self.count_label.setStyleSheet("color: #666;")
        layout.addWidget(self.count_label)
        
        return group
    
    def create_preview_section(self) -> QGroupBox:
        """Create image preview section."""
        group = QGroupBox("Image Preview")
        layout = QVBoxLayout(group)
        
        # Image display
        self.preview_label = QLabel("Select an image to preview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(400, 300)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #1e1e1e;
                color: #666;
                border: 1px solid #333;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.preview_label, 1)
        
        # Image info
        self.image_info_label = QLabel("")
        self.image_info_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.image_info_label)
        
        # Overlay options
        overlay_layout = QHBoxLayout()
        self.show_bbox = QCheckBox("Show Bounding Boxes")
        self.show_bbox.setChecked(True)
        self.show_bbox.toggled.connect(self.refresh_preview)
        overlay_layout.addWidget(self.show_bbox)
        
        self.show_labels = QCheckBox("Show Labels")
        self.show_labels.setChecked(True)
        self.show_labels.toggled.connect(self.refresh_preview)
        overlay_layout.addWidget(self.show_labels)
        
        overlay_layout.addStretch()
        layout.addLayout(overlay_layout)
        
        return group
    
    def create_details_section(self) -> QGroupBox:
        """Create annotation details section."""
        group = QGroupBox("Annotation Details")
        layout = QVBoxLayout(group)
        
        # Details table
        self.details_table = QTableWidget()
        self.details_table.setColumnCount(2)
        self.details_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.details_table.horizontalHeader().setStretchLastSection(True)
        self.details_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.details_table.setAlternatingRowColors(True)
        self.details_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        layout.addWidget(self.details_table)
        
        return group
    
    def browse_dicom_folder(self):
        """Browse for DICOM folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select DICOM Folder", "",
            QFileDialog.Option.ShowDirsOnly
        )
        if folder:
            self.dicom_path.setText(folder)
    
    def add_csv_file(self):
        """Add a CSV file."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select CSV File(s)", "",
            "CSV Files (*.csv);;All Files (*)"
        )
        for file in files:
            if file and file not in self.csv_data:
                try:
                    df = pd.read_csv(file)
                    self.csv_data[file] = df
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to load {file}:\n{str(e)}")
        
        self.update_csv_display()
        self.update_column_dropdowns()
    
    def clear_csv_files(self):
        """Clear all CSV files."""
        self.csv_data.clear()
        self.combined_csv = None
        self.update_csv_display()
        self.update_column_dropdowns()
    
    def update_csv_display(self):
        """Update CSV files display."""
        if self.csv_data:
            names = [Path(f).name for f in self.csv_data.keys()]
            self.csv_list.setText(", ".join(names))
            self.csv_mapping_frame.setVisible(True)  # Show mapping options
        else:
            self.csv_list.setText("")
            self.csv_list.setPlaceholderText("No CSV file loaded (optional - skip if you just want to browse images)")
            self.csv_mapping_frame.setVisible(False)  # Hide mapping options
    
    def update_column_dropdowns(self):
        """Update column selection dropdowns based on loaded CSVs."""
        # Get all unique columns from all CSVs
        all_columns = set()
        for df in self.csv_data.values():
            all_columns.update(df.columns.tolist())
        
        columns = sorted(list(all_columns))
        
        # Update ID column dropdown
        current_id = self.id_column.currentText()
        self.id_column.clear()
        self.id_column.addItem("-- Select column with image filename --")
        self.id_column.addItems(columns)
        idx = self.id_column.findText(current_id)
        if idx >= 0:
            self.id_column.setCurrentIndex(idx)
        
        # Update filter column dropdown
        current_filter = self.filter_column.currentText()
        self.filter_column.clear()
        self.filter_column.addItem("-- Select column --")
        self.filter_column.addItems(columns)
        idx = self.filter_column.findText(current_filter)
        if idx >= 0:
            self.filter_column.setCurrentIndex(idx)
        
        # Update other dropdowns for compatibility
        for combo in [self.label_column, 
                      self.bbox_x1_col, self.bbox_y1_col, 
                      self.bbox_x2_col, self.bbox_y2_col]:
            current = combo.currentText()
            combo.clear()
            if combo == self.label_column:
                combo.addItem("-- None --")
            else:
                combo.addItem("-- Select --")
            combo.addItems(columns)
            idx = combo.findText(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
    
    def on_filter_column_changed(self, column_name: str):
        """Update filter value dropdown when filter column changes."""
        self.filter_value.clear()
        self.filter_value.addItem("-- Select value --")
        
        if column_name.startswith("--") or not self.csv_data:
            return
        
        # Get unique values from selected column across all CSVs
        all_values = set()
        for df in self.csv_data.values():
            if column_name in df.columns:
                values = df[column_name].dropna().unique()
                all_values.update(str(v) for v in values)
        
        # Add sorted values to dropdown
        for value in sorted(all_values):
            self.filter_value.addItem(str(value))
    
    def on_annotation_type_changed(self, text: str):
        """Handle annotation type change."""
        show_bbox = "Bounding Box" in text
        self.bbox_frame.setVisible(show_bbox)
    
    def on_bbox_toggled(self, checked: bool):
        """Handle bounding box checkbox toggle."""
        # Refresh the preview to show/hide bounding boxes
        self.refresh_preview()
    
    def on_mapping_changed(self):
        """Handle mapping change - re-match annotations."""
        # Don't auto-match, wait for user to click Match button
        pass
    
    def do_filter_by_csv(self):
        """Filter images by CSV column value."""
        if not self.dicom_files:
            QMessageBox.warning(self, "No Images", "Please scan a DICOM folder first (Step 1).")
            return
        
        if not self.csv_data:
            QMessageBox.warning(self, "No CSV", "Please add a CSV file first.")
            return
        
        id_col = self.id_column.currentText()
        if id_col.startswith("--"):
            QMessageBox.warning(self, "Select Column", 
                "Please select which column contains the image filename\n"
                "(This links CSV rows to your DICOM images)")
            return
        
        filter_col = self.filter_column.currentText()
        if filter_col.startswith("--"):
            QMessageBox.warning(self, "Select Filter Column", 
                "Please select which column to filter by\n"
                "(e.g., 'diagnosis', 'label', 'finding')")
            return
        
        filter_val = self.filter_value.currentText()
        if filter_val.startswith("--"):
            QMessageBox.warning(self, "Select Value", 
                "Please select what value to look for\n"
                "(e.g., 'TB', 'Pneumonia', 'Normal')")
            return
        
        # Combine CSV data
        self.combined_csv = pd.concat(self.csv_data.values(), ignore_index=True)
        
        # Store filter values for result message
        self._filter_col = filter_col
        self._filter_val = filter_val
        
        # Show progress bar and disable filter button
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.filter_btn.setEnabled(False)
        self.clear_filter_btn.setEnabled(False)
        self.statusBar().showMessage("Filtering images...")
        
        # Start filter thread
        self.filter_thread = FilterThread(
            self.dicom_files, 
            self.combined_csv, 
            id_col, 
            filter_col, 
            filter_val
        )
        self.filter_thread.progress.connect(self.on_filter_progress)
        self.filter_thread.finished.connect(self.on_filter_finished)
        self.filter_thread.error.connect(self.on_filter_error)
        self.filter_thread.start()
    
    def on_filter_progress(self, value: int, message: str):
        """Handle filter progress."""
        self.progress_bar.setValue(value)
        self.statusBar().showMessage(message)
    
    def on_filter_finished(self, dicom_files: list, matched_count: int):
        """Handle filter completion."""
        self.dicom_files = dicom_files
        self.progress_bar.setVisible(False)
        self.filter_btn.setEnabled(True)
        self.clear_filter_btn.setEnabled(True)
        
        # Update display
        self.apply_filters()
        self.update_statistics()
        
        # Show matched only by default
        self.match_filter.setCurrentText("Matched Only")
        
        filter_col = getattr(self, '_filter_col', 'column')
        filter_val = getattr(self, '_filter_val', 'value')
        
        self.statusBar().showMessage(f"Filter complete - {matched_count} images matched")
        QMessageBox.information(self, "Filter Applied", 
            f"Found {matched_count} images where '{filter_col}' = '{filter_val}'")
    
    def on_filter_error(self, error: str):
        """Handle filter error."""
        self.progress_bar.setVisible(False)
        self.filter_btn.setEnabled(True)
        self.clear_filter_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Filter failed:\n{error}")
    
    def clear_csv_filter(self):
        """Clear the CSV filter and show all images."""
        for dicom_file in self.dicom_files:
            dicom_file.matched = False
            dicom_file.annotations = []
        
        self.match_filter.setCurrentText("All Images")
        self.apply_filters()
        self.update_statistics()
        self.statusBar().showMessage("Filter cleared - showing all images")
    
    def do_match_annotations(self):
        """Match annotations when user clicks Match button."""
        if not self.dicom_files:
            QMessageBox.warning(self, "No Images", "Please scan a DICOM folder first.")
            return
        
        if not self.csv_data:
            QMessageBox.warning(self, "No CSV", "Please add a CSV file first.")
            return
        
        id_col = self.id_column.currentText()
        if id_col.startswith("--"):
            QMessageBox.warning(self, "Select Column", "Please select which column contains the image ID.")
            return
        
        # Combine CSV data and match
        self.combined_csv = pd.concat(self.csv_data.values(), ignore_index=True)
        self.match_annotations()
        self.apply_filters()
        self.update_statistics()
        
        matched = sum(1 for f in self.dicom_files if f.matched)
        QMessageBox.information(self, "Matching Complete", 
            f"Matched {matched} of {len(self.dicom_files)} images with annotations.")
    
    def scan_and_load(self):
        """Scan DICOM folder and load data."""
        dicom_folder = self.dicom_path.text()
        
        if not dicom_folder:
            QMessageBox.warning(self, "Error", "Please select a DICOM folder.")
            return
        
        if not Path(dicom_folder).exists():
            QMessageBox.warning(self, "Error", "DICOM folder does not exist.")
            return
        
        # Clear previous results
        self.image_list.clear()
        self.dicom_files = []
        
        # Start scanning
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.scan_btn.setEnabled(False)
        self.statusBar().showMessage("Scanning DICOM files...")
        
        self.scan_thread = ScanThread(Path(dicom_folder))
        self.scan_thread.progress.connect(self.on_scan_progress)
        self.scan_thread.finished.connect(self.on_scan_finished)
        self.scan_thread.error.connect(self.on_scan_error)
        self.scan_thread.log.connect(self.on_scan_log)
        self.scan_thread.start()
    
    def on_scan_log(self, message: str):
        """Handle scan log messages."""
        self.statusBar().showMessage(message)
    
    def on_scan_progress(self, value: int, message: str):
        """Handle scan progress."""
        self.progress_bar.setValue(value)
        self.statusBar().showMessage(message)
    
    def on_scan_finished(self, files: List[DicomFileInfo]):
        """Handle scan completion."""
        self.dicom_files = files
        self.progress_bar.setVisible(False)
        self.scan_btn.setEnabled(True)
        
        # Combine CSV data
        if self.csv_data:
            self.combined_csv = pd.concat(self.csv_data.values(), ignore_index=True)
            self.update_label_filter()
        
        # Match annotations
        self.match_annotations()
        
        # Apply filters and display
        self.apply_filters()
        
        # Update statistics
        self.update_statistics()
        
        self.statusBar().showMessage(f"Loaded {len(files)} DICOM files")
    
    def on_scan_error(self, error: str):
        """Handle scan error."""
        self.progress_bar.setVisible(False)
        self.scan_btn.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Scan failed:\n{error}")
    
    def match_annotations(self):
        """Match DICOM files with annotations."""
        if not self.dicom_files or self.combined_csv is None:
            return
        
        id_col = self.id_column.currentText()
        if id_col.startswith("--"):
            return
        
        if id_col not in self.combined_csv.columns:
            return
        
        match_method = self.match_method.currentText().lower()
        label_col = self.label_column.currentText()
        if label_col.startswith("--"):
            label_col = None
        
        # Reset all matches
        for dicom_file in self.dicom_files:
            dicom_file.matched = False
            dicom_file.annotations = []
        
        # Create lookup for faster matching
        dicom_lookup = {}
        for dicom_file in self.dicom_files:
            # Add multiple keys for different match methods
            dicom_lookup[dicom_file.filename] = dicom_file
            dicom_lookup[dicom_file.filename.lower()] = dicom_file
            if dicom_file.sop_instance_uid:
                dicom_lookup[dicom_file.sop_instance_uid] = dicom_file
        
        # Match annotations
        for _, row in self.combined_csv.iterrows():
            csv_id = str(row[id_col]).strip()
            
            matched_file = None
            
            if "exact" in match_method:
                # Try exact filename match
                matched_file = dicom_lookup.get(csv_id) or dicom_lookup.get(csv_id.lower())
                # Try with/without extension
                if not matched_file:
                    csv_id_no_ext = Path(csv_id).stem
                    matched_file = dicom_lookup.get(csv_id_no_ext) or dicom_lookup.get(csv_id_no_ext.lower())
            
            elif "partial" in match_method or "contains" in match_method:
                for key, dicom_file in dicom_lookup.items():
                    if csv_id.lower() in key.lower() or key.lower() in csv_id.lower():
                        matched_file = dicom_file
                        break
            
            elif "starts" in match_method:
                for key, dicom_file in dicom_lookup.items():
                    if key.lower().startswith(csv_id.lower()) or csv_id.lower().startswith(key.lower()):
                        matched_file = dicom_file
                        break
            
            elif "ends" in match_method:
                for key, dicom_file in dicom_lookup.items():
                    if key.lower().endswith(csv_id.lower()) or csv_id.lower().endswith(key.lower()):
                        matched_file = dicom_file
                        break
            
            if matched_file:
                matched_file.matched = True
                annotation = row.to_dict()
                matched_file.annotations.append(annotation)
        
        self.update_statistics()
    
    def update_label_filter(self):
        """Update label filter dropdown with unique labels."""
        self.label_filter.clear()
        self.label_filter.addItem("All Labels")
        
        label_col = self.label_column.currentText()
        if label_col.startswith("--") or self.combined_csv is None:
            return
        
        if label_col in self.combined_csv.columns:
            unique_labels = self.combined_csv[label_col].dropna().unique()
            for label in sorted(unique_labels, key=str):
                self.label_filter.addItem(str(label))
    
    def apply_filters(self):
        """Apply filters and update image list."""
        search_text = self.search_input.text().lower()
        label_filter = self.label_filter.currentText()
        match_filter = self.match_filter.currentText()
        label_col = self.label_column.currentText()
        
        self.image_list.clear()
        filtered_count = 0
        
        for dicom_file in self.dicom_files:
            # Search filter
            if search_text:
                search_match = (
                    search_text in dicom_file.filename.lower() or
                    (dicom_file.sop_instance_uid and search_text in dicom_file.sop_instance_uid.lower()) or
                    (dicom_file.patient_id and search_text in dicom_file.patient_id.lower())
                )
                if not search_match:
                    continue
            
            # Match status filter
            if match_filter == "Matched Only" and not dicom_file.matched:
                continue
            elif match_filter == "Unmatched Only" and dicom_file.matched:
                continue
            elif match_filter == "With Annotations" and not dicom_file.annotations:
                continue
            
            # Label filter
            if label_filter != "All Labels" and not label_col.startswith("--"):
                has_label = False
                for ann in dicom_file.annotations:
                    if label_col in ann and str(ann[label_col]) == label_filter:
                        has_label = True
                        break
                if not has_label:
                    continue
            
            # Add to list
            item = QListWidgetItem()
            
            # Set icon based on status
            if dicom_file.annotations:
                icon = "✓"
                color = "#4CAF50"
            elif dicom_file.matched:
                icon = "○"
                color = "#9E9E9E"
            else:
                icon = "?"
                color = "#FF9800"
            
            display_text = f"{icon} {dicom_file.filename}"
            if dicom_file.annotations:
                display_text += f" ({len(dicom_file.annotations)} ann.)"
            
            item.setText(display_text)
            item.setData(Qt.ItemDataRole.UserRole, dicom_file)
            item.setForeground(QColor(color))
            
            self.image_list.addItem(item)
            filtered_count += 1
        
        self.count_label.setText(f"{filtered_count} of {len(self.dicom_files)} images")
        self.export_btn.setEnabled(filtered_count > 0)
    
    def update_statistics(self):
        """Update statistics display."""
        total = len(self.dicom_files)
        matched = sum(1 for f in self.dicom_files if f.matched)
        with_annotations = sum(1 for f in self.dicom_files if f.annotations)
        total_annotations = sum(len(f.annotations) for f in self.dicom_files)
        
        self.stats_label.setText(
            f"Total: {total} | Matched: {matched} | "
            f"With Annotations: {with_annotations} | Total Annotations: {total_annotations}"
        )
    
    def on_image_selected(self, item: QListWidgetItem):
        """Handle image selection."""
        dicom_file: DicomFileInfo = item.data(Qt.ItemDataRole.UserRole)
        self.current_dicom = dicom_file
        self.show_preview(dicom_file)
        self.show_details(dicom_file)
    
    def on_image_double_clicked(self, item: QListWidgetItem):
        """Handle image double-click - open in image viewer."""
        dicom_file: DicomFileInfo = item.data(Qt.ItemDataRole.UserRole)
        self.open_in_viewer(dicom_file)
    
    def open_current_in_viewer(self):
        """Open the currently selected image in an external viewer."""
        if hasattr(self, 'current_dicom') and self.current_dicom:
            self.open_in_viewer(self.current_dicom)
        else:
            self.statusBar().showMessage("No image selected")
    
    def open_in_viewer(self, dicom_file: DicomFileInfo):
        """Open a DICOM image in an external image viewer with annotations."""
        import subprocess
        import tempfile
        import time
        
        try:
            # First, try to export as PNG for better compatibility with image viewers
            ds = pydicom.dcmread(dicom_file.path)
            
            if hasattr(ds, 'PixelData'):
                try:
                    ds.decompress()
                except Exception:
                    pass
                
                try:
                    pixel_array = ds.pixel_array
                    
                    # Apply windowing if available
                    if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                        wc = ds.WindowCenter
                        ww = ds.WindowWidth
                        if isinstance(wc, pydicom.multival.MultiValue):
                            wc = wc[0]
                        if isinstance(ww, pydicom.multival.MultiValue):
                            ww = ww[0]
                        min_val = wc - ww / 2
                        max_val = wc + ww / 2
                        pixel_array = np.clip(pixel_array, min_val, max_val)
                    
                    # Normalize to 8-bit
                    if pixel_array.max() > pixel_array.min():
                        pixel_array = ((pixel_array - pixel_array.min()) / 
                                      (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
                    else:
                        pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
                    
                    # Convert to PIL Image
                    from PIL import Image, ImageDraw
                    
                    # Handle grayscale vs RGB
                    if len(pixel_array.shape) == 2:
                        img = Image.fromarray(pixel_array, mode='L').convert('RGB')
                    else:
                        img = Image.fromarray(pixel_array)
                    
                    # Debug info
                    has_bbox_checked = self.show_bbox.isChecked()
                    has_annotations = len(dicom_file.annotations) if dicom_file.annotations else 0
                    
                    # Draw bounding boxes if enabled
                    if has_bbox_checked and has_annotations > 0:
                        # Get column selections for debug
                        x1_col = self.bbox_x1_col.currentText()
                        y1_col = self.bbox_y1_col.currentText()
                        x2_col = self.bbox_x2_col.currentText()
                        y2_col = self.bbox_y2_col.currentText()
                        
                        print(f"DEBUG: Drawing {has_annotations} annotations")
                        print(f"DEBUG: Columns: {x1_col}, {y1_col}, {x2_col}, {y2_col}")
                        
                        img = self.draw_annotations_pil(img, dicom_file)
                        self.statusBar().showMessage(f"Opening with {has_annotations} bounding boxes...")
                    else:
                        print(f"DEBUG: Not drawing - bbox_checked={has_bbox_checked}, annotations={has_annotations}")
                    
                    # Create temp file with unique timestamp to avoid caching
                    temp_dir = tempfile.gettempdir()
                    timestamp = int(time.time() * 1000)
                    temp_path = Path(temp_dir) / f"{dicom_file.filename}_{timestamp}.png"
                    img.save(temp_path)
                    
                    print(f"DEBUG: Saved to {temp_path}")
                    
                    # Open with default image viewer
                    subprocess.Popen(['xdg-open', str(temp_path)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.statusBar().showMessage(f"Opened {dicom_file.filename} in image viewer")
                    return
                    
                except Exception as e:
                    # If pixel extraction fails, try opening the DICOM directly
                    pass
            
            # Fallback: try to open the DICOM file directly
            subprocess.Popen(['xdg-open', str(dicom_file.path)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.statusBar().showMessage(f"Opened {dicom_file.filename}")
            
        except Exception as e:
            # Last resort: copy path to clipboard
            from PyQt6.QtWidgets import QApplication
            clipboard = QApplication.clipboard()
            clipboard.setText(str(dicom_file.path))
            self.statusBar().showMessage(f"Could not open viewer. Path copied: {dicom_file.path}")
    
    def draw_annotations_pil(self, img, dicom_file: DicomFileInfo):
        """Draw annotation overlays on PIL Image."""
        from PIL import ImageDraw
        
        # Get bounding box columns
        x1_col = self.bbox_x1_col.currentText()
        y1_col = self.bbox_y1_col.currentText()
        x2_col = self.bbox_x2_col.currentText()
        y2_col = self.bbox_y2_col.currentText()
        label_col = self.label_column.currentText()
        
        # Check if all bbox columns are selected
        if any(col.startswith("--") for col in [x1_col, y1_col, x2_col, y2_col]):
            return img
        
        # Check if there are annotations to draw
        if not dicom_file.annotations:
            return img
        
        draw = ImageDraw.Draw(img)
        ann_type = self.annotation_type.currentText()
        
        # Box color (orange/red)
        box_color = (255, 87, 34)  # #FF5722
        
        for ann in dicom_file.annotations:
            try:
                # Get coordinates - handle both dict key access methods
                x1_val = ann.get(x1_col)
                y1_val = ann.get(y1_col)
                x2_val = ann.get(x2_col)
                y2_val = ann.get(y2_col)
                
                # Convert to float, handling None/NaN
                x1 = float(x1_val) if x1_val is not None and pd.notna(x1_val) else None
                y1 = float(y1_val) if y1_val is not None and pd.notna(y1_val) else None
                x2 = float(x2_val) if x2_val is not None and pd.notna(x2_val) else None
                y2 = float(y2_val) if y2_val is not None and pd.notna(y2_val) else None
                
                if None in [x1, y1, x2, y2]:
                    continue
                
                # Handle different bbox formats
                if "x,y,w,h" in ann_type:
                    # x,y,w,h format - x2 is width, y2 is height
                    w, h = x2, y2
                    x2, y2 = x1 + w, y1 + h
                
                # Draw rectangle (outline) with thicker line
                draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline=box_color, width=3)
                
                # Draw label if enabled
                if self.show_labels.isChecked() and not label_col.startswith("--"):
                    label_val = ann.get(label_col, "")
                    label = str(label_val) if label_val is not None and pd.notna(label_val) else ""
                    if label and label != "nan":
                        # Draw label background
                        try:
                            text_bbox = draw.textbbox((int(x1), int(y1) - 18), label)
                            draw.rectangle(text_bbox, fill=box_color)
                            draw.text((int(x1), int(y1) - 18), label, fill=(255, 255, 255))
                        except Exception:
                            # Fallback for older PIL versions
                            draw.text((int(x1), int(y1) - 15), label, fill=box_color)
                        
            except (ValueError, TypeError) as e:
                continue
        
        return img
    
    def show_preview(self, dicom_file: DicomFileInfo):
        """Show image preview."""
        try:
            ds = pydicom.dcmread(dicom_file.path)
            
            # Get pixel data
            if not hasattr(ds, 'PixelData'):
                self.preview_label.setText("No pixel data in this DICOM file")
                return
            
            # Try to decompress if needed
            try:
                ds.decompress()
            except Exception:
                pass  # May already be uncompressed or we can still try pixel_array
            
            try:
                pixel_array = ds.pixel_array
            except Exception as decomp_error:
                # Show more helpful error message for decompression failures
                transfer_syntax = getattr(ds.file_meta, 'TransferSyntaxUID', 'Unknown')
                self.preview_label.setText(
                    f"Cannot decompress image\n\n"
                    f"Transfer Syntax: {transfer_syntax}\n\n"
                    f"This image uses compression that requires\n"
                    f"additional packages (gdcm or pylibjpeg).\n\n"
                    f"Try: pip install python-gdcm"
                )
                return
            
            # Apply windowing if available
            if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                wc = ds.WindowCenter
                ww = ds.WindowWidth
                if isinstance(wc, pydicom.multival.MultiValue):
                    wc = wc[0]
                if isinstance(ww, pydicom.multival.MultiValue):
                    ww = ww[0]
                
                min_val = wc - ww / 2
                max_val = wc + ww / 2
                pixel_array = np.clip(pixel_array, min_val, max_val)
            
            # Normalize to 8-bit
            if pixel_array.max() > pixel_array.min():
                pixel_array = ((pixel_array - pixel_array.min()) / 
                              (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
            else:
                pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
            
            # Create QImage
            height, width = pixel_array.shape[:2]
            if len(pixel_array.shape) == 2:
                # Grayscale
                qimage = QImage(pixel_array.data, width, height, width, QImage.Format.Format_Grayscale8)
            else:
                # RGB
                bytes_per_line = 3 * width
                qimage = QImage(pixel_array.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            
            # Convert to pixmap and scale
            pixmap = QPixmap.fromImage(qimage)
            
            # Draw bounding boxes if enabled
            if self.show_bbox.isChecked() and dicom_file.annotations:
                pixmap = self.draw_annotations(pixmap, dicom_file, width, height)
            
            # Scale to fit preview
            preview_size = self.preview_label.size()
            scaled_pixmap = pixmap.scaled(
                preview_size.width() - 10, preview_size.height() - 10,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.preview_label.setPixmap(scaled_pixmap)
            
            # Update info
            info_parts = [f"Size: {width}x{height}"]
            if dicom_file.modality:
                info_parts.append(f"Modality: {dicom_file.modality}")
            if dicom_file.patient_id:
                info_parts.append(f"Patient: {dicom_file.patient_id}")
            self.image_info_label.setText(" | ".join(info_parts))
            
        except Exception as e:
            self.preview_label.setText(f"Error loading image:\n{str(e)}")
            self.image_info_label.setText("")
    
    def draw_annotations(self, pixmap: QPixmap, dicom_file: DicomFileInfo, 
                         img_width: int, img_height: int) -> QPixmap:
        """Draw annotation overlays on pixmap."""
        # Get bounding box columns
        x1_col = self.bbox_x1_col.currentText()
        y1_col = self.bbox_y1_col.currentText()
        x2_col = self.bbox_x2_col.currentText()
        y2_col = self.bbox_y2_col.currentText()
        label_col = self.label_column.currentText()
        
        if any(col.startswith("--") for col in [x1_col, y1_col, x2_col, y2_col]):
            return pixmap
        
        painter = QPainter(pixmap)
        pen = QPen(QColor("#FF5722"))
        pen.setWidth(2)
        painter.setPen(pen)
        
        ann_type = self.annotation_type.currentText()
        
        for ann in dicom_file.annotations:
            try:
                x1 = float(ann.get(x1_col, 0)) if pd.notna(ann.get(x1_col)) else None
                y1 = float(ann.get(y1_col, 0)) if pd.notna(ann.get(y1_col)) else None
                x2 = float(ann.get(x2_col, 0)) if pd.notna(ann.get(x2_col)) else None
                y2 = float(ann.get(y2_col, 0)) if pd.notna(ann.get(y2_col)) else None
                
                if None in [x1, y1, x2, y2]:
                    continue
                
                # Handle different bbox formats
                if "x,y,w,h" in ann_type:
                    # x,y,w,h format
                    w, h = x2, y2
                    x2, y2 = x1 + w, y1 + h
                
                # Draw rectangle
                painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                
                # Draw label
                if self.show_labels.isChecked() and not label_col.startswith("--"):
                    label = str(ann.get(label_col, ""))
                    if label:
                        painter.drawText(int(x1), int(y1) - 5, label)
                        
            except (ValueError, TypeError):
                continue
        
        painter.end()
        return pixmap
    
    def refresh_preview(self):
        """Refresh the current preview."""
        if hasattr(self, 'current_dicom'):
            self.show_preview(self.current_dicom)
    
    def show_details(self, dicom_file: DicomFileInfo):
        """Show annotation details."""
        # Build details text
        lines = []
        lines.append(f"📁 File: {dicom_file.filename}")
        lines.append(f"📍 Path: {dicom_file.path}")
        
        if dicom_file.sop_instance_uid:
            lines.append(f"🆔 SOP UID: {dicom_file.sop_instance_uid}")
        if dicom_file.patient_id:
            lines.append(f"👤 Patient: {dicom_file.patient_id}")
        if dicom_file.modality:
            lines.append(f"📋 Modality: {dicom_file.modality}")
        if dicom_file.rows and dicom_file.cols:
            lines.append(f"📐 Size: {dicom_file.cols}x{dicom_file.rows}")
        
        lines.append("")
        lines.append(f"✓ Matched: {'Yes' if dicom_file.matched else 'No'}")
        lines.append(f"📝 Annotations: {len(dicom_file.annotations)}")
        
        if dicom_file.annotations:
            lines.append("")
            lines.append("─── Annotation Details ───")
            for i, ann in enumerate(dicom_file.annotations[:5]):  # Limit to first 5
                lines.append(f"\n[{i+1}]")
                for key, value in list(ann.items())[:6]:  # Limit fields shown
                    lines.append(f"  {key}: {value}")
            if len(dicom_file.annotations) > 5:
                lines.append(f"\n... and {len(dicom_file.annotations) - 5} more")
        
        self.details_text.setText("\n".join(lines))
    
    def export_filtered(self):
        """Export filtered images to a folder."""
        output_folder = QFileDialog.getExistingDirectory(
            self, "Select Export Folder", "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not output_folder:
            return
        
        output_path = Path(output_folder)
        
        # Get filtered items
        items_to_export = []
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            dicom_file = item.data(Qt.ItemDataRole.UserRole)
            items_to_export.append(dicom_file)
        
        if not items_to_export:
            QMessageBox.warning(self, "Warning", "No images to export.")
            return
        
        # Export
        import shutil
        exported = 0
        errors = 0
        
        for dicom_file in items_to_export:
            try:
                dest = output_path / dicom_file.path.name
                shutil.copy2(dicom_file.path, dest)
                exported += 1
            except Exception:
                errors += 1
        
        # Export annotations CSV
        if self.combined_csv is not None:
            try:
                id_col = self.id_column.currentText()
                if not id_col.startswith("--"):
                    exported_ids = [f.filename for f in items_to_export]
                    exported_ids.extend([f.sop_instance_uid for f in items_to_export if f.sop_instance_uid])
                    
                    filtered_csv = self.combined_csv[
                        self.combined_csv[id_col].astype(str).isin(exported_ids)
                    ]
                    filtered_csv.to_csv(output_path / "annotations.csv", index=False)
            except Exception:
                pass
        
        QMessageBox.information(
            self, "Export Complete",
            f"Exported {exported} images to:\n{output_folder}\n\n"
            f"Errors: {errors}"
        )
    
    def clear_all(self):
        """Clear all data."""
        self.dicom_files = []
        self.csv_data = {}
        self.combined_csv = None
        self.dicom_path.setText("")
        self.csv_list.setText("")
        self.image_list.clear()
        self.details_table.setRowCount(0)
        self.preview_label.clear()
        self.preview_label.setText("Select an image to preview")
        self.stats_label.setText("No data loaded")
        self.update_column_dropdowns()
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About DICOM Explorer",
            "<h2>DICOM Explorer</h2>"
            "<p>Version 0.1.0</p>"
            "<p>A tool for browsing DICOM images and matching them with CSV annotations.</p>"
            "<p>Features:</p>"
            "<ul>"
            "<li>Scan DICOM folders</li>"
            "<li>Load multiple CSV annotation files</li>"
            "<li>Flexible column mapping</li>"
            "<li>Search and filter images</li>"
            "<li>Preview with bounding box overlay</li>"
            "<li>Export filtered subsets</li>"
            "</ul>"
            "<p>Part of the CAD Preprocess toolkit.</p>"
        )


def main():
    """Main entry point for DICOM Explorer."""
    app = QApplication(sys.argv)
    app.setApplicationName("DICOM Explorer")
    app.setApplicationVersion("0.1.0")
    app.setStyle("Fusion")
    
    window = DicomExplorerGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

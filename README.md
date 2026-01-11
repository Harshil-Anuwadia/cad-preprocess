<h1 align="center">CAD Preprocess</h1>

<p align="center">
  <strong>Medical DICOM Image Preprocessing Pipeline for CAD Systems</strong>
</p>

<p align="center">
  A production-ready Python library for standardized DICOM preprocessing in<br>
  Computer-Aided Detection and Diagnosis (CAD) systems.
</p>

<p align="center">
  <a href="https://harshil-anuwadia.github.io/cad-preprocess/"><strong>Documentation</strong></a> ·
  <a href="https://harshil-anuwadia.github.io/cad-preprocess/docs/api/index.html">API Reference</a> ·
  <a href="https://harshil-anuwadia.github.io/cad-preprocess/docs/examples.html">Examples</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/pydicom-2.3+-orange?style=flat-square" alt="pydicom">
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey?style=flat-square" alt="Platform">
</p>

---

## Overview

CAD Preprocess standardizes DICOM image preprocessing for machine learning pipelines. Write your preprocessing logic once, use it identically across training, inference, and production environments.

### Key Features

| Feature | Description |
|---------|-------------|
| **Reproducible Processing** | Configuration hashing ensures identical results between training and inference |
| **DICOM Support** | Automatic file discovery, validation, decompression (JPEG Lossless, JPEG 2000) |
| **Flexible Output** | PNG, JPEG, or NumPy arrays with configurable naming policies |
| **Metadata Extraction** | Profile-based extraction: minimal, patient, geometry, ML, acquisition |
| **Dual Interface** | Python API and command-line tool |
| **GUI Explorer** | Interactive DICOM browser with CSV filtering and bounding box overlay |

## Installation

### From Source

```bash
git clone https://github.com/Harshil-Anuwadia/cad-preprocess.git
cd cad-preprocess
pip install -e .
```

### Debian Package (Ubuntu/Debian)

```bash
# Build the self-contained .deb package
./build_deb.sh

# Install
sudo dpkg -i cad-preprocess_0.1.0_all.deb
```

### Dependencies

```bash
pip install pydicom numpy Pillow PyYAML scikit-image pylibjpeg pylibjpeg-libjpeg pandas
# For GUI
pip install PyQt6
```

## Usage

### Command Line

```bash
# Basic usage
cad-preprocess -i ./dicoms -o ./output

# With configuration file
cad-preprocess -i ./dicoms -o ./output -c config.yaml

# ML-focused metadata extraction
cad-preprocess -i ./dicoms -o ./output --metadata-profile ml

# Custom CT windowing
cad-preprocess -i ./dicoms -o ./output --window-center 40 --window-width 400

# Preview without processing
cad-preprocess -i ./dicoms -o ./output --dry-run

# See all options
cad-preprocess --help
```

### DICOM Explorer (GUI)

```bash
# Launch the interactive DICOM browser
cad-preprocess-explorer
```

Features:
- Browse and preview DICOM images
- Filter by CSV annotations
- Overlay bounding boxes from coordinate data
- Double-click to open in external viewer

### Python API

```python
from cad_preprocess import preprocess, CADPreprocessor

# Quick single-file processing
result = preprocess("scan.dcm", "output/")
print(f"Processed: {result.sop_instance_uid}")
print(f"Shape: {result.image.shape}")

# Batch processing with configuration
processor = CADPreprocessor.from_config("config.yaml")
results = processor.process_directory("dicoms/", "output/")

# Get config hash for reproducibility tracking
print(f"Config hash: {processor.config_hash}")
```

### Configuration

```yaml
# config.yaml
preprocessing:
  windowing:
    strategy: fixed_window
    window_center: 40
    window_width: 400
  normalization: min_max
  resizing:
    target_height: 512
    target_width: 512
    preserve_aspect_ratio: true

metadata:
  profiles:
    - minimal
    - ml

output:
  naming_policy: sop_instance_uid
  format: png
```

## Output Structure

```
output/
├── images/
│   ├── 1.2.840.113619.2.55.3.png
│   └── ...
├── metadata/
│   ├── 1.2.840.113619.2.55.3.json
│   └── ...
├── logs/
│   └── processing_log.json
└── manifest.json
```

## CLI Reference

| Option | Description |
|--------|-------------|
| `-i, --input` | Input DICOM file or directory |
| `-o, --output` | Output directory |
| `-c, --config` | YAML configuration file |
| `-m, --metadata-profile` | Metadata profile: minimal, patient, geometry, ml, acquisition, all |
| `--window-center` | CT window center value |
| `--window-width` | CT window width value |
| `--target-size` | Output image dimensions |
| `--overwrite` | Overwrite existing output files |
| `--dry-run` | Preview files without processing |
| `-l, --log-level` | Logging verbosity: debug, info, warning, error |

## Requirements

| Package | Version |
|---------|---------|
| Python | >= 3.9 |
| pydicom | >= 2.3.0 |
| numpy | >= 1.21.0 |
| Pillow | >= 9.0.0 |
| PyYAML | >= 6.0 |
| scikit-image | >= 0.19.0 |

## Project Structure

```
cad-preprocess/
├── src/cad_preprocess/
│   ├── cli.py                  # Command-line interface
│   ├── explorer.py             # DICOM Explorer GUI
│   ├── config.py               # Configuration management
│   ├── input_handler.py        # DICOM discovery & validation
│   ├── preprocessing_engine.py # Image processing pipeline
│   ├── metadata_extractor.py   # Metadata extraction
│   ├── output_writer.py        # File output handling
│   ├── logging_utils.py        # Logging & statistics
│   ├── integration.py          # High-level API
│   └── api.py                  # Simple API functions
├── tests/                      # Unit tests
├── docs/                       # Documentation website
├── debian/                     # Debian packaging files
├── build_deb.sh               # Self-contained .deb builder
└── pyproject.toml             # Project configuration
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>CAD Preprocess</strong> — Standardized DICOM preprocessing for medical imaging AI
</p>

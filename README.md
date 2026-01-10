# CAD Preprocess

<p align="center">
  <img src="cad-preprocess-white.png" alt="CAD Preprocess Logo" width="300">
</p>

**A Python library for DICOM preprocessing in CAD (Computer-Aided Detection/Diagnosis) systems.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

CAD Preprocess is a Python library that standardizes DICOM image preprocessing so it doesn't need to be rewritten across training, inference, and UI workflows. Write your preprocessing logic once, use it everywhere.

### Key Features

- 🔍 **DICOM Discovery** - Recursive scanning with validation
- 🖼️ **Deterministic Preprocessing** - Identical results for train & inference
- 📊 **Metadata Extraction** - Profile-based (minimal, patient, geometry, ML, acquisition)
- ⚙️ **Configurable** - YAML configuration with CLI overrides
- 🐍 **Easy Integration** - Use as a library or CLI tool

## Installation

### From Source

```bash
git clone https://github.com/Harshil-Anuwadia/cad-preprocess.git
cd cad-preprocess
pip install -e .
```

### Using pip (with dependencies)

```bash
pip install pydicom numpy Pillow PyYAML scikit-image
pip install -e .
```

## Quick Start

### CLI Usage

```bash
# Process single file
cad-preprocess -i image.dcm -o ./output

# Process directory
cad-preprocess -i ./dicoms -o ./output

# With configuration
cad-preprocess -i ./dicoms -o ./output -c config.yaml

# With metadata profile
cad-preprocess -i ./dicoms -o ./output --metadata-profile ml

# Custom windowing (CT)
cad-preprocess -i ./dicoms -o ./output --window-center 40 --window-width 400

# Preview without processing
cad-preprocess -i ./dicoms -o ./output --dry-run
```

### Python API

```python
from cad_preprocess import preprocess, CADPreprocessor

# Simple one-liner
result = preprocess("image.dcm", "output/")
print(f"Shape: {result.image.shape}")

# Batch processing with guaranteed reproducibility
processor = CADPreprocessor.from_config("config.yaml")
batch = processor.process_directory("dicoms/", "output/")

# Store config hash for train/inference consistency
config_hash = processor.config_hash
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
│   └── processing_log_20260110.json
└── manifest.json
```

## Configuration

Create `config.yaml`:

```yaml
preprocessing:
  windowing:
    strategy: use_dicom_window  # or "fixed_window"
    window_center: 40           # for fixed_window
    window_width: 400           # for fixed_window
  normalization: min_max        # or "z_score"
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
  overwrite_policy: skip
```

## CLI Options

| Option | Short | Description |
|--------|-------|-------------|
| `--input` | `-i` | Input file or directory |
| `--output` | `-o` | Output directory |
| `--config` | `-c` | YAML configuration file |
| `--metadata-profile` | `-m` | Metadata profile (minimal/patient/geometry/ml/acquisition/all) |
| `--overwrite` | | Overwrite existing files |
| `--log-level` | `-l` | Logging level (debug/info/warning/error) |
| `--dry-run` | | Preview without processing |
| `--window-center` | | Fixed window center value |
| `--window-width` | | Fixed window width value |
| `--target-size` | | Target size (height width) |
| `--normalization` | | Normalization method |

## Building .deb Package

```bash
./build-deb.sh
# Creates: cad-preprocess_0.1.0_amd64.deb
```

## Project Structure

```
cad-preprocess/
├── src/cad_preprocess/
│   ├── __init__.py
│   ├── cli.py                 # Command-line interface
│   ├── config.py              # Configuration system
│   ├── input_handler.py       # DICOM discovery & validation
│   ├── preprocessing_engine.py # Image processing pipeline
│   ├── metadata_extractor.py  # Metadata extraction
│   ├── output_writer.py       # Output persistence
│   ├── logging_utils.py       # Logging & error handling
│   └── integration.py         # High-level API
├── tests/
├── debian/
├── build-deb.sh
├── config.example.yaml
└── pyproject.toml
```

## Use Cases

- **ML Training Pipeline** - Preprocess training data with traceable configuration
- **Model Inference** - Identical preprocessing guaranteed via config hash
- **CAD Visualization** - Consistent image display across applications

## Requirements

- Python >= 3.9
- pydicom >= 2.3.0
- numpy >= 1.21.0
- Pillow >= 9.0.0
- PyYAML >= 6.0
- scikit-image >= 0.19.0

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read our contributing guidelines first.

---

**Made for medical imaging professionals and ML engineers** 🏥🤖

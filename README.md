# CAD Preprocess

<p align="center">
  <img src="cad-preprocess.png" alt="CAD Preprocess Logo" width="300">
</p>

A Python library for DICOM preprocessing in CAD systems.

## Installation

```bash
git clone https://github.com/Harshil-Anuwadia/cad-preprocess.git
cd cad-preprocess
pip install -e .
```

## Quick Start

### Command Line

```bash
cad-preprocess -i ./dicoms -o ./output
cad-preprocess -i ./dicoms -o ./output -c config.yaml
cad-preprocess -i ./dicoms -o ./output --metadata-profile ml
```

### Python

```python
from cad_preprocess import preprocess, CADPreprocessor

# Single file
result = preprocess("image.dcm", "output/")

# Batch processing
processor = CADPreprocessor.from_config("config.yaml")
results = processor.process_directory("dicoms/", "output/")
```

## Features

- DICOM file discovery and validation
- Deterministic preprocessing pipeline
- Profile-based metadata extraction
- YAML configuration support
- CLI and Python API

## Requirements

- Python >= 3.9
- pydicom, numpy, Pillow, PyYAML, scikit-image

## License

MIT

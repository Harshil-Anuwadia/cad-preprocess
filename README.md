# CAD-Preprocess

**Standardized DICOM Preprocessing for Computer-Aided Detection (CAD) Systems.**

CAD-Preprocess is a high-performance Python library and CLI tool designed to provide deterministic, reproducible preprocessing of medical DICOM images. It ensures that the exact same image transformations are applied across training, research, and production inference environments.

## 🚀 Key Features

*   **Deterministic Pipeline**: Consistent windowing (VOI LUT), intensity normalization, and resizing.
*   **High Performance**: Fully parallelized batch processing utilizing multi-core CPUs.
*   **Universal Installers**: One-line installation for Linux (Arch, Debian, Fedora) and Windows.
*   **GUI Explorer**: Integrated tool to visualize preprocessing results and DICOM metadata.
*   **Native Packaging**: Automatically builds and installs `.deb` and Arch Linux packages.
*   **Built-in Benchmarking**: Advanced performance diagnostic suite with auto-tuning recommendations.

## 🛠️ Installation

### Linux (One-Liner)
```bash
curl -sSL https://raw.githubusercontent.com/Harshil-Anuwadia/cad-preprocess/main/install.sh | bash
```

### Windows (PowerShell)
```powershell
irm https://raw.githubusercontent.com/Harshil-Anuwadia/cad-preprocess/main/install.ps1 | iex
```

## 📖 Usage

### CLI
```bash
# Process a directory of DICOMs
cad-preprocess ./raw_dicoms ./output_processed

# Process with specific windowing and resizing
cad-preprocess -i ./input -o ./output --window-center 40 --window-width 400 --target-size 512 512
```

### Python API
```python
from cad_preprocess import preprocess, CADPreprocessor

# Simple one-line processing
result = preprocess('image.dcm', 'output/')

# Advanced batch processing
processor = CADPreprocessor(config=my_config, num_workers=8)
processor.process_directory('./input_dir', './output_dir')
```

## 📊 Performance
Run the built-in benchmark to measure your system's throughput:
```bash
cad-preprocess-benchmark
```

## ⚖️ License
MIT | Copyright (c) 2024-2026 Harshil Anuwadia

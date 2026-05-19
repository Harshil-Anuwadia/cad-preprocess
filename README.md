# CAD Preprocess

A Python toolkit for standardized DICOM preprocessing. Built to ensure medical imaging models receive consistent, high-quality input during both training and inference.

## Why this exists?

Medical AI models are notoriously sensitive to how DICOM pixels are handled. Subtle differences in windowing, normalization, or resizing between your training script and your production inference server can lead to catastrophic performance drift. 

`cad-preprocess` solves this by providing a deterministic, configuration-driven pipeline that handles the "boring but critical" parts of medical imaging: decompression, Hounsfield Unit scaling, VOI LUT application, and standardized resizing.

## Key Capabilities

*   **Deterministic Pipeline:** Hashing-based configuration checks ensure training/inference parity.
*   **Decompression:** Built-in support for JPEG Lossless, JPEG 2000, and RLE via `pydicom` handlers.
*   **Windowing:** Proper application of DICOM VOI LUTs or custom fixed windowing (e.g., Lung/Soft Tissue/Bone).
*   **Metadata:** Profile-based extraction into JSON/CSV (Patient, Geometry, Acquisition, or ML-specific fields).
*   **Dual-Interface:** Use as a Python library or a standalone CLI for batch processing.
*   **Visual Validation:** Includes `cad-preprocess-explorer` for interactive DICOM browsing and annotation matching.

## Installation

### For Users (CLI/GUI)

The easiest way to get the tools without messing with your system Python:

```bash
# Core CLI tools
pipx install cad-preprocess

# With GUI (Explorer) support
pipx install "cad-preprocess[gui]"
```

### For Developers (Library)

```bash
pip install cad-preprocess
```

*Note: For image decompression support, ensure you have `gdcm` or `pylibjpeg` handlers installed.*

### Standalone Binaries
If you don't use Python, grab the pre-compiled binaries for Windows or Linux from the [Releases](https://github.com/Harshil-Anuwadia/cad-preprocess/releases) page.

## Quick Start

### Python API
Standardize a DICOM file in three lines:

```python
from cad_preprocess import preprocess

result = preprocess("input.dcm", "output_dir/")
print(f"Processed SOP: {result.processed_files[0]}")
```

### Command Line
Batch process a directory with a specific metadata profile:

```bash
cad-preprocess -i ./raw_data -o ./clean_data --metadata-profile ml --target-size 512 512
```

### Interactive Explorer
Launch the GUI to browse images and match them with CSV labels:

```bash
cad-preprocess-explorer
```

## Project Structure

*   `src/cad_preprocess/`: Core logic (engine, metadata, IO).
*   `tests/`: Unit tests and integration checks.
*   `build_standalone.py`: PyInstaller bundling script for cross-platform binaries.

## License

MIT - See [LICENSE](LICENSE) for details.

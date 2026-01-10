"""
Preprocessing Engine Module for CAD Preprocess.

This module provides deterministic transformation of DICOM pixel data
into model-ready images.

Processing steps (in order):
1. Load DICOM
2. Extract pixel_array
3. Apply RescaleSlope and RescaleIntercept
4. Apply windowing (configurable strategy)
5. Normalize intensity values
6. Resize image
7. Convert to output format

Windowing strategies:
- use_dicom_window: Use WindowCenter/WindowWidth from DICOM metadata
- fixed_window: Use user-specified window values

Normalization methods:
- min_max: Scale to [0, 1] range
- z_score: Standardize to zero mean and unit variance
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pydicom
from numpy.typing import NDArray
from PIL import Image

# Configure module logger
logger = logging.getLogger(__name__)


class WindowingStrategy(str, Enum):
    """Windowing strategy options."""

    USE_DICOM_WINDOW = "use_dicom_window"
    FIXED_WINDOW = "fixed_window"


class NormalizationMethod(str, Enum):
    """Normalization method options."""

    MIN_MAX = "min_max"
    Z_SCORE = "z_score"


class InterpolationMethod(str, Enum):
    """Interpolation methods for resizing."""

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"

    def to_pil(self) -> int:
        """Convert to PIL resampling constant."""
        mapping = {
            InterpolationMethod.NEAREST: Image.Resampling.NEAREST,
            InterpolationMethod.BILINEAR: Image.Resampling.BILINEAR,
            InterpolationMethod.BICUBIC: Image.Resampling.BICUBIC,
            InterpolationMethod.LANCZOS: Image.Resampling.LANCZOS,
        }
        return mapping[self]


@dataclass
class WindowingConfig:
    """Configuration for windowing operation."""

    strategy: WindowingStrategy = WindowingStrategy.USE_DICOM_WINDOW
    window_center: Optional[float] = None
    window_width: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if isinstance(self.strategy, str):
            self.strategy = WindowingStrategy(self.strategy)

        if self.strategy == WindowingStrategy.FIXED_WINDOW:
            if self.window_center is None or self.window_width is None:
                raise ValueError(
                    "window_center and window_width must be provided for fixed_window strategy"
                )


@dataclass
class ResizingConfig:
    """Configuration for resizing operation."""

    enabled: bool = True
    target_size: Tuple[int, int] = (1024, 1024)
    keep_aspect_ratio: bool = True
    interpolation: InterpolationMethod = InterpolationMethod.BILINEAR
    padding_value: int = 0

    def __post_init__(self) -> None:
        """Validate configuration."""
        if isinstance(self.interpolation, str):
            self.interpolation = InterpolationMethod(self.interpolation)


@dataclass
class PreprocessingConfig:
    """Complete configuration for preprocessing pipeline."""

    windowing: WindowingConfig = field(default_factory=WindowingConfig)
    normalization: NormalizationMethod = NormalizationMethod.MIN_MAX
    resizing: ResizingConfig = field(default_factory=ResizingConfig)
    output_dtype: str = "uint8"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if isinstance(self.normalization, str):
            self.normalization = NormalizationMethod(self.normalization)

        if isinstance(self.windowing, dict):
            self.windowing = WindowingConfig(**self.windowing)

        if isinstance(self.resizing, dict):
            self.resizing = ResizingConfig(**self.resizing)


@dataclass
class PreprocessingResult:
    """Result of preprocessing operation."""

    image: NDArray[np.uint8]
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    applied_window_center: Optional[float] = None
    applied_window_width: Optional[float] = None
    rescale_slope: float = 1.0
    rescale_intercept: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


def load_dicom(file_path: Path) -> pydicom.Dataset:
    """
    Load a DICOM file.

    Args:
        file_path: Path to the DICOM file.

    Returns:
        pydicom Dataset object.

    Raises:
        FileNotFoundError: If file does not exist.
        pydicom.errors.InvalidDicomError: If file is not valid DICOM.
    """
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise FileNotFoundError(f"DICOM file not found: {file_path}")

    logger.debug(f"Loading DICOM file: {file_path}")
    ds = pydicom.dcmread(file_path)

    return ds


def extract_pixel_array(ds: pydicom.Dataset) -> NDArray:
    """
    Extract pixel array from DICOM dataset.

    Args:
        ds: pydicom Dataset object.

    Returns:
        NumPy array of pixel data.

    Raises:
        ValueError: If PixelData is missing.
    """
    if not hasattr(ds, "PixelData"):
        raise ValueError("DICOM dataset missing PixelData")

    pixel_array = ds.pixel_array.astype(np.float64)
    logger.debug(f"Extracted pixel array with shape: {pixel_array.shape}")

    return pixel_array


def apply_rescale(
    pixel_array: NDArray,
    ds: pydicom.Dataset,
) -> Tuple[NDArray, float, float]:
    """
    Apply RescaleSlope and RescaleIntercept to pixel array.

    The formula is: output = pixel_value * RescaleSlope + RescaleIntercept

    For CT images, this converts to Hounsfield Units (HU).
    For other modalities, default values (slope=1, intercept=0) are used.

    Args:
        pixel_array: Input pixel array.
        ds: pydicom Dataset object.

    Returns:
        Tuple of (rescaled array, slope used, intercept used).
    """
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    if slope != 1.0 or intercept != 0.0:
        logger.debug(f"Applying rescale: slope={slope}, intercept={intercept}")
        rescaled = pixel_array * slope + intercept
    else:
        rescaled = pixel_array

    return rescaled, slope, intercept


def get_dicom_window(ds: pydicom.Dataset) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract window center and width from DICOM metadata.

    Handles both single values and lists (uses first value if multiple).

    Args:
        ds: pydicom Dataset object.

    Returns:
        Tuple of (window_center, window_width), or (None, None) if not present.
    """
    window_center = getattr(ds, "WindowCenter", None)
    window_width = getattr(ds, "WindowWidth", None)

    # Handle multi-valued window settings (take first)
    if window_center is not None:
        if hasattr(window_center, "__iter__") and not isinstance(window_center, str):
            window_center = float(window_center[0])
        else:
            window_center = float(window_center)

    if window_width is not None:
        if hasattr(window_width, "__iter__") and not isinstance(window_width, str):
            window_width = float(window_width[0])
        else:
            window_width = float(window_width)

    return window_center, window_width


def apply_windowing(
    pixel_array: NDArray,
    window_center: float,
    window_width: float,
) -> NDArray:
    """
    Apply windowing (VOI LUT) to pixel array.

    Windowing clips pixel values to a range centered at window_center
    with width window_width, then scales to [0, 1].

    Args:
        pixel_array: Input pixel array (typically in HU or raw values).
        window_center: Center of the window.
        window_width: Width of the window.

    Returns:
        Windowed array scaled to [0, 1] range.
    """
    logger.debug(f"Applying windowing: center={window_center}, width={window_width}")

    # Calculate window bounds
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2

    # Apply windowing
    windowed = np.clip(pixel_array, window_min, window_max)

    # Scale to [0, 1]
    if window_max > window_min:
        windowed = (windowed - window_min) / (window_max - window_min)
    else:
        windowed = np.zeros_like(pixel_array)

    return windowed


def apply_windowing_with_config(
    pixel_array: NDArray,
    ds: pydicom.Dataset,
    config: WindowingConfig,
) -> Tuple[NDArray, Optional[float], Optional[float]]:
    """
    Apply windowing based on configuration.

    Args:
        pixel_array: Input pixel array.
        ds: pydicom Dataset object.
        config: Windowing configuration.

    Returns:
        Tuple of (windowed array, window_center used, window_width used).
    """
    if config.strategy == WindowingStrategy.FIXED_WINDOW:
        window_center = config.window_center
        window_width = config.window_width
    else:
        # Use DICOM window
        window_center, window_width = get_dicom_window(ds)

        if window_center is None or window_width is None:
            # Fall back to min/max if DICOM window not available
            logger.warning("DICOM window not available, using min/max of image")
            img_min = float(np.min(pixel_array))
            img_max = float(np.max(pixel_array))
            window_center = (img_max + img_min) / 2
            window_width = img_max - img_min

    windowed = apply_windowing(pixel_array, window_center, window_width)

    return windowed, window_center, window_width


def normalize_min_max(pixel_array: NDArray) -> NDArray:
    """
    Normalize pixel array using min-max scaling to [0, 1] range.

    Args:
        pixel_array: Input pixel array.

    Returns:
        Normalized array in [0, 1] range.
    """
    arr_min = np.min(pixel_array)
    arr_max = np.max(pixel_array)

    if arr_max > arr_min:
        normalized = (pixel_array - arr_min) / (arr_max - arr_min)
    else:
        normalized = np.zeros_like(pixel_array)

    logger.debug(f"Min-max normalization: min={arr_min:.2f}, max={arr_max:.2f}")

    return normalized


def normalize_z_score(pixel_array: NDArray) -> NDArray:
    """
    Normalize pixel array using z-score standardization.

    Result is then clipped to [-3, 3] and scaled to [0, 1].

    Args:
        pixel_array: Input pixel array.

    Returns:
        Normalized array in [0, 1] range.
    """
    mean = np.mean(pixel_array)
    std = np.std(pixel_array)

    if std > 0:
        normalized = (pixel_array - mean) / std
    else:
        normalized = np.zeros_like(pixel_array)

    # Clip to [-3, 3] (99.7% of data for normal distribution)
    normalized = np.clip(normalized, -3, 3)

    # Scale to [0, 1]
    normalized = (normalized + 3) / 6

    logger.debug(f"Z-score normalization: mean={mean:.2f}, std={std:.2f}")

    return normalized


def normalize_intensity(
    pixel_array: NDArray,
    method: NormalizationMethod,
) -> NDArray:
    """
    Normalize pixel intensity values.

    Args:
        pixel_array: Input pixel array.
        method: Normalization method to use.

    Returns:
        Normalized array in [0, 1] range.
    """
    if method == NormalizationMethod.MIN_MAX:
        return normalize_min_max(pixel_array)
    elif method == NormalizationMethod.Z_SCORE:
        return normalize_z_score(pixel_array)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def calculate_resize_dimensions(
    original_size: Tuple[int, int],
    target_size: Tuple[int, int],
    keep_aspect_ratio: bool,
) -> Tuple[Tuple[int, int], Tuple[int, int, int, int]]:
    """
    Calculate resize dimensions and padding.

    Args:
        original_size: Original (height, width).
        target_size: Target (height, width).
        keep_aspect_ratio: Whether to preserve aspect ratio.

    Returns:
        Tuple of (new_size, padding) where:
        - new_size is (height, width) after resize
        - padding is (top, bottom, left, right) padding needed
    """
    orig_h, orig_w = original_size
    target_h, target_w = target_size

    if not keep_aspect_ratio:
        return target_size, (0, 0, 0, 0)

    # Calculate scale factor to fit within target while preserving aspect ratio
    scale_h = target_h / orig_h
    scale_w = target_w / orig_w
    scale = min(scale_h, scale_w)

    new_h = int(orig_h * scale)
    new_w = int(orig_w * scale)

    # Calculate padding
    pad_h = target_h - new_h
    pad_w = target_w - new_w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    return (new_h, new_w), (pad_top, pad_bottom, pad_left, pad_right)


def resize_image(
    pixel_array: NDArray,
    config: ResizingConfig,
) -> NDArray:
    """
    Resize image according to configuration.

    Args:
        pixel_array: Input pixel array (2D, values in [0, 1]).
        config: Resizing configuration.

    Returns:
        Resized array.
    """
    if not config.enabled:
        return pixel_array

    original_shape = pixel_array.shape
    target_h, target_w = config.target_size

    new_size, padding = calculate_resize_dimensions(
        original_size=original_shape,
        target_size=config.target_size,
        keep_aspect_ratio=config.keep_aspect_ratio,
    )

    new_h, new_w = new_size
    pad_top, pad_bottom, pad_left, pad_right = padding

    # Convert to PIL for resizing
    # Scale to 0-255 temporarily for PIL
    img_uint8 = (pixel_array * 255).astype(np.uint8)
    pil_image = Image.fromarray(img_uint8, mode="L")

    # Resize
    resample = config.interpolation.to_pil()
    pil_resized = pil_image.resize((new_w, new_h), resample=resample)

    # Convert back to numpy and scale back to [0, 1]
    resized = np.array(pil_resized).astype(np.float64) / 255.0

    # Apply padding if needed
    if any(p > 0 for p in padding):
        padded = np.full(
            (target_h, target_w),
            config.padding_value / 255.0,
            dtype=np.float64,
        )
        padded[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
        resized = padded

    logger.debug(
        f"Resized from {original_shape} to {resized.shape} "
        f"(intermediate: {new_size}, padding: {padding})"
    )

    return resized


def convert_to_output_format(
    pixel_array: NDArray,
    output_dtype: str = "uint8",
) -> NDArray:
    """
    Convert pixel array to output format.

    Args:
        pixel_array: Input array with values in [0, 1].
        output_dtype: Output data type ("uint8" or "float32").

    Returns:
        Array converted to specified dtype.
    """
    if output_dtype == "uint8":
        # Scale to [0, 255]
        output = (pixel_array * 255).astype(np.uint8)
    elif output_dtype == "float32":
        output = pixel_array.astype(np.float32)
    else:
        raise ValueError(f"Unsupported output dtype: {output_dtype}")

    return output


def save_image(
    pixel_array: NDArray[np.uint8],
    output_path: Path,
) -> None:
    """
    Save processed image to file.

    Args:
        pixel_array: Processed pixel array (uint8).
        output_path: Path to save the image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pil_image = Image.fromarray(pixel_array, mode="L")
    pil_image.save(output_path)

    logger.debug(f"Saved image to: {output_path}")


class PreprocessingEngine:
    """
    Deterministic preprocessing engine for DICOM images.

    This class provides a complete, deterministic pipeline for transforming
    DICOM pixel data into model-ready images.

    Pipeline steps:
    1. Load DICOM
    2. Extract pixel_array
    3. Apply RescaleSlope and RescaleIntercept
    4. Apply windowing
    5. Normalize intensity values
    6. Resize image
    7. Convert to output format

    Example:
        >>> engine = PreprocessingEngine()
        >>> result = engine.process("/path/to/image.dcm")
        >>> processed_image = result.image  # numpy array ready for model
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        """
        Initialize the preprocessing engine.

        Args:
            config: Preprocessing configuration. If None, uses defaults.
        """
        self.config = config or PreprocessingConfig()

    def process(
        self,
        input_path: Union[Path, str],
    ) -> PreprocessingResult:
        """
        Process a single DICOM file through the complete pipeline.

        Args:
            input_path: Path to the DICOM file.

        Returns:
            PreprocessingResult containing the processed image and metadata.
        """
        input_path = Path(input_path).resolve()

        try:
            # Step 1: Load DICOM
            ds = load_dicom(input_path)

            # Step 2: Extract pixel array
            pixel_array = extract_pixel_array(ds)
            original_shape = pixel_array.shape

            # Step 3: Apply rescale slope/intercept
            pixel_array, slope, intercept = apply_rescale(pixel_array, ds)

            # Step 4: Apply windowing
            pixel_array, wc, ww = apply_windowing_with_config(
                pixel_array, ds, self.config.windowing
            )

            # Step 5: Normalize intensity (if not already in [0,1] from windowing)
            # Note: After windowing, data is already in [0,1], so we skip if windowing was applied
            # Only apply normalization if no windowing was performed
            if wc is None and ww is None:
                pixel_array = normalize_intensity(pixel_array, self.config.normalization)

            # Step 6: Resize image
            pixel_array = resize_image(pixel_array, self.config.resizing)

            # Step 7: Convert to output format
            output_array = convert_to_output_format(pixel_array, self.config.output_dtype)

            return PreprocessingResult(
                image=output_array,
                original_shape=original_shape,
                final_shape=output_array.shape,
                applied_window_center=wc,
                applied_window_width=ww,
                rescale_slope=slope,
                rescale_intercept=intercept,
                success=True,
            )

        except Exception as e:
            logger.error(f"Preprocessing failed for {input_path}: {type(e).__name__}: {e}")
            return PreprocessingResult(
                image=np.array([], dtype=np.uint8),
                original_shape=(0, 0),
                final_shape=(0, 0),
                success=False,
                error_message=f"{type(e).__name__}: {e}",
            )

    def process_from_dataset(
        self,
        ds: pydicom.Dataset,
    ) -> PreprocessingResult:
        """
        Process a pydicom Dataset through the pipeline.

        Useful when DICOM is already loaded (e.g., from memory).

        Args:
            ds: pydicom Dataset object.

        Returns:
            PreprocessingResult containing the processed image and metadata.
        """
        try:
            # Step 2: Extract pixel array
            pixel_array = extract_pixel_array(ds)
            original_shape = pixel_array.shape

            # Step 3: Apply rescale slope/intercept
            pixel_array, slope, intercept = apply_rescale(pixel_array, ds)

            # Step 4: Apply windowing
            pixel_array, wc, ww = apply_windowing_with_config(
                pixel_array, ds, self.config.windowing
            )

            # Step 5: Normalize intensity (if windowing wasn't applied)
            if wc is None and ww is None:
                pixel_array = normalize_intensity(pixel_array, self.config.normalization)

            # Step 6: Resize image
            pixel_array = resize_image(pixel_array, self.config.resizing)

            # Step 7: Convert to output format
            output_array = convert_to_output_format(pixel_array, self.config.output_dtype)

            return PreprocessingResult(
                image=output_array,
                original_shape=original_shape,
                final_shape=output_array.shape,
                applied_window_center=wc,
                applied_window_width=ww,
                rescale_slope=slope,
                rescale_intercept=intercept,
                success=True,
            )

        except Exception as e:
            logger.error(f"Preprocessing failed: {type(e).__name__}: {e}")
            return PreprocessingResult(
                image=np.array([], dtype=np.uint8),
                original_shape=(0, 0),
                final_shape=(0, 0),
                success=False,
                error_message=f"{type(e).__name__}: {e}",
            )

    def save_result(
        self,
        result: PreprocessingResult,
        output_path: Union[Path, str],
    ) -> bool:
        """
        Save preprocessing result to file.

        Args:
            result: PreprocessingResult to save.
            output_path: Output file path (PNG).

        Returns:
            True if save was successful, False otherwise.
        """
        if not result.success:
            logger.error("Cannot save failed preprocessing result")
            return False

        try:
            save_image(result.image, Path(output_path))
            return True
        except Exception as e:
            logger.error(f"Failed to save image: {type(e).__name__}: {e}")
            return False

    def __repr__(self) -> str:
        return (
            f"PreprocessingEngine(windowing={self.config.windowing.strategy.value}, "
            f"normalization={self.config.normalization.value}, "
            f"resize={self.config.resizing.target_size if self.config.resizing.enabled else 'disabled'})"
        )

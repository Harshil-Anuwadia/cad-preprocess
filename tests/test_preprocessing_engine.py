"""
Tests for the Preprocessing Engine module.

These tests verify:
- DICOM loading and pixel extraction
- Rescale slope/intercept application
- Windowing (DICOM and fixed)
- Normalization (min_max and z_score)
- Resizing with aspect ratio preservation
- Output format conversion
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cad_preprocess.preprocessing_engine import (
    InterpolationMethod,
    NormalizationMethod,
    PreprocessingConfig,
    PreprocessingEngine,
    PreprocessingResult,
    ResizingConfig,
    WindowingConfig,
    WindowingStrategy,
    apply_rescale,
    apply_windowing,
    apply_windowing_with_config,
    calculate_resize_dimensions,
    convert_to_output_format,
    extract_pixel_array,
    get_dicom_window,
    normalize_intensity,
    normalize_min_max,
    normalize_z_score,
    resize_image,
)


class TestWindowingConfig:
    """Tests for WindowingConfig."""

    def test_default_config(self):
        """Test default windowing configuration."""
        config = WindowingConfig()
        assert config.strategy == WindowingStrategy.USE_DICOM_WINDOW
        assert config.window_center is None
        assert config.window_width is None

    def test_fixed_window_requires_values(self):
        """Test that fixed_window requires center and width."""
        with pytest.raises(ValueError, match="window_center and window_width"):
            WindowingConfig(strategy=WindowingStrategy.FIXED_WINDOW)

    def test_fixed_window_with_values(self):
        """Test fixed_window with valid values."""
        config = WindowingConfig(
            strategy=WindowingStrategy.FIXED_WINDOW,
            window_center=40,
            window_width=400,
        )
        assert config.window_center == 40
        assert config.window_width == 400

    def test_string_strategy(self):
        """Test strategy from string."""
        config = WindowingConfig(strategy="use_dicom_window")
        assert config.strategy == WindowingStrategy.USE_DICOM_WINDOW


class TestResizingConfig:
    """Tests for ResizingConfig."""

    def test_default_config(self):
        """Test default resizing configuration."""
        config = ResizingConfig()
        assert config.enabled is True
        assert config.target_size == (1024, 1024)
        assert config.keep_aspect_ratio is True
        assert config.interpolation == InterpolationMethod.BILINEAR

    def test_custom_config(self):
        """Test custom resizing configuration."""
        config = ResizingConfig(
            enabled=True,
            target_size=(512, 512),
            keep_aspect_ratio=False,
            interpolation=InterpolationMethod.BICUBIC,
        )
        assert config.target_size == (512, 512)
        assert config.keep_aspect_ratio is False


class TestPreprocessingConfig:
    """Tests for PreprocessingConfig."""

    def test_default_config(self):
        """Test default preprocessing configuration."""
        config = PreprocessingConfig()
        assert config.normalization == NormalizationMethod.MIN_MAX
        assert config.output_dtype == "uint8"

    def test_dict_windowing(self):
        """Test config with dict windowing."""
        config = PreprocessingConfig(
            windowing={
                "strategy": "fixed_window",
                "window_center": 40,
                "window_width": 400,
            }
        )
        assert config.windowing.strategy == WindowingStrategy.FIXED_WINDOW

    def test_dict_resizing(self):
        """Test config with dict resizing."""
        config = PreprocessingConfig(
            resizing={
                "enabled": True,
                "target_size": (256, 256),
            }
        )
        assert config.resizing.target_size == (256, 256)


class TestExtractPixelArray:
    """Tests for extract_pixel_array function."""

    def test_extract_valid(self):
        """Test extraction from valid dataset."""
        mock_ds = MagicMock()
        mock_ds.pixel_array = np.array([[100, 200], [150, 250]], dtype=np.uint16)

        result = extract_pixel_array(mock_ds)
        assert result.dtype == np.float64
        assert result.shape == (2, 2)

    def test_missing_pixel_data(self):
        """Test extraction fails when PixelData missing."""
        mock_ds = MagicMock(spec=[])  # No attributes

        with pytest.raises(ValueError, match="missing PixelData"):
            extract_pixel_array(mock_ds)


class TestApplyRescale:
    """Tests for apply_rescale function."""

    def test_no_rescale(self):
        """Test when no rescale parameters."""
        pixel_array = np.array([[100, 200], [150, 250]], dtype=np.float64)
        mock_ds = MagicMock(spec=[])  # No RescaleSlope/Intercept

        result, slope, intercept = apply_rescale(pixel_array, mock_ds)

        assert slope == 1.0
        assert intercept == 0.0
        np.testing.assert_array_equal(result, pixel_array)

    def test_with_rescale(self):
        """Test with rescale parameters."""
        pixel_array = np.array([[100, 200], [150, 250]], dtype=np.float64)
        mock_ds = MagicMock()
        mock_ds.RescaleSlope = 2.0
        mock_ds.RescaleIntercept = -100.0

        result, slope, intercept = apply_rescale(pixel_array, mock_ds)

        assert slope == 2.0
        assert intercept == -100.0
        expected = pixel_array * 2.0 - 100.0
        np.testing.assert_array_equal(result, expected)


class TestGetDicomWindow:
    """Tests for get_dicom_window function."""

    def test_single_value(self):
        """Test single window value."""
        mock_ds = MagicMock()
        mock_ds.WindowCenter = 40.0
        mock_ds.WindowWidth = 400.0

        wc, ww = get_dicom_window(mock_ds)
        assert wc == 40.0
        assert ww == 400.0

    def test_multi_value(self):
        """Test multiple window values (takes first)."""
        mock_ds = MagicMock()
        mock_ds.WindowCenter = [40.0, 80.0]
        mock_ds.WindowWidth = [400.0, 800.0]

        wc, ww = get_dicom_window(mock_ds)
        assert wc == 40.0
        assert ww == 400.0

    def test_no_window(self):
        """Test when no window values."""
        mock_ds = MagicMock(spec=[])

        wc, ww = get_dicom_window(mock_ds)
        assert wc is None
        assert ww is None


class TestApplyWindowing:
    """Tests for apply_windowing function."""

    def test_windowing(self):
        """Test windowing operation."""
        # Values: -200, 0, 200, 400
        pixel_array = np.array([[-200, 0], [200, 400]], dtype=np.float64)

        # Window: center=100, width=400 -> range [-100, 300]
        result = apply_windowing(pixel_array, window_center=100, window_width=400)

        # -200 -> clipped to -100 -> scaled to 0
        # 0 -> scaled to 0.25
        # 200 -> scaled to 0.75
        # 400 -> clipped to 300 -> scaled to 1.0
        assert result.shape == (2, 2)
        assert result[0, 0] == pytest.approx(0.0, abs=0.01)
        assert result[0, 1] == pytest.approx(0.25, abs=0.01)
        assert result[1, 0] == pytest.approx(0.75, abs=0.01)
        assert result[1, 1] == pytest.approx(1.0, abs=0.01)

    def test_windowing_all_same(self):
        """Test windowing when all values same."""
        pixel_array = np.array([[100, 100], [100, 100]], dtype=np.float64)
        result = apply_windowing(pixel_array, window_center=100, window_width=0)
        assert np.all(result == 0)


class TestApplyWindowingWithConfig:
    """Tests for apply_windowing_with_config function."""

    def test_fixed_window(self):
        """Test with fixed window config."""
        pixel_array = np.array([[0, 100], [200, 300]], dtype=np.float64)
        mock_ds = MagicMock()
        config = WindowingConfig(
            strategy=WindowingStrategy.FIXED_WINDOW,
            window_center=150,
            window_width=200,
        )

        result, wc, ww = apply_windowing_with_config(pixel_array, mock_ds, config)
        assert wc == 150
        assert ww == 200

    def test_dicom_window(self):
        """Test with DICOM window config."""
        pixel_array = np.array([[0, 100], [200, 300]], dtype=np.float64)
        mock_ds = MagicMock()
        mock_ds.WindowCenter = 150.0
        mock_ds.WindowWidth = 200.0
        config = WindowingConfig(strategy=WindowingStrategy.USE_DICOM_WINDOW)

        result, wc, ww = apply_windowing_with_config(pixel_array, mock_ds, config)
        assert wc == 150.0
        assert ww == 200.0


class TestNormalization:
    """Tests for normalization functions."""

    def test_min_max(self):
        """Test min-max normalization."""
        pixel_array = np.array([[0, 50], [100, 200]], dtype=np.float64)
        result = normalize_min_max(pixel_array)

        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_min_max_constant(self):
        """Test min-max with constant values."""
        pixel_array = np.array([[100, 100], [100, 100]], dtype=np.float64)
        result = normalize_min_max(pixel_array)
        assert np.all(result == 0)

    def test_z_score(self):
        """Test z-score normalization."""
        pixel_array = np.array([[0, 50], [100, 150]], dtype=np.float64)
        result = normalize_z_score(pixel_array)

        # Result should be in [0, 1] after clipping and scaling
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_normalize_intensity_dispatch(self):
        """Test normalize_intensity dispatches correctly."""
        pixel_array = np.array([[0, 100], [200, 300]], dtype=np.float64)

        result_minmax = normalize_intensity(pixel_array, NormalizationMethod.MIN_MAX)
        result_zscore = normalize_intensity(pixel_array, NormalizationMethod.Z_SCORE)

        # Both should produce valid results
        assert result_minmax.shape == pixel_array.shape
        assert result_zscore.shape == pixel_array.shape


class TestCalculateResizeDimensions:
    """Tests for calculate_resize_dimensions function."""

    def test_no_aspect_ratio(self):
        """Test resize without aspect ratio preservation."""
        new_size, padding = calculate_resize_dimensions(
            original_size=(100, 200),
            target_size=(256, 256),
            keep_aspect_ratio=False,
        )
        assert new_size == (256, 256)
        assert padding == (0, 0, 0, 0)

    def test_with_aspect_ratio_wide(self):
        """Test resize with aspect ratio for wide image."""
        new_size, padding = calculate_resize_dimensions(
            original_size=(100, 200),
            target_size=(256, 256),
            keep_aspect_ratio=True,
        )
        # Wide image scaled by width: 200 -> 256, so 100 -> 128
        assert new_size == (128, 256)
        # Vertical padding needed: 256 - 128 = 128
        assert padding[0] + padding[1] == 128  # top + bottom

    def test_with_aspect_ratio_tall(self):
        """Test resize with aspect ratio for tall image."""
        new_size, padding = calculate_resize_dimensions(
            original_size=(200, 100),
            target_size=(256, 256),
            keep_aspect_ratio=True,
        )
        # Tall image scaled by height: 200 -> 256, so 100 -> 128
        assert new_size == (256, 128)
        # Horizontal padding needed: 256 - 128 = 128
        assert padding[2] + padding[3] == 128  # left + right


class TestResizeImage:
    """Tests for resize_image function."""

    def test_resize_disabled(self):
        """Test resize when disabled."""
        pixel_array = np.random.rand(100, 100)
        config = ResizingConfig(enabled=False)

        result = resize_image(pixel_array, config)
        np.testing.assert_array_equal(result, pixel_array)

    def test_resize_enabled(self):
        """Test resize when enabled."""
        pixel_array = np.random.rand(100, 100)
        config = ResizingConfig(enabled=True, target_size=(256, 256))

        result = resize_image(pixel_array, config)
        assert result.shape == (256, 256)

    def test_resize_with_padding(self):
        """Test resize with aspect ratio and padding."""
        pixel_array = np.random.rand(100, 200)  # Wide image
        config = ResizingConfig(
            enabled=True,
            target_size=(256, 256),
            keep_aspect_ratio=True,
            padding_value=0,
        )

        result = resize_image(pixel_array, config)
        assert result.shape == (256, 256)


class TestConvertToOutputFormat:
    """Tests for convert_to_output_format function."""

    def test_uint8_output(self):
        """Test conversion to uint8."""
        pixel_array = np.array([[0.0, 0.5], [0.75, 1.0]])
        result = convert_to_output_format(pixel_array, "uint8")

        assert result.dtype == np.uint8
        assert result[0, 0] == 0
        assert result[1, 1] == 255

    def test_float32_output(self):
        """Test conversion to float32."""
        pixel_array = np.array([[0.0, 0.5], [0.75, 1.0]])
        result = convert_to_output_format(pixel_array, "float32")

        assert result.dtype == np.float32

    def test_invalid_dtype(self):
        """Test invalid dtype raises error."""
        pixel_array = np.array([[0.0, 0.5], [0.75, 1.0]])
        with pytest.raises(ValueError, match="Unsupported output dtype"):
            convert_to_output_format(pixel_array, "int16")


class TestPreprocessingEngine:
    """Tests for PreprocessingEngine class."""

    def test_init_default(self):
        """Test default initialization."""
        engine = PreprocessingEngine()
        assert engine.config.normalization == NormalizationMethod.MIN_MAX

    def test_init_custom_config(self):
        """Test custom configuration."""
        config = PreprocessingConfig(
            normalization=NormalizationMethod.Z_SCORE,
            resizing=ResizingConfig(target_size=(512, 512)),
        )
        engine = PreprocessingEngine(config=config)
        assert engine.config.normalization == NormalizationMethod.Z_SCORE
        assert engine.config.resizing.target_size == (512, 512)

    @patch("cad_preprocess.preprocessing_engine.load_dicom")
    def test_process_success(self, mock_load):
        """Test successful processing."""
        # Create mock dataset
        mock_ds = MagicMock()
        mock_ds.pixel_array = np.random.randint(0, 4096, (512, 512), dtype=np.uint16)
        mock_ds.RescaleSlope = 1.0
        mock_ds.RescaleIntercept = 0.0
        mock_ds.WindowCenter = 2048.0
        mock_ds.WindowWidth = 4096.0
        mock_load.return_value = mock_ds

        engine = PreprocessingEngine()
        result = engine.process("/fake/path.dcm")

        assert result.success is True
        assert result.image.dtype == np.uint8
        assert result.final_shape == (1024, 1024)

    @patch("cad_preprocess.preprocessing_engine.load_dicom")
    def test_process_failure(self, mock_load):
        """Test failed processing."""
        mock_load.side_effect = FileNotFoundError("File not found")

        engine = PreprocessingEngine()
        result = engine.process("/fake/path.dcm")

        assert result.success is False
        assert "FileNotFoundError" in result.error_message

    def test_process_from_dataset(self):
        """Test processing from dataset."""
        mock_ds = MagicMock()
        mock_ds.pixel_array = np.random.randint(0, 4096, (512, 512), dtype=np.uint16)
        mock_ds.RescaleSlope = 1.0
        mock_ds.RescaleIntercept = 0.0
        mock_ds.WindowCenter = 2048.0
        mock_ds.WindowWidth = 4096.0

        engine = PreprocessingEngine()
        result = engine.process_from_dataset(mock_ds)

        assert result.success is True
        assert result.image.shape == (1024, 1024)

    def test_repr(self):
        """Test string representation."""
        engine = PreprocessingEngine()
        repr_str = repr(engine)
        assert "PreprocessingEngine" in repr_str
        assert "use_dicom_window" in repr_str

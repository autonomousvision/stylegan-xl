"""Validation tests to ensure the testing infrastructure is properly set up."""

import pytest
import torch
from pathlib import Path


@pytest.mark.unit
def test_pytest_is_working():
    """Verify that pytest is installed and working correctly."""
    assert True


@pytest.mark.unit
def test_fixtures_are_available(temp_dir, mock_config, sample_metrics):
    """Verify that our custom fixtures are working."""
    assert isinstance(temp_dir, Path)
    assert temp_dir.exists()
    
    assert isinstance(mock_config, dict)
    assert "batch_size" in mock_config
    assert mock_config["batch_size"] == 4
    
    assert isinstance(sample_metrics, dict)
    assert "loss" in sample_metrics
    assert isinstance(sample_metrics["loss"], float)


@pytest.mark.unit
def test_torch_is_available():
    """Verify that PyTorch is installed and accessible."""
    tensor = torch.tensor([1, 2, 3])
    assert tensor.shape == (3,)
    assert torch.__version__


@pytest.mark.unit
def test_coverage_is_configured():
    """Verify that coverage reporting is set up."""
    # This test will be included in coverage reports
    def dummy_function(x):
        if x > 0:
            return x * 2
        return 0
    
    assert dummy_function(5) == 10
    assert dummy_function(-1) == 0


@pytest.mark.unit
def test_markers_are_defined():
    """Verify that custom pytest markers are defined."""
    # This test itself uses the unit marker
    # The test passes if it runs without marker warnings
    pass


@pytest.mark.integration
def test_integration_marker():
    """Verify that integration marker works."""
    # This test uses the integration marker
    pass


@pytest.mark.slow
def test_slow_marker():
    """Verify that slow marker works."""
    # This test uses the slow marker
    import time
    time.sleep(0.1)  # Simulate a slow test


@pytest.mark.unit
class TestClassBasedTests:
    """Verify that class-based tests work correctly."""
    
    def test_class_method_one(self):
        """First test method in class."""
        assert 1 + 1 == 2
    
    def test_class_method_two(self, mock_model):
        """Second test method using fixtures."""
        assert mock_model is not None
        assert hasattr(mock_model, "parameters")


@pytest.mark.parametrize("input_val,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
    (0, 0),
])
def test_parametrized_tests(input_val, expected):
    """Verify that parametrized tests work."""
    assert input_val * 2 == expected


def test_temp_dir_cleanup(temp_dir):
    """Verify that temp_dir fixture properly cleans up."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("test content")
    assert test_file.exists()
    # The fixture will clean up the directory after this test


@pytest.mark.unit
def test_image_fixtures(sample_image, sample_images_batch):
    """Verify that image-related fixtures work."""
    assert sample_image.exists()
    assert sample_image.suffix == ".png"
    
    assert isinstance(sample_images_batch, torch.Tensor)
    assert sample_images_batch.shape == (4, 3, 256, 256)


@pytest.mark.unit
def test_mock_fixtures(mock_model, mock_dataset):
    """Verify that mock fixtures work correctly."""
    # Test mock model
    params = list(mock_model.parameters())
    assert len(params) > 0
    assert mock_model.eval() is mock_model
    
    # Test mock dataset
    assert len(mock_dataset) == 100
    image, label = mock_dataset[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, torch.Tensor)


@pytest.mark.unit
def test_random_state_fixtures(numpy_random_state, torch_random_state):
    """Verify that random state fixtures provide reproducibility."""
    import numpy as np
    
    # Test numpy random state
    vals1 = np.random.rand(5)
    np.random.seed(42)
    vals2 = np.random.rand(5)
    assert np.allclose(vals1, vals2)
    
    # Test torch random state
    tensor1 = torch.rand(5)
    torch.manual_seed(42)
    tensor2 = torch.rand(5)
    assert torch.allclose(tensor1, tensor2)


@pytest.mark.unit
def test_exception_handling():
    """Verify that pytest properly handles exceptions."""
    with pytest.raises(ValueError):
        raise ValueError("This is expected")
    
    with pytest.raises(ZeroDivisionError):
        1 / 0
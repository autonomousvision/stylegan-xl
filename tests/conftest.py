"""Shared pytest fixtures and configuration for all tests."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import MagicMock

import pytest
import torch
import numpy as np
from PIL import Image


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory that is cleaned up after the test."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Return a mock configuration dictionary for testing."""
    return {
        "batch_size": 4,
        "num_epochs": 10,
        "learning_rate": 0.001,
        "device": "cpu",
        "seed": 42,
        "image_size": 256,
        "num_channels": 3,
        "latent_dim": 512,
        "num_classes": 10,
    }


@pytest.fixture
def sample_image(temp_dir: Path) -> Path:
    """Create a sample image file for testing."""
    img_path = temp_dir / "test_image.png"
    img = Image.new("RGB", (256, 256), color="red")
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_images_batch() -> torch.Tensor:
    """Create a batch of sample images as torch tensors."""
    batch_size = 4
    channels = 3
    height = width = 256
    return torch.randn(batch_size, channels, height, width)


@pytest.fixture
def sample_latent_codes() -> torch.Tensor:
    """Create sample latent codes for testing generators."""
    batch_size = 4
    latent_dim = 512
    return torch.randn(batch_size, latent_dim)


@pytest.fixture
def sample_labels() -> torch.Tensor:
    """Create sample class labels for testing."""
    batch_size = 4
    num_classes = 10
    return torch.randint(0, num_classes, (batch_size,))


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock neural network model."""
    model = MagicMock()
    model.parameters.return_value = [torch.randn(10, 10)]
    model.eval.return_value = model
    model.train.return_value = model
    model.to.return_value = model
    return model


@pytest.fixture
def mock_dataset() -> MagicMock:
    """Create a mock dataset."""
    dataset = MagicMock()
    dataset.__len__.return_value = 100
    dataset.__getitem__.return_value = (
        torch.randn(3, 256, 256),  # image
        torch.tensor(0)  # label
    )
    return dataset


@pytest.fixture
def numpy_random_state():
    """Set and restore numpy random state for reproducible tests."""
    original_state = np.random.get_state()
    np.random.seed(42)
    yield
    np.random.set_state(original_state)


@pytest.fixture
def torch_random_state():
    """Set and restore torch random state for reproducible tests."""
    original_state = torch.get_rng_state()
    torch.manual_seed(42)
    yield
    torch.set_rng_state(original_state)


@pytest.fixture
def mock_checkpoint(temp_dir: Path) -> Dict[str, Any]:
    """Create a mock model checkpoint."""
    checkpoint = {
        "epoch": 5,
        "model_state_dict": {"layer1.weight": torch.randn(10, 10)},
        "optimizer_state_dict": {"param_groups": []},
        "loss": 0.5,
        "config": {"learning_rate": 0.001},
    }
    checkpoint_path = temp_dir / "checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)
    return {"path": checkpoint_path, "data": checkpoint}


@pytest.fixture
def sample_metrics() -> Dict[str, float]:
    """Create sample training metrics."""
    return {
        "loss": 0.523,
        "accuracy": 0.912,
        "fid_score": 12.34,
        "inception_score": 8.76,
        "precision": 0.89,
        "recall": 0.87,
    }


@pytest.fixture(autouse=True)
def cleanup_cuda_cache():
    """Automatically cleanup CUDA cache after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def mock_gui_context() -> MagicMock:
    """Create a mock GUI context for testing visualization components."""
    context = MagicMock()
    context.width = 800
    context.height = 600
    context.frame_count = 0
    return context


@pytest.fixture
def sample_style_vectors() -> Dict[str, torch.Tensor]:
    """Create sample style vectors for StyleGAN testing."""
    return {
        "w": torch.randn(1, 18, 512),  # W space
        "w_plus": torch.randn(1, 18, 512),  # W+ space
        "s": torch.randn(1, 26, 512),  # S space
    }


def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
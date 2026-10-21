"""Synthetic spectra with exactly known properties."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from bobspec.constants import C_KMS
from bobspec.io import Spec

LineModel = Callable[..., np.ndarray]


@pytest.fixture(scope="session")
def line_model() -> LineModel:
    """Normalised spectrum with 300 Gaussian lines; ``model(wave, rv, sigma_aa)``."""
    rng = np.random.default_rng(12345)
    centres = rng.uniform(5005.0, 5295.0, 300)
    depths = rng.uniform(0.1, 0.7, 300)

    def model(wave: np.ndarray, rv: float = 0.0, sigma: float = 0.06) -> np.ndarray:
        shifted = centres * (1.0 + rv / C_KMS)
        profile = depths[:, None] * np.exp(-0.5 * ((wave[None, :] - shifted[:, None]) / sigma) ** 2)
        return 1.0 - profile.sum(axis=0)

    return model


@pytest.fixture(scope="session")
def template(line_model: LineModel) -> Spec:
    """Rest-frame template sampled at 0.01 A."""
    wave = np.arange(5000.0, 5300.0, 0.01)
    return Spec(wave, line_model(wave), name="template")


@pytest.fixture
def obs_wave() -> np.ndarray:
    """Observed pixel grid (0.03 A)."""
    return np.arange(5010.0, 5290.0, 0.03)


def gaussian_line(wave: np.ndarray, centre: float, sigma: float, depth: float) -> np.ndarray:
    """Single normalised absorption line."""
    return 1.0 - depth * np.exp(-0.5 * ((wave - centre) / sigma) ** 2)

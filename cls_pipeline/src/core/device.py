"""
device.py — Device detection and management for PyTorch.

Handles automatic detection and selection of MPS (Apple Silicon), CUDA, or CPU
devices for optimal performance on different hardware.
"""

import logging
import os
from dataclasses import dataclass
from typing import Literal

import torch

logger = logging.getLogger(__name__)

DeviceType = Literal["mps", "cuda", "cpu"]


@dataclass
class DeviceInfo:
    """Information about the selected device."""
    device: torch.device
    device_type: DeviceType
    device_name: str
    supports_float16: bool


class DeviceManager:
    """
    Manages PyTorch device selection and memory optimization.

    Supports:
    - MPS (Metal Performance Shaders) for Apple Silicon
    - CUDA for NVIDIA GPUs
    - CPU fallback
    """

    def __init__(self, preferred: str = "auto"):
        """
        Initialize DeviceManager.

        Parameters
        ----------
        preferred : str
            Preferred device: "auto", "mps", "cuda", or "cpu".
        """
        self.preferred = preferred
        self._device_info: DeviceInfo | None = None

    def detect(self) -> DeviceInfo:
        """
        Detect and select the best available device.

        Returns
        -------
        DeviceInfo
            Information about the selected device.
        """
        if self._device_info is not None:
            return self._device_info

        # Check environment override
        env_device = os.getenv("DEVICE")
        if env_device:
            self.preferred = env_device.lower()

        device_type: DeviceType
        device_name: str

        if self.preferred == "auto":
            # Auto-detect best device
            if torch.backends.mps.is_available():
                device_type = "mps"
                device_name = "Apple Silicon (MPS)"
            elif torch.cuda.is_available():
                device_type = "cuda"
                device_name = torch.cuda.get_device_name(0)
            else:
                device_type = "cpu"
                device_name = "CPU"
        elif self.preferred == "mps":
            if not torch.backends.mps.is_available():
                logger.warning("MPS requested but not available, falling back to CPU")
                device_type = "cpu"
                device_name = "CPU"
            else:
                device_type = "mps"
                device_name = "Apple Silicon (MPS)"
        elif self.preferred == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                device_type = "cpu"
                device_name = "CPU"
            else:
                device_type = "cuda"
                device_name = torch.cuda.get_device_name(0)
        else:
            device_type = "cpu"
            device_name = "CPU"

        device = torch.device(device_type)
        supports_float16 = device_type in ("cuda", "mps")

        self._device_info = DeviceInfo(
            device=device,
            device_type=device_type,
            device_name=device_name,
            supports_float16=supports_float16,
        )

        logger.info("Selected device: %s (%s)", device_name, device_type)
        return self._device_info

    @property
    def device(self) -> torch.device:
        """Get the PyTorch device."""
        return self.detect().device

    @property
    def device_type(self) -> DeviceType:
        """Get the device type string."""
        return self.detect().device_type

    def clear_cache(self) -> None:
        """Clear device memory cache."""
        info = self.detect()
        if info.device_type == "mps":
            torch.mps.empty_cache()
            logger.debug("MPS cache cleared")
        elif info.device_type == "cuda":
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")

    def synchronize(self) -> None:
        """Synchronize device operations."""
        info = self.detect()
        if info.device_type == "mps":
            torch.mps.synchronize()
        elif info.device_type == "cuda":
            torch.cuda.synchronize()


# Module-level convenience functions
_device_manager: DeviceManager | None = None


def get_device(preferred: str = "auto") -> torch.device:
    """
    Get the best available PyTorch device.

    Parameters
    ----------
    preferred : str
        Preferred device: "auto", "mps", "cuda", or "cpu".

    Returns
    -------
    torch.device
        The selected device.
    """
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager(preferred)
    return _device_manager.device


def get_device_manager(preferred: str = "auto") -> DeviceManager:
    """Get or create the global DeviceManager instance."""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager(preferred)
    return _device_manager


def clear_device_cache() -> None:
    """Clear the device memory cache."""
    if _device_manager is not None:
        _device_manager.clear_cache()

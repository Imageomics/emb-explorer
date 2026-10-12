"""
Progress management utilities for Streamlit UI.
"""

from abc import ABC, abstractmethod
from typing import Optional
import streamlit as st


class ProgressContext(ABC):
    """Base class for different progress UI patterns.

    Entering the context returns an object that is callable as
    ``progress(fraction, text)`` and also offers ``progress.fail(message)``
    for a step that finishes without raising but did not succeed (for
    example "no images found"), so the context reports an error instead of
    its success message.
    """

    @abstractmethod
    def __enter__(self) -> "ProgressContext":
        pass

    @abstractmethod
    def __call__(self, progress: float, text: str) -> None:
        pass

    @abstractmethod
    def fail(self, message: str) -> None:
        pass
    
    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class StreamlitProgressContext(ProgressContext):
    """Standard Streamlit progress bar with automatic cleanup"""
    
    def __init__(self, placeholder, success_message: Optional[str] = None):
        self.placeholder = placeholder
        self.success_message = success_message
        self.progress_bar = None
        self.failure_message: Optional[str] = None

    def __enter__(self):
        self.progress_bar = self.placeholder.progress(0, text="Starting...")
        return self

    def __call__(self, progress: float, text: str) -> None:
        self.update_progress(progress, text)

    def update_progress(self, progress: float, text: str):
        if self.progress_bar:
            self.progress_bar.progress(progress, text=text)

    def fail(self, message: str) -> None:
        """Mark the step as failed: on exit the placeholder shows this error
        instead of the success message."""
        self.failure_message = message

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress_bar:
            self.progress_bar.empty()

        if exc_type is not None:
            self.placeholder.error(f"Error: {exc_val}")
        elif self.failure_message:
            self.placeholder.error(self.failure_message)
        elif self.success_message:
            self.placeholder.success(self.success_message)


class MockProgressContext(ProgressContext):
    """Mock progress context for testing - captures progress updates without UI"""
    
    def __init__(self):
        self.updates = []
        self.failure_message: Optional[str] = None

    def __enter__(self):
        return self

    def __call__(self, progress: float, text: str) -> None:
        self.capture_progress(progress, text)

    def capture_progress(self, progress: float, text: str):
        self.updates.append((progress, text))

    def fail(self, message: str) -> None:
        self.failure_message = message

    def __exit__(self, *args):
        pass

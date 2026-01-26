"""
Progress management utilities for Streamlit UI.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable
import streamlit as st


class ProgressContext(ABC):
    """Base class for different progress UI patterns"""
    
    @abstractmethod
    def __enter__(self) -> Callable[[float, str], None]:
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
    
    def __enter__(self):
        self.progress_bar = self.placeholder.progress(0, text="Starting...")
        return self.update_progress
    
    def update_progress(self, progress: float, text: str):
        if self.progress_bar:
            self.progress_bar.progress(progress, text=text)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress_bar:
            self.progress_bar.empty()
        
        if exc_type is None and self.success_message:
            self.placeholder.success(self.success_message)
        elif exc_type is not None:
            self.placeholder.error(f"Error: {exc_val}")


class MockProgressContext(ProgressContext):
    """Mock progress context for testing - captures progress updates without UI"""
    
    def __init__(self):
        self.updates = []
    
    def __enter__(self):
        return self.capture_progress
    
    def capture_progress(self, progress: float, text: str):
        self.updates.append((progress, text))
    
    def __exit__(self, *args):
        pass

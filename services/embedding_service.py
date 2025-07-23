"""
Embedding generation service.
"""

import torch
import numpy as np
import open_clip
import streamlit as st
from typing import Tuple, List, Optional, Callable

from utils.io import list_image_files
from utils.models import list_available_models
from hpc_inference.datasets.image_folder_dataset import ImageFolderDataset


class EmbeddingService:
    """Service for handling embedding generation workflows"""
    
    @staticmethod
    @st.cache_data
    def get_model_options() -> List[str]:
        """Get formatted model options for selectbox."""
        models_data = list_available_models()
        options = []
        
        # Add all models from list
        for model in models_data:
            name = model['name']
            pretrained = model['pretrained']
                
            if pretrained is None or pretrained == "":
                display_name = name
            else:
                display_name = f"{name} ({pretrained})"
            options.append(display_name)
        
        return options
    
    @staticmethod
    def parse_model_selection(selected_model: str) -> Tuple[str, Optional[str]]:
        """Parse the selected model string to extract model name and pretrained."""
        # Parse OpenCLIP format: "model_name (pretrained)" or just "model_name"
        if "(" in selected_model and selected_model.endswith(")"):
            name = selected_model.split(" (")[0]
            pretrained = selected_model.split(" (")[1].rstrip(")")
            return name, pretrained
        else:
            return selected_model, None
    
    @staticmethod
    @st.cache_resource(show_spinner=True)
    def load_model_unified(selected_model: str, device: str = "cuda"):
        """Unified model loading function that handles all model types."""
        model_name, pretrained = EmbeddingService.parse_model_selection(selected_model)
        
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        
        model = torch.compile(model.to(device))
        return model, preprocess
    
    @staticmethod
    @torch.no_grad()
    def generate_embeddings(
        image_dir: str,
        model_name: str,
        batch_size: int,
        n_workers: int,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Generate embeddings for images in a directory.
        
        Args:
            image_dir: Path to directory containing images
            model_name: Name of the model to use
            batch_size: Batch size for processing
            n_workers: Number of worker processes
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (embeddings array, list of valid image paths)
        """
        if progress_callback:
            progress_callback(0.0, "Listing images...")
        
        image_paths = list_image_files(image_dir)
        
        if progress_callback:
            progress_callback(0.1, f"Found {len(image_paths)} images. Loading model...")
        
        torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = EmbeddingService.load_model_unified(model_name, torch_device)
        
        if progress_callback:
            progress_callback(0.2, "Creating dataset...")
        
        # Create dataset & DataLoader
        dataset = ImageFolderDataset(
            image_dir=image_dir,
            preprocess=preprocess,
            uuid_mode="fullpath",
            rank=0, 
            world_size=1,
            evenly_distribute=True,
            validate=True
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
            pin_memory=True
        )

        total = len(image_paths)
        valid_paths = []
        embeddings = []
        
        processed = 0
        for batch_paths, batch_imgs in dataloader:
            batch_imgs = batch_imgs.to(torch_device, non_blocking=True)
            batch_embeds = model.encode_image(batch_imgs).cpu().numpy()
            embeddings.append(batch_embeds)
            valid_paths.extend(batch_paths)
            processed += len(batch_paths)
            
            if progress_callback:
                progress = 0.2 + (processed / total) * 0.8  # Use 20% to 100% for actual processing
                progress_callback(progress, f"Embedding {processed}/{total}")

        # Stack embeddings if available
        if embeddings:
            embeddings = np.vstack(embeddings)
        else:
            embeddings = np.empty((0, model.visual.output_dim))

        if progress_callback:
            progress_callback(1.0, f"Complete! Generated {embeddings.shape[0]} embeddings")

        return embeddings, valid_paths

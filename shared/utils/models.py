import pandas as pd
import open_clip

def list_available_models():
    """List all available models."""
    
    # Create list of all models
    models_data = []
    
    # Add special models first
    models_data.extend([
        {"name": "hf-hub:imageomics/bioclip-2", "pretrained": None},
        {"name": "hf-hub:imageomics/bioclip", "pretrained": None}
    ])
    
    # OpenCLIP models
    openclip_models = open_clip.list_pretrained()
    for model_name, pretrained in openclip_models:
        models_data.append({
            "name": model_name,
            "pretrained": pretrained
        })
    
    return models_data

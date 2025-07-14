import pandas as pd
import open_clip

def create_models_csv():
    """Create a CSV file with all available models."""
    
    # Get OpenCLIP models
    openclip_models = open_clip.list_pretrained()
    
    # Create list of all models
    models_data = []
    
    # Add special models first
    models_data.extend([
        {"name": "hf-hub:imageomics/bioclip", "pretrained": None}
    ])
    
    # Add OpenCLIP models
    for model_name, pretrained in openclip_models:
        models_data.append({
            "name": model_name,
            "pretrained": pretrained
        })
    
    # Create DataFrame
    df = pd.DataFrame(models_data)
    
    # Save as CSV
    df.to_csv("data/available_models.csv", index=False)

    print(f"Saved {len(df)} models to available_models.csv")
    print("\nFirst 10 rows:")
    print(df.head(10))
    
    return df

if __name__ == "__main__":
    df = create_models_csv()
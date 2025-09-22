#!/usr/bin/env python3
"""
Command-line script to list available models from the emb-explorer utils.
"""

import json
import argparse
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.models import list_available_models


def main():
    """Main function to list available models."""
    parser = argparse.ArgumentParser(
        description="List all available models for the embedding explorer"
    )
    parser.add_argument(
        "--format", 
        choices=["json", "table", "names"], 
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--pretty", 
        action="store_true",
        help="Pretty print JSON output"
    )
    
    args = parser.parse_args()
    
    try:
        models = list_available_models()
        
        if args.format == "json":
            if args.pretty:
                print(json.dumps(models, indent=2))
            else:
                print(json.dumps(models))
        
        elif args.format == "table":
            print(f"{'Model Name':<40} {'Pretrained':<30}")
            print("-" * 70)
            for model in models:
                name = model['name']
                pretrained = model['pretrained'] or "None"
                print(f"{name:<40} {pretrained:<30}")
        
        elif args.format == "names":
            for model in models:
                name = model['name']
                pretrained = model['pretrained']
                if pretrained:
                    print(f"{name} ({pretrained})")
                else:
                    print(name)
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Utility functions for building and displaying taxonomic trees.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter


def build_taxonomic_tree(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build a hierarchical taxonomic tree from a dataframe.
    
    Args:
        df: DataFrame containing taxonomic columns
        
    Returns:
        Nested dictionary representing the taxonomic tree with counts
    """
    taxonomic_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    
    # Filter to only include rows that have at least kingdom
    df_clean = df[df['kingdom'].notna()].copy()
    
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))))))
    
    for _, row in df_clean.iterrows():
        # Get values for each taxonomic level, using 'Unknown' for nulls
        kingdom = row.get('kingdom', 'Unknown') or 'Unknown'
        phylum = row.get('phylum', 'Unknown') or 'Unknown'
        class_name = row.get('class', 'Unknown') or 'Unknown'
        order = row.get('order', 'Unknown') or 'Unknown'
        family = row.get('family', 'Unknown') or 'Unknown'
        genus = row.get('genus', 'Unknown') or 'Unknown'
        species = row.get('species', 'Unknown') or 'Unknown'
        
        # Build the nested structure
        tree[kingdom][phylum][class_name][order][family][genus][species] += 1
    
    return dict(tree)


def format_tree_string(tree: Dict[str, Any], max_depth: int = 7, min_count: int = 1) -> str:
    """
    Format the taxonomic tree as a string similar to the 'tree' command output.
    
    Args:
        tree: Taxonomic tree dictionary
        max_depth: Maximum depth to display
        min_count: Minimum count to include in the tree
        
    Returns:
        Formatted tree string
    """
    lines = []
    
    def format_level(node, level=0, prefix="", is_last=True, path=""):
        if level >= max_depth:
            return
            
        if isinstance(node, dict):
            items = list(node.items())
            # Sort by count (descending) if we're at the species level
            if level == 6:  # species level
                items = sorted(items, key=lambda x: x[1] if isinstance(x[1], int) else 0, reverse=True)
            else:
                # Sort by name for higher levels
                items = sorted(items, key=lambda x: x[0])
            
            # Filter by minimum count
            items = [(k, v) for k, v in items if (
                isinstance(v, int) and v >= min_count) or (
                isinstance(v, dict) and any(
                    get_total_count(subv) >= min_count for subv in v.values()
                )
            )]
            
            for i, (key, value) in enumerate(items):
                is_last_item = (i == len(items) - 1)
                
                # Create the tree characters
                if level == 0:
                    connector = ""
                    new_prefix = ""
                else:
                    connector = "└── " if is_last_item else "├── "
                    new_prefix = prefix + ("    " if is_last_item else "│   ")
                
                # Get count for this node
                if isinstance(value, int):
                    count = value
                    count_str = f" ({count})"
                else:
                    count = get_total_count(value)
                    count_str = f" ({count})" if count > 0 else ""
                
                # Add the line
                lines.append(f"{prefix}{connector}{key}{count_str}")
                
                # Recurse if it's a dictionary
                if isinstance(value, dict):
                    format_level(value, level + 1, new_prefix, is_last_item, f"{path}/{key}")
    
    format_level(tree)
    return "\n".join(lines)


def get_total_count(node: Any) -> int:
    """
    Get the total count for a tree node.
    
    Args:
        node: Tree node (dict or int)
        
    Returns:
        Total count for this node and all children
    """
    if isinstance(node, int):
        return node
    elif isinstance(node, dict):
        return sum(get_total_count(child) for child in node.values())
    else:
        return 0


def get_tree_statistics(tree: Dict[str, Any]) -> Dict[str, int]:
    """
    Get statistics about the taxonomic tree.
    
    Args:
        tree: Taxonomic tree dictionary
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_records': get_total_count(tree),
        'kingdoms': len(tree),
        'phyla': 0,
        'classes': 0,
        'orders': 0,
        'families': 0,
        'genera': 0,
        'species': 0
    }
    
    for kingdom, phyla in tree.items():
        stats['phyla'] += len(phyla)
        for phylum, classes in phyla.items():
            stats['classes'] += len(classes)
            for class_name, orders in classes.items():
                stats['orders'] += len(orders)
                for order, families in orders.items():
                    stats['families'] += len(families)
                    for family, genera in families.items():
                        stats['genera'] += len(genera)
                        for genus, species in genera.items():
                            stats['species'] += len(species)
    
    return stats

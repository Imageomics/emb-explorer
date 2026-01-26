import os
import shutil

def list_image_files(image_dir, allowed_extensions=('jpg', 'jpeg', 'png')):
    """
    List image file paths in a directory with allowed extensions.

    Args:
        image_dir (str): Path to the directory containing images.
        allowed_extensions (tuple, optional): Allowed file extensions. Defaults to ('jpg', 'jpeg', 'png').

    Returns:
        list: List of full file paths for images with allowed extensions.
    """
    return [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(allowed_extensions)
    ]
    
def copy_image(row, repartition_dir):
    """
    Copy an image file to a cluster-specific subdirectory.

    Args:
        row (dict): A dictionary containing at least 'cluster' (cluster ID) and 'image_path' (source image path).
        repartition_dir (str): The root directory where cluster subfolders will be created.

    Returns:
        dict or None: A dictionary with keys 'abs_path', 'file_name', and 'cluster' if successful; None if an error occurs.
    """
    cluster_id = row['cluster']
    src_img_path = row['image_path']
    cluster_folder = os.path.join(repartition_dir, f"cluster_{cluster_id}")
    os.makedirs(cluster_folder, exist_ok=True)
    dst_img_path = os.path.join(cluster_folder, os.path.basename(src_img_path))
    try:
        shutil.copy2(src_img_path, dst_img_path)
        return {
            "abs_path": os.path.abspath(dst_img_path),
            "file_name": os.path.basename(src_img_path),
            "cluster": cluster_id
        }
    except Exception as e:
        return None  # Optionally log error

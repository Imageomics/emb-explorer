import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageFolderDataset(Dataset):
    """
    A PyTorch Dataset for loading images from a directory.

    This dataset scans a given folder for image files with common extensions,
    loads each image using PIL, applies an optional transform, and returns
    the image path and the processed image.

    Attributes:
        image_dir (str): Path to the directory containing images.
        transform (callable, optional): Optional transform to be applied on a sample.
        image_files (List[str]): Sorted list of image file paths in the directory.

    Args:
        image_dir (str): Path to the directory containing images.
        transform (callable, optional): Optional transform to be applied on a sample.

    Returns:
        Tuple[str, torch.Tensor]: The image file path and the transformed image tensor.
    """
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # List image filepaths
        self.image_files = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(self.IMG_EXTS)
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('RGB')  # Ensure 3-channel
        if self.transform:
            img = self.transform(img)
        return img_path, img

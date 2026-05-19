import os
import io
from PIL import Image
import numpy as np
import base64
import logging

def sanitize_for_json(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):  # np.float32, np.int64, etc.
        return x.item()
    if isinstance(x, dict):
        return {str(k): sanitize_for_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [sanitize_for_json(v) for v in x]
    if isinstance(x, tuple):
        return [sanitize_for_json(v) for v in x]
    return x

def upload_numpy_array_to_local(image_np, file_path, base_dir):
    """Converts a NumPy array to a PIL Image and saves locally."""
    if not isinstance(image_np, np.ndarray):
        logging.error(f"Error: Input is not a NumPy array. Type: {type(image_np)}")
        return None

    try:
        pil_image = Image.fromarray(image_np.astype(np.uint8))

        full_path = os.path.join(base_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        pil_image.save(full_path, format='PNG')
        logging.info(f"Saved to {full_path}")
        return full_path

    except Exception as e:
        logging.error(f"Failed to save {file_path}: {e}")
        return None

def upload_image_bytes_to_local(image_bytes, file_path, base_dir, content_type='image/png'):
    """
    Saves raw image bytes (e.g., PNG, JPEG) to local filesystem.

    Args:
        image_bytes (bytes): The raw bytes of the image (e.g., from a .png file or matplotlib.savefig).
        file_path (str): The desired file path relative to base_dir (e.g., 'visualizations/my_image.png').
        base_dir (str): The base directory for storage.
        content_type (str): The MIME type of the image (e.g., 'image/png', 'image/jpeg').
    Returns:
        str: The full path of the saved file, or None if save fails.
    """
    if not isinstance(image_bytes, bytes):
        logging.error(f"Error: Input is not bytes. Type: {type(image_bytes)}")
        return None

    try:
        full_path = os.path.join(base_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, 'wb') as f:
            f.write(image_bytes)

        logging.info(f"Saved to {full_path}")
        return full_path

    except Exception as e:
        logging.error(f"Failed to save {file_path}: {e}")
        return None

def upload_html_to_local(html_file, destination_file_path, base_dir):
    """Copies HTML file to local destination."""
    full_path = os.path.join(base_dir, destination_file_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    with open(html_file, 'r') as src:
        content = src.read()
    
    with open(full_path, 'w') as dst:
        dst.write(content)
    
    logging.info(f"Copied {html_file} to {full_path}")
    return full_path

def upload_html_content_to_local(html_content, destination_file_path, base_dir):
    """Saves HTML content directly to local filesystem."""
    full_path = os.path.join(base_dir, destination_file_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    with open(full_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logging.info(f"Saved HTML content to {full_path}")
    return full_path

def get_image_base64_from_local_path(local_path):
    """Reads image from local path and returns base64 encoded string."""
    try:
        if not os.path.exists(local_path):
            logging.error(f"Warning: File not found - {local_path}")
            return None
        
        with open(local_path, 'rb') as f:
            image_bytes = f.read()
        
        img_str = base64.b64encode(image_bytes).decode("utf-8")
        
        # Determine content type from extension
        ext = os.path.splitext(local_path)[1].lower()
        content_type_map = {
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        content_type = content_type_map.get(ext, 'image/png')
        
        return f"data:{content_type};base64,{img_str}"
    except Exception as e:
        logging.error(f"Error reading {local_path}: {e}")
        return None


import os
from pathlib import Path
import requests
import shutil
import re
from urllib.parse import urlparse

# Constants and configurations
HOME = Path.home()
SCR_PATH = Path(HOME / 'ANXETY')
SCRIPTS = SCR_PATH / 'scripts'
SETTINGS_PATH = SCR_PATH / 'settings.json'

# Directory mappings
PREFIX_MAP = {
    'model': (HOME / 'models', '$ckpt'),
    'vae': (HOME / 'vae', '$vae'),
    'lora': (HOME / 'lora', '$lora'),
    'embed': (HOME / 'embeddings', '$emb'),
    'extension': (HOME / 'extensions', '$ext'),
    'adetailer': (HOME / 'adetailer', '$ad'),
    'control': (HOME / 'controlnet', '$cnet'),
    'upscale': (HOME / 'upscale', '$ups'),
    'clip': (HOME / 'clip', '$clip'),
    'unet': (HOME / 'unet', '$unet'),
    'vision': (HOME / 'vision', '$vis'),
    'encoder': (HOME / 'encoder', '$enc'),
    'diffusion': (HOME / 'diffusion', '$diff'),
    'config': (HOME / 'config', '$cfg')
}

# Ensure directories exist
for dir_path, _ in PREFIX_MAP.values():
    os.makedirs(dir_path, exist_ok=True)

def _clean_url(url):
    """Clean the URL to ensure it's in the correct format."""
    url_cleaners = {
        'huggingface.co': lambda u: u.replace('/blob/', '/resolve/').split('?')[0],
        'github.com': lambda u: u.replace('/blob/', '/raw/')
    }
    for domain, cleaner in url_cleaners.items():
        if domain in url:
            return cleaner(url)
    return url

def _extract_filename(url):
    """Extract the filename from the URL."""
    if match := re.search(r'\[(.*?)\]', url):
        return match.group(1)
    if any(d in urlparse(url).netloc for d in ["civitai.com", "drive.google.com"]):
        return None
    return Path(urlparse(url).path).name

def _download_file(url, dst_dir, file_name=None):
    """Download a file from a URL to the specified directory."""
    if not file_name:
        file_name = _extract_filename(url) or os.path.basename(urlparse(url).path)

    dst_path = os.path.join(dst_dir, file_name)

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(dst_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Downloaded {file_name} to {dst_dir}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def _process_download_link(link):
    """Process a download link to extract prefix, URL, and filename."""
    link = _clean_url(link)
    if ':' in link:
        prefix, path = link.split(':', 1)
        if prefix in PREFIX_MAP:
            return prefix, re.sub(r'\[.*?\]', '', path), _extract_filename(path)
    return None, link, None

def download(line):
    """Download files from comma-separated links."""
    for link in filter(None, map(str.strip, line.split(','))):
        prefix, url, filename = _process_download_link(link)

        if prefix:
            dir_path, _ = PREFIX_MAP[prefix]
            _download_file(url, dir_path, filename)
        else:
            # Default directory for generic downloads
            dir_path = HOME / 'downloads'
            os.makedirs(dir_path, exist_ok=True)
            _download_file(url, dir_path, filename)

# Example usage
line = "model:https://example.com/model1.zip,https://example.com/model2.zip"
download(line)

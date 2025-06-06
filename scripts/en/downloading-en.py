# ~ download.py | by ANXETY ~

from webui_utils import handle_setup_timer    # WEBUI
from CivitaiAPI import CivitAiAPI              # CivitAI API
from Manager import m_download                 # Every Download
import json_utils as js                        # JSON

from IPython.display import clear_output
from IPython.utils import capture
from urllib.parse import urlparse
from IPython import get_ipython
from datetime import timedelta
from pathlib import Path
import subprocess
import requests
import zipfile
import shutil
import shlex
import time
import json
import sys
import re
import os


CD = os.chdir
ipySys = get_ipython().system
ipyRun = get_ipython().run_line_magic

# Constants
HOME = Path.home()
VENV = HOME / 'venv'
SCR_PATH = Path(HOME / 'ANXETY')
SCRIPTS = SCR_PATH / 'scripts'
SETTINGS_PATH = SCR_PATH / 'settings.json'

LANG = js.read(SETTINGS_PATH, 'ENVIRONMENT.lang')
ENV_NAME = js.read(SETTINGS_PATH, 'ENVIRONMENT.env_name')
UI = js.read(SETTINGS_PATH, 'WEBUI.current')
WEBUI = js.read(SETTINGS_PATH, 'WEBUI.webui_path')
civitai_token = js.read(SETTINGS_PATH, 'ENVIRONMENT.civitai_token', '') # Assume this is in settings.json
huggingface_token = js.read(SETTINGS_PATH, 'ENVIRONMENT.huggingface_token', '') # Assume this is in settings.json


# Text Colors (\033)
class COLORS:
    R  =  "\033[31m"      # Red
    G  =  "\033[32m"      # Green
    Y  =  "\033[33m"      # Yellow
    B  =  "\033[34m"      # Blue
    lB =  "\033[36;1m"    # lightBlue
    X  =  "\033[0m"       # Reset

COL = COLORS


## =================== LIBRARIES | VENV ==================

def install_dependencies(commands):
    """Run a list of installation commands."""
    for cmd in commands:
        try:
            subprocess.run(shlex.split(cmd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

def setup_venv(url):
    """Customize the virtual environment using the specified URL."""
    CD(HOME)
    fn = Path(url).name

    # Use the new download fallback mechanism
    download_from_url_with_fallbacks(url, HOME, fn, log=True)

    # Install dependencies based on environment
    install_commands = []
    if ENV_NAME == 'Kaggle':
        install_commands.extend([
            'pip install ipywidgets jupyterlab_widgets --upgrade',
            'rm -f /usr/lib/python3.10/sitecustomize.py'
        ])

    install_commands.append('sudo apt-get -y install lz4 pv')
    install_dependencies(install_commands)

    # Unpack and clean
    ipySys(f"pv {fn} | lz4 -d | tar xf -")
    Path(fn).unlink()

    BIN = str(VENV / 'bin')
    PKG = str(VENV / 'lib/python3.10/site-packages')

    os.environ['PYTHONWARNINGS'] = 'ignore'

    sys.path.insert(0, PKG)
    if BIN not in os.environ['PATH']:
        os.environ['PATH'] = BIN + ':' + os.environ['PATH']
    if PKG not in os.environ['PYTHONPATH']:
        os.environ['PYTHONPATH'] = PKG + ':' + os.environ['PYTHONPATH']

def install_packages(install_lib):
    """Install packages from the provided library dictionary."""
    for index, (package, install_cmd) in enumerate(install_lib.items(), start=1):
        print(f"\r[{index}/{len(install_lib)}] {COL.G}>>{COL.X} Installing {COL.Y}{package}{COL.X}..." + ' ' * 35, end='')
        try:
            result = subprocess.run(install_cmd, shell=True, capture_output=True)
            if result.returncode != 0:
                print(f"\n{COL.R}Error installing {package}{COL.X}")
        except Exception:
            pass

# Check and install dependencies
if not js.key_exists(SETTINGS_PATH, 'ENVIRONMENT.install_deps', True):
    install_lib = {
        ## Libs
        'aria2': "pip install aria2",
        'gdown': "pip install gdown",
        ## Tunnels
        'localtunnel': "npm install -g localtunnel",
        'cloudflared': "wget -qO /usr/bin/cl https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64; chmod +x /usr/bin/cl",
        'zrok': "wget -qO zrok_1.0.4_linux_amd64.tar.gz https://github.com/openziti/zrok/releases/download/v1.0.4/zrok_1.0.4_linux_amd64.tar.gz; tar -xzf zrok_1.0.4_linux_amd64.tar.gz -C /usr/bin; rm -f zrok_1.0.4_linux_amd64.tar.gz",
        'ngrok': "wget -qO ngrok-v3-stable-linux-amd64.tgz https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz; tar -xzf ngrok-v3-stable-linux-amd64.tgz -C /usr/bin; rm -f ngrok-v3-stable-linux-amd64.tgz"
    }

    print('💿 Installing the libraries will take a bit of time.')
    install_packages(install_lib)
    clear_output()
    js.update(SETTINGS_PATH, 'ENVIRONMENT.install_deps', True)

# Install VENV
current_ui = js.read(SETTINGS_PATH, 'WEBUI.current')
latest_ui = js.read(SETTINGS_PATH, 'WEBUI.latest')

# Determine whether to reinstall venv
venv_needs_reinstall = (
    not VENV.exists()  # venv is missing
    # Check category change (Classic <-> other)
    or (latest_ui == 'Classic') != (current_ui == 'Classic')
)

if venv_needs_reinstall:
    if VENV.exists():
        print("🗑️ Remove old venv...")
        shutil.rmtree(VENV)
        clear_output()

    if current_ui == 'Classic':
        venv_url = "https://huggingface.co/NagisaNao/ANXETY/resolve/main/python31112-venv-torch251-cu121-C-Classic.tar.lz4"
        py_version = '(3.11.12)'
    else:
        venv_url = "https://huggingface.co/NagisaNao/ANXETY/resolve/main/python31017-venv-torch251-cu121-C-fca.tar.lz4"
        py_version = '(3.10.17)'

    print(f"♻️ Installing VENV {py_version}, this will take some time...")
    setup_venv(venv_url)
    clear_output()

    # Update latest UI version...
    js.update(SETTINGS_PATH, 'WEBUI.latest', current_ui)

## ================ loading settings V5 ==================

def load_settings(path):
    """Load settings from a JSON file."""
    try:
        return {
            **js.read(path, 'ENVIRONMENT'),
            **js.read(path, 'WIDGETS'),
            **js.read(path, 'WEBUI')
        }
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading settings: {e}")
        return {}

# Load settings
settings = load_settings(SETTINGS_PATH)
locals().update(settings)

## ======================== WEBUI ========================

if UI in ['A1111', 'SD-UX'] and not os.path.exists('/root/.cache/huggingface/hub/models--Bingsu--adetailer'):
    print('🚚 Unpacking ADetailer model cache...')

    name_zip = 'hf_cache_adetailer'
    chache_url = 'https://huggingface.co/NagisaNao/ANXETY/resolve/main/hf_chache_adetailer.zip'

    zip_path = f"{HOME}/{name_zip}.zip"
    download_from_url_with_fallbacks(chache_url, HOME, name_zip + ".zip", log=True) # Use new download method
    ipySys(f"unzip -q -o {zip_path} -d /")
    ipySys(f"rm -rf {zip_path}")

    clear_output()

start_timer = js.read(SETTINGS_PATH, 'ENVIRONMENT.start_timer')

if not os.path.exists(WEBUI):
    start_install = time.time()
    print(f"⌚ Unpacking Stable Diffusion... | WEBUI: {COL.B}{UI}{COL.X}", end='')

    ipyRun('run', f"{SCRIPTS}/UIs/{UI}.py")
    handle_setup_timer(WEBUI, start_timer)		# Setup timer (for timer-extensions)

    install_time = time.time() - start_install
    minutes, seconds = divmod(int(install_time), 60)
    print(f"\r🚀 Unpacking {COL.B}{UI}{COL.X} is complete! {minutes:02}:{seconds:02} ⚡" + ' '*25)

else:
    print(f"🔧 Current WebUI: {COL.B}{UI}{COL.X}")
    print('🚀 Unpacking is complete. Pass. ⚡')

    timer_env = handle_setup_timer(WEBUI, start_timer)
    elapsed_time = str(timedelta(seconds=time.time() - timer_env)).split('.')[0]
    print(f"⌚️ Session duration: {COL.Y}{elapsed_time}{COL.X}")


## Changes extensions and WebUi
if latest_webui or latest_extensions:
    action = 'WebUI and Extensions' if latest_webui and latest_extensions else ('WebUI' if latest_webui else 'Extensions')
    print(f"⌚️ Update {action}...", end='')
    with capture.capture_output():
        ipySys('git config --global user.email "you@example.com"')
        ipySys('git config --global user.name "Your Name"')

        ## Update Webui
        if latest_webui:
            CD(WEBUI)
            ipySys('git stash push --include-untracked')
            ipySys('git pull --rebase')
            ipySys('git stash pop')

        ## Update extensions
        if latest_extensions:
            for entry in os.listdir(f"{WEBUI}/extensions"):
                dir_path = f"{WEBUI}/extensions/{entry}"
                if os.path.isdir(dir_path):
                    subprocess.run(['git', 'reset', '--hard'], cwd=dir_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(['git', 'pull'], cwd=dir_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"\r✨ Update {action} Completed!")


# === FIXING EXTENSIONS ===
with capture.capture_output():
    # --- Umi-Wildcard ---
    ipySys(f"sed -i '521s/open=\\(False\\|True\\)/open=False/' {WEBUI}/extensions/Umi-AI-Wildcards/scripts/wildcard_recursive.py")    # Closed accordion by default


## Version switching
if commit_hash:
    print('🔄 Switching to the specified version...', end='')
    with capture.capture_output():
        CD(WEBUI)
        ipySys('git config --global user.email "you@example.com"')
        ipySys('git config --global user.name "Your Name"')
        ipySys(f'git reset --hard {commit_hash}')
        ipySys(f'git pull origin {commit_hash}')    # Get last changes in branch
    print(f"\r🔄 Switch complete! Current commit: {COL.B}{commit_hash}{COL.X}")


# === Google Drive Mounting | EXCLUSIVE for Colab ===
from google.colab import drive
mountGDrive = js.read(SETTINGS_PATH, 'mountGDrive')  # Mount/unmount flag

# Configuration
GD_BASE = "/content/drive/MyDrive/sdAIgen"
SYMLINK_CONFIG = [
    {    # model
        'local_dir': model_dir,
        'gdrive_subpath': 'Checkpoints',
    },
    {    # vae
        'local_dir': vae_dir,
        'gdrive_subpath': 'VAE',
    },
    {    # lora
        'local_dir': lora_dir,
        'gdrive_subpath': 'Lora',
    }
]

def create_symlink(src_path, gdrive_path, log=False):
    """Create symbolic link with content migration and cleanup"""
    try:
        src_exists = os.path.exists(src_path)
        is_real_dir = src_exists and os.path.isdir(src_path) and not os.path.islink(src_path)

        # Handle real directory migration
        if is_real_dir and os.path.exists(gdrive_path):
            for item in os.listdir(src_path):
                src_item = os.path.join(src_path, item)
                dst_item = os.path.join(gdrive_path, item)

                if os.path.exists(dst_item):
                    shutil.rmtree(dst_item) if os.path.isdir(dst_item) else os.remove(dst_item)
                shutil.move(src_item, dst_item)

            shutil.rmtree(src_path)
            if log:
                print(f"Moved contents from {src_path} to {gdrive_path}")

        # Cleanup existing path
        if os.path.exists(src_path) and not is_real_dir:
            if os.path.islink(src_path):
                os.unlink(src_path)
            else:
                os.remove(src_path)

        # Create new symlink
        if not os.path.exists(src_path):
            os.symlink(gdrive_path, src_path)
            if log:
                print(f"Created symlink: {src_path} → {gdrive_path}")

    except Exception as e:
        print(f"Error processing {src_path}: {str(e)}")

def handle_gdrive(mount_flag, log=False):
    """Main handler for Google Drive mounting and symlink management"""
    if mount_flag:
        if os.path.exists("/content/drive/MyDrive"):
            print("🎉 Google Drive is connected~")
        else:
            try:
                print("⏳ Mounting Google Drive...", end='')
                with capture.capture_output():
                    drive.mount('/content/drive')
                print("\r🚀 Google Drive mounted successfully!")
            except Exception as e:
                clear_output()
                print(f"❌ Mounting failed: {str(e)}\n")
                return

        try:
            # Create base directory structure
            os.makedirs(GD_BASE, exist_ok=True)
            for cfg in SYMLINK_CONFIG:
                path = os.path.join(GD_BASE, cfg['gdrive_subpath'])
                os.makedirs(path, exist_ok=True)
            print(f"📁 → {GD_BASE}")

            # Create symlinks
            for cfg in SYMLINK_CONFIG:
                src = os.path.join(cfg['local_dir'], 'GDrive')
                dst = os.path.join(GD_BASE, cfg['gdrive_subpath'])
                create_symlink(src, dst, log)

            print("✅ Symlinks created successfully!")

        except Exception as e:
            print(f"❌ Setup error: {str(e)}\n")

        # Trashing
        cmd = f"find {GD_BASE} -type d -name .ipynb_checkpoints -exec rm -rf {{}} +"
        subprocess.run(shlex.split(cmd), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    else:
        if os.path.exists("/content/drive/MyDrive"):
            try:
                print("⏳ Unmounting Google Drive...", end='')
                with capture.capture_output():
                    drive.flush_and_unmount()
                    os.system("rm -rf /content/drive")
                print("\r✅ Google Drive unmounted and cleaned!")

                # Remove symlinks
                for cfg in SYMLINK_CONFIG:
                    link_path = os.path.join(cfg['local_dir'], 'GDrive')
                    if os.path.islink(link_path):
                        os.unlink(link_path)

                print("🗑️ Symlinks removed successfully!")

            except Exception as e:
                print(f"❌ Unmount error: {str(e)}\n")

handle_gdrive(mountGDrive)


# Get XL or 1.5 models list
## model_list | vae_list | controlnet_list
model_files = '_xl-models-data.py' if XL_models else '_models-data.py'
with open(f"{SCRIPTS}/{model_files}") as f:
    exec(f.read())

## Downloading model and stuff | oh~ Hey! If you're freaked out by that code too, don't worry, me too!
print('📦 Downloading models and stuff...', end='')

extension_repo = []
PREFIX_MAP = {
    # prefix : (dir_path , short-tag)
    'model': (model_dir, '$ckpt'),
    'vae': (vae_dir, '$vae'),
    'lora': (lora_dir, '$lora'),
    'embed': (embed_dir, '$emb'),
    'extension': (extension_dir, '$ext'),
    'adetailer': (adetailer_dir, '$ad'),
    'control': (control_dir, '$cnet'),
    'upscale': (upscale_dir, '$ups'),
    # Other
    'clip': (clip_dir, '$clip'),
    'unet': (unet_dir, '$unet'),
    'vision': (vision_dir, '$vis'),
    'encoder': (encoder_dir, '$enc'),
    'diffusion': (diffusion_dir, '$diff'),
    'config': (config_dir, '$cfg')
}
for dir_path, _ in PREFIX_MAP.values():
    os.makedirs(dir_path, exist_ok=True)

''' Formatted Info Output '''

def _center_text(text, terminal_width=45):
    padding = (terminal_width - len(text)) // 2
    return f"{' ' * padding}{text}{' ' * padding}"

def format_output(url, dst_dir, file_name, image_url=None, image_name=None):
    """Formats and prints download details with colored text."""
    info = '[NONE]'
    if file_name:
        info = _center_text(f"[{file_name.rsplit('.', 1)[0]}]")
    if not file_name and 'drive.google.com' in url:
        info = _center_text('[GDrive]')

    sep_line = '───' * 20

    print()
    print(f"{COL.G}{sep_line}{COL.lB}{info}{COL.G}{sep_line}{COL.X}")
    print(f"{COL.Y}{'URL:':<12}{COL.X}{url}")
    print(f"{COL.Y}{'SAVE DIR:':<12}{COL.B}{dst_dir}")
    print(f"{COL.Y}{'FILE NAME:':<12}{COL.B}{file_name}{COL.X}")
    if 'civitai' in url and image_url:
        print(f"{COL.G}{'[Preview]:':<12}{COL.X}{image_name} → {image_url}")
    print()

''' Main Download Code '''

def _clean_url(url):
    url_cleaners = {
        'huggingface.co': lambda u: u.replace('/blob/', '/resolve/').split('?')[0],
        'github.com': lambda u: u.replace('/blob/', '/raw/')
    }
    for domain, cleaner in url_cleaners.items():
        if domain in url:
            return cleaner(url)
    return url

def _extract_filename(url):
    if match := re.search(r'\[(.*?)', url): # Adjusted regex to correctly capture filename in brackets
        return match.group(1).split(']')[0] # Extract content and remove trailing ']'
    if any(d in urlparse(url).netloc for d in ["civitai.com", "drive.google.com", "orchestration.civitai.com"]):
        # For Civitai direct URLs, try to get the filename from the path
        if 'orchestration.civitai.com' in url:
            return url.split('/')[-1].split('?')[0]
        # For other known domains, if no bracketed filename, let manual_download infer
        return Path(urlparse(url).path).name if Path(urlparse(url).path).name else None # If no filename in path, return None
    return Path(urlparse(url).path).name

def _unpack_zips():
    """Recursively extract and delete all .zip files in PREFIX_MAP directories."""
    for dir_path, _ in PREFIX_MAP.values():
        for zip_file in Path(dir_path).rglob('*.zip'):
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(zip_file.with_suffix(''))
            zip_file.unlink()

# --- New: HuggingFace API Wrapper ---
class HuggingFaceAPI:
    def __init__(self, token=None):
        self.headers = {}
        if token:
            self.headers['Authorization'] = f'Bearer {token}'

    def download_file(self, repo_id, file_path, local_dir, local_filename=None):
        base_url = "https://huggingface.co"
        url = f"{base_url}/{repo_id}/resolve/main/{file_path}"
        if not local_filename:
            local_filename = Path(file_path).name

        os.makedirs(local_dir, exist_ok=True)
        local_filepath = Path(local_dir) / local_filename

        print(f"Attempting Hugging Face API download for {local_filename} from {url}")
        try:
            with requests.get(url, headers=self.headers, stream=True) as r:
                r.raise_for_status()
                with open(local_filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f"Hugging Face API download successful: {local_filepath}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Hugging Face API download failed: {e}")
            return False

# --- New: Download Fallback Mechanism ---
def download_from_url_with_fallbacks(url, dst_dir, file_name=None, log=False):
    """
    Attempts to download a file using m_download, then falls back to Hugging Face API,
    then Civitai API if applicable.
    """
    if not file_name:
        file_name = _extract_filename(url)
    if not file_name:
        print(f"{COL.R}Error: Could not determine filename for URL: {url}{COL.X}")
        return False

    os.makedirs(dst_dir, exist_ok=True)
    target_path = Path(dst_dir) / file_name

    # Attempt 1: Normal m_download
    print(f"\nAttempting normal download for {file_name} to {dst_dir}...")
    try:
        # m_download typically takes arguments like: "url dst_dir filename"
        # It's assumed m_download handles the actual file saving and progress.
        success = m_download(f"{url} {dst_dir} {file_name}", log=log)
        if success and target_path.exists() and target_path.stat().st_size > 0:
            print(f"{COL.G}Normal download successful: {file_name}{COL.X}")
            return True
        else:
            print(f"{COL.Y}Normal download did not complete successfully or file is empty. Trying fallbacks...{COL.X}")
    except Exception as e:
        print(f"{COL.Y}Normal download failed with error: {e}. Trying fallbacks...{COL.X}")

    # Attempt 2: Hugging Face API
    if "huggingface.co" in url:
        print(f"Attempting Hugging Face API download for {file_name}...")
        try:
            hf_api = HuggingFaceAPI(huggingface_token)
            # Heuristic to extract repo_id and file_path from Hugging Face URL
            match = re.match(r"https://huggingface.co/([^/]+/[^/]+)/resolve/main/(.*)", url)
            if match:
                repo_id = match.group(1)
                hf_file_path = match.group(2)
                if hf_api.download_file(repo_id, hf_file_path, dst_dir, file_name):
                    print(f"{COL.G}Hugging Face API download successful: {file_name}{COL.X}")
                    return True
            else:
                print(f"{COL.Y}Could not parse Hugging Face URL for API download: {url}{COL.X}")
        except Exception as e:
            print(f"{COL.R}Hugging Face API download failed: {e}{COL.X}")

    # Attempt 3: Civitai API
    if "civitai.com" in url:
        print(f"Attempting Civitai API download for {file_name}...")
        try:
            api = CivitAiAPI(civitai_token)
            data = api.validate_download(url, file_name)
            if data and data.download_url:
                civitai_download_url = data.download_url
                # Use m_download again, but with the specific Civitai direct download URL
                success = m_download(f"{civitai_download_url} {dst_dir} {file_name}", log=log)
                if success and target_path.exists() and target_path.stat().st_size > 0:
                    print(f"{COL.G}Civitai API download successful: {file_name}{COL.X}")
                    # Download preview images if available
                    if data.image_url and data.image_name:
                        download_from_url_with_fallbacks(data.image_url, dst_dir, data.image_name, log=log)
                    return True
                else:
                    print(f"{COL.R}Civitai API download via m_download failed or file is empty.{COL.X}")
            else:
                print(f"{COL.R}Civitai API validation failed or no download URL found for {url}{COL.X}")
        except Exception as e:
            print(f"{COL.R}Civitai API download failed: {e}{COL.X}")

    print(f"{COL.R}Failed to download {file_name} from any method.{COL.X}")
    return False


def _process_download_link(link):
    """Processes a download link, splitting prefix, URL, and filename."""
    link = _clean_url(link)
    
    # Try to parse as a general URL with optional destination and filename
    # Format: "url [destination_dir] [filename]"
    parts = shlex.split(link) # Use shlex to handle spaces in paths/filenames correctly

    if len(parts) >= 1 and parts[0].startswith('http'):
        url = parts[0]
        dst_dir = str(Path.cwd()) # Default to current working directory
        file_name = None

        if len(parts) > 1:
            # Check if the second part is a valid directory
            if Path(parts[1]).is_dir() or not Path(parts[1]).suffix: # Assume it's a directory if it exists or has no extension
                dst_dir = parts[1]
                if len(parts) > 2:
                    file_name = parts[2]
            else: # If it's not a directory, assume it's the filename
                file_name = parts[1]

        # Extract filename if not explicitly provided
        if not file_name:
            file_name = _extract_filename(url)

        return None, url, dst_dir, file_name # Return None for prefix to indicate general download
    
    # Fallback to original prefixed format if not a general URL
    if ':' in link and link.split(':', 1)[0] in PREFIX_MAP:
        prefix, path = link.split(':', 1)
        return prefix, re.sub(r'\[.*?\]', '', path), None, _extract_filename(path)
    
    # If nothing matches, it's an unhandled format, return None for all
    return None, None, None, None


def download(line):
    """Downloads files from comma-separated links, processes prefixes, and unpacks zips post-download."""
    for link in filter(None, map(str.strip, line.split(','))):
        prefix, url, dst_dir, filename = _process_download_link(link)

        if url is None: # Handle cases where _process_download_link couldn't parse the link
            print(f"\n{COL.R}Skipping invalid download link: {link}{COL.X}")
            continue

        if prefix: # Original prefixed download
            dir_path, _ = PREFIX_MAP[prefix]
            if prefix == 'extension':
                extension_repo.append((url, filename))
                continue
            try:
                # For prefixed downloads, still try fallbacks if manual_download uses it
                manual_download(url, dir_path, filename, prefix)
            except Exception as e:
                print(f"\n> Download error: {e}")
        else: # General download (no prefix specified, or direct URL with optional path/filename)
            try:
                manual_download(url, dst_dir, filename)
            except Exception as e:
                print(f"\n> Download error: {e}")

    _unpack_zips()

def manual_download(url, dst_dir, file_name=None, prefix=None):
    clean_url = url
    image_url, image_name = None, None

    # Note: Civitai API handling is now primarily within download_from_url_with_fallbacks,
    # but we keep this initial validation for consistency if needed for specific logic outside the fallback.
    if 'civitai.com' in url and prefix is not None and prefix in PREFIX_MAP and prefix != 'extension':
        api = CivitAiAPI(civitai_token)
        data = api.validate_download(url, file_name)
        if not data: # Check if validation failed
            print(f"\n{COL.R}[API Error]: Invalid Civitai model URL or API validation failed: {url}{COL.X}")
            return # Exit if API validation fails

        # These are used for formatting output and potentially for the Civitai API specific download within fallbacks
        model_type, file_name = data.model_type, data.model_name    # Type, name
        clean_url, url = data.clean_url, data.download_url          # Clean_URL, URL (this is the direct download URL)
        image_url, image_name = data.image_url, data.image_name    # Img_URL, Img_Name

        # Download preview images using the fallback mechanism
        if image_url and image_name:
            download_from_url_with_fallbacks(image_url, dst_dir, image_name, log=True)

    elif any(s in url for s in ('github', 'huggingface.co')) and prefix is not None and prefix in PREFIX_MAP:
        if file_name and '.' not in file_name:
            file_name += f".{clean_url.split('.')[-1]}"
    else: # For general downloads and direct Civitai URLs that bypass API
        if not file_name:
            file_name = Path(urlparse(url).path).name
            if not file_name and 'orchestration.civitai.com' in url:
                file_name = url.split('/')[-1].split('?')[0] # Get filename from direct Civitai asset URL

    # Ensure dst_dir exists
    os.makedirs(dst_dir, exist_ok=True)

    # Formatted info output
    format_output(clean_url, dst_dir, file_name, image_url, image_name)

    # Use the new download fallback mechanism for the actual download
    download_from_url_with_fallbacks(url, dst_dir, file_name, log=True)


''' SubModels - Added URLs '''

# Separation of merged numbers
def _parse_selection_numbers(num_str, max_num):
    """Split a string of numbers into unique integers, considering max_num as the upper limit."""
    num_str = num_str.replace(',', ' ').strip()
    unique_numbers = set()
    max_length = len(str(max_num))

    for part in num_str.split():
        if not part.isdigit():
            continue

        # Check if the entire part is a valid number
        part_int = int(part)
        if part_int <= max_num:
            unique_numbers.add(part_int)
            continue  # No need to split further

        # Split the part into valid numbers starting from the longest possible
        current_position = 0
        part_len = len(part)
        while current_position < part_len:
            found = False
            # Try lengths from max_length down to 1
            for length in range(min(max_length, part_len - current_position), 0, -1):
                substring = part[current_position:current_position + length]
                if substring.isdigit():
                    num = int(substring)
                    if num <= max_num and num != 0:
                        unique_numbers.add(num)
                        current_position += length
                        found = True
                        break
            if not found:
                # Move to the next character if no valid number found
                current_position += 1

    return sorted(unique_numbers)

def handle_submodels(selection, num_selection, model_dict, dst_dir, base_url, inpainting_model=False):
    selected = []
    if selection == "ALL":
        selected = sum(model_dict.values(), [])
    elif selection in model_dict:
        selected.extend(model_dict[selection])

    if num_selection:
        max_num = len(model_dict)
        for num in _parse_selection_numbers(num_selection, max_num):
            if 1 <= num <= max_num:
                name = list(model_dict.keys())[num - 1]
                selected.extend(model_dict[name])

    unique_models = {}
    for model in selected:
        name = model.get('name') or os.path.basename(model['url'])
        if not inpainting_model and "inpainting" in name:
            continue
        unique_models[name] = {
            'url': model['url'],
            'dst_dir': model.get('dst_dir', dst_dir),
            'name': name
        }

    return base_url + ', '.join(
        f"{m['url']} {m['dst_dir']} {m['name']}"
        for m in unique_models.values()
    )

line = ""
line = handle_submodels(model, model_num, model_list, model_dir, line)
line = handle_submodels(vae, vae_num, vae_list, vae_dir, line)
line = handle_submodels(controlnet, controlnet_num, controlnet_list, control_dir, line)

''' File.txt - added urls '''

def _process_lines(lines):
    """Processes text lines, extracts valid URLs with tags/filenames, and ensures uniqueness."""
    current_tag = None
    processed_entries = set()    # Store (tag, clean_url) to check uniqueness
    result_urls = []

    for line in lines:
        clean_line = line.strip()

        # Update the current tag when detected
        found_tag = False
        for prefix, (_, short_tag) in PREFIX_MAP.items():
            if (f"# {prefix}".lower() in clean_line.lower()) or (short_tag and short_tag.lower() in clean_line.lower()):
                current_tag = prefix
                found_tag = True
                break

        # Process the line based on whether a tag was found or if it's a direct URL
        if found_tag:
            # If a tag is explicitly present, it's a prefixed download
            normalized_line = re.sub(r'[\s,]+', ',', clean_line)
            for url_entry in normalized_line.split(','):
                url_part = url_entry.split('#')[0].strip()
                if not url_part.startswith('http'):
                    continue

                clean_url_for_key = re.sub(r'\[.*?\]', '', url_part)
                entry_key = (current_tag, clean_url_for_key)

                if entry_key not in processed_entries:
                    filename = _extract_filename(url_part)
                    formatted_url = f"{current_tag}:{clean_url_for_key}"
                    if filename:
                        formatted_url += f"[{filename}]"
                    result_urls.append(formatted_url)
                    processed_entries.add(entry_key)
        elif clean_line.startswith('http'):
            # It's a direct URL, handle as general download
            # Use shlex.split to handle spaces in potential path/filename components
            parts = shlex.split(clean_line)
            url = parts[0]
            dst_dir = str(Path.cwd())
            file_name = None

            if len(parts) > 1:
                if Path(parts[1]).is_dir() or not Path(parts[1]).suffix:
                    dst_dir = parts[1]
                    if len(parts) > 2:
                        file_name = parts[2]
                else:
                    file_name = parts[1]
            
            if not file_name:
                file_name = _extract_filename(url)

            entry_key = (None, url) # For general downloads, key only on the URL
            if entry_key not in processed_entries:
                result_urls.append(f"{url} {dst_dir} {file_name if file_name else ''}".strip())
                processed_entries.add(entry_key)

    return ', '.join(result_urls) if result_urls else ''


def process_file_downloads(file_urls, additional_lines=None):
    """Reads URLs from files/HTTP sources."""
    lines = []

    if additional_lines:
        lines.extend(additional_lines.splitlines())

    for source in file_urls:
        if source.startswith('http'):
            try:
                response = requests.get(_clean_url(source))
                response.raise_for_status()
                lines.extend(response.text.splitlines())
            except requests.RequestException:
                continue
        else:
            try:
                with open(source, 'r', encoding='utf-8') as f:
                    lines.extend(f.readlines())
            except FileNotFoundError:
                continue

    return _process_lines(lines)

# File URLs processing
urls_sources = (Model_url, Vae_url, LoRA_url, Embedding_url, Extensions_url, ADetailer_url)
file_urls = [f"{f}.txt" if not f.endswith('.txt') else f for f in custom_file_urls.replace(',', '').split()] if custom_file_urls else []

# p -> prefix ; u -> url | Remember: don't touch the prefix!
prefixed_urls = [f"{p}:{u}" for p, u in zip(PREFIX_MAP, urls_sources) if u for u in u.replace(',', '').split()]
line += ', ' + ', '.join(prefixed_urls + [process_file_downloads(file_urls, empowerment_output)])

# Add a placeholder for directly passed URLs
# Example: If you want to download a specific file, you can set this variable.
# For example: url_to_download_any_file = "https://example.com/somefile.zip /content/my_downloads/my_archive.zip"
# Or simple: url_to_download_any_file = "https://orchestration.civitai.com/v1/consumer/jobs/e4799ea1-fbff-40d4-81cd-6184248fab5e/assets/Serapias_Alice.safetensors /root/ComfyUI/models/loras/Serapias_Alice.safetensors"
# If no destination directory is given, it will default to the current working directory.
# If no filename is given, it will try to infer it from the URL.
url_to_download_any_file = "https://orchestration.civitai.com/v1/consumer/jobs/e4799ea1-fbff-40d4-81cd-6184248fab5e/assets/Serapias_Alice.safetensors /root/ComfyUI/models/loras/Serapias_Alice.safetensors"


if 'url_to_download_any_file' in locals() and url_to_download_any_file:
    # Ensure the path is properly formatted for the `download` function
    line += ", " + url_to_download_any_file


if detailed_download == 'on':
    print(f"\n\n{COL.Y}# ====== Detailed Download ====== #\n{COL.X}")
    download(line)
    print(f"\n{COL.Y}# =============================== #\n{COL.X}")
else:
    with capture.capture_output():
        download(line)

print('\r🏁 Download Complete!' + ' '*15)


## Install of Custom extensions
def _clone_repository(repo, repo_name, extension_dir):
    """Clones the repository to the specified directory."""
    repo_name = repo_name or repo.split('/')[-1]
    command = f"cd {extension_dir} && git clone --depth 1 --recursive {repo} {repo_name} && cd {repo_name} && git fetch"
    ipySys(command)

extension_type = 'nodes' if UI == 'ComfyUI' else 'extensions'

if extension_repo:
    print(f"✨ Installing custom {extension_type}...", end='')
    with capture.capture_output():
        for repo, repo_name in extension_repo:
            _clone_repository(repo, repo_name, extension_dir)
    print(f"\r📦 Installed '{len(extension_repo)}' custom {extension_type}!")


# === SPECIAL ===
## Sorting models `bbox` and `segm` | Only ComfyUI
if UI == 'ComfyUI':
    dirs = {'segm': '-seg.pt', 'bbox': None}
    for d in dirs:
        os.makedirs(os.path.join(adetailer_dir, d), exist_ok=True)

    for filename in os.listdir(adetailer_dir):
        src = os.path.join(adetailer_dir, filename)

        if os.path.isfile(src) and filename.endswith('.pt'):
            dest_dir = 'segm' if filename.endswith('-seg.pt') else 'bbox'
            dest = os.path.join(adetailer_dir, dest_dir, filename)

            if os.path.exists(dest):
                os.remove(src)
            else:
                shutil.move(src, dest)


## List Models and stuff
ipyRun('run', f"{SCRIPTS}/download-result.py")

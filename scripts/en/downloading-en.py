# ~ download.py | by ANXETY ~

from webui_utils import handle_setup_timer    # WEBUI
from CivitaiAPI import CivitAiAPI             # CivitAI API
from Manager import m_download                # Every Download
import json_utils as js                       # JSON

from IPython.display import clear_output
from IPython.utils import capture
from urllib.parse import urlparse, unquote
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

# Text Colors (\033)
class COLORS:
    R  =  "\033[31m"     # Red
    G  =  "\033[32m"     # Green
    Y  =  "\033[33m"     # Yellow
    B  =  "\033[34m"     # Blue
    lB =  "\033[36;1m"   # lightBlue
    X  =  "\033[0m"      # Reset

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

    m_download(f"{url} {HOME} {fn}")

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
    m_download(f"{chache_url} {HOME} {name_zip}")
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
        ipySys('git reset --hard {commit_hash}')
        ipySys('git pull origin {commit_hash}')    # Get last changes in branch
    print(f"\r🔄 Switch complete! Current commit: {COL.B}{commit_hash}{COL.X}")


# === Google Drive Mounting | EXCLUSIVE for Colab ===
from google.colab import drive
mountGDrive = js.read(SETTINGS_PATH, 'mountGDrive')  # Mount/unmount flag

# Configuration
GD_BASE = "/content/drive/MyDrive/sdAIgen"
SYMLINK_CONFIG = [
    {   # model
        'local_dir': model_dir,
        'gdrive_subpath': 'Checkpoints',
    },
    {   # vae
        'local_dir': vae_dir,
        'gdrive_subpath': 'VAE',
    },
    {   # lora
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
    parsed_url = urlparse(url)
    if 'huggingface.co' in parsed_url.netloc:
        return url.replace('/blob/', '/resolve/').split('?')[0]
    elif 'github.com' in parsed_url.netloc:
        return url.replace('/blob/', '/raw/')
    return url.split('?')[0] # Remove query parameters for generic URLs

def _extract_filename_from_url(url):
    parsed_url = urlparse(url)
    filename = Path(unquote(parsed_url.path)).name
    if filename:
        return filename
    return None

def _get_download_info(original_url, desired_filename=None, suggested_prefix=None):
    """
    Analyzes the URL to determine download type, actual URL, target directory, and filename.
    Returns: (download_url, target_dir, filename, image_url, image_name, is_extension)
    """
    clean_url = _clean_url(original_url)
    parsed_url = urlparse(clean_url)
    netloc = parsed_url.netloc

    target_dir = None
    file_name = desired_filename
    image_url = None
    image_name = None
    is_extension = False

    # 1. Handle prefixed URLs (e.g., 'model:https://...')
    if suggested_prefix and suggested_prefix in PREFIX_MAP:
        target_dir = PREFIX_MAP[suggested_prefix][0]
        if suggested_prefix == 'extension':
            is_extension = True
        if not file_name:
            file_name = _extract_filename_from_url(clean_url)

    # 2. Civitai API
    if 'civitai.com' in netloc:
        try:
            api = CivitAiAPI(civitai_token)
            data = api.validate_download(original_url, desired_filename)
            if data:
                download_url = data.download_url
                file_name = data.model_name
                image_url = data.image_url
                image_name = data.image_name
                # Determine target_dir based on model_type from Civitai if not already set by prefix
                if not target_dir:
                    if data.model_type == 'Checkpoint':
                        target_dir = model_dir
                    elif data.model_type == 'VAE':
                        target_dir = vae_dir
                    elif data.model_type == 'LORA':
                        target_dir = lora_dir
                    elif data.model_type == 'TextualInversion':
                        target_dir = embed_dir
                    elif data.model_type == 'Controlnet':
                        target_dir = control_dir
                    elif data.model_type == 'LoCon':
                        target_dir = lora_dir # LoCon is a type of LoRA
                    else:
                        target_dir = model_dir # Default if unknown

                return download_url, target_dir, file_name, image_url, image_name, is_extension
        except Exception as e:
            print(f"{COL.R}Error with Civitai API for {original_url}: {e}{COL.X}")

    # 3. Hugging Face and GitHub (already cleaned by _clean_url)
    elif 'huggingface.co' in netloc or 'github.com' in netloc:
        if not file_name:
            file_name = _extract_filename_from_url(clean_url)
        # Attempt to infer target_dir if not set by prefix
        if not target_dir:
            if any(ext in file_name for ext in ['.safetensors', '.ckpt', '.pt']):
                target_dir = model_dir
            elif 'vae' in file_name.lower():
                target_dir = vae_dir
            elif 'lora' in file_name.lower():
                target_dir = lora_dir
            elif 'embeddings' in clean_url or 'embed' in file_name.lower():
                target_dir = embed_dir
            elif 'controlnet' in clean_url or 'cnet' in file_name.lower():
                target_dir = control_dir
            else:
                target_dir = model_dir # Default fallback

        return clean_url, target_dir, file_name, None, None, is_extension

    # 4. Generic URL
    if not file_name:
        file_name = _extract_filename_from_url(clean_url)

    # If target_dir is still None, default to a general downloads directory or model_dir
    if not target_dir:
        target_dir = model_dir # Or a more generic 'downloads_dir' if you create one

    return clean_url, target_dir, file_name, None, None, is_extension

def _unpack_zips():
    """Recursively extract and delete all .zip files in PREFIX_MAP directories."""
    for dir_path, _ in PREFIX_MAP.values():
        for zip_file in Path(dir_path).rglob('*.zip'):
            try:
                print(f"📦 Extracting {zip_file.name}...")
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    zf.extractall(zip_file.with_suffix(''))
                zip_file.unlink()
                print(f"🗑️ Deleted {zip_file.name}.")
            except zipfile.BadZipFile:
                print(f"{COL.R}Warning: {zip_file.name} is a bad zip file, skipping extraction.{COL.X}")
            except Exception as e:
                print(f"{COL.R}Error extracting {zip_file.name}: {e}{COL.X}")

def manual_download(url, dst_dir, file_name=None, image_url=None, image_name=None):
    """Performs the actual download using m_download."""
    format_output(url, dst_dir, file_name, image_url, image_name)
    m_download(f"{url} {dst_dir} {file_name or ''}", log=True)
    
def download(line):
    """
    Downloads files from comma-separated links, processes prefixes,
    and unpacks zips post-download. Now handles any URL.
    """
    download_items = []
    # Parse the input line for multiple URLs
    for item in filter(None, map(str.strip, line.split(','))):
        # Try to parse prefix:url[filename] format
        prefix_match = re.match(r'^(.*?):(.*?)\[(.*?)\]$', item)
        if prefix_match:
            prefix, url_part, filename_part = prefix_match.groups()
            download_items.append((url_part, filename_part, prefix))
        else:
            # Handle plain URL or URL with implicit filename/no prefix
            filename_match = re.search(r'\[(.*?)\]', item)
            if filename_match:
                filename_part = filename_match.group(1)
                url_part = re.sub(r'\[.*?\]', '', item).strip() # .strip() added for safety
            else:
                url_part = item.strip() # .strip() added for safety
                filename_part = None

            prefix_from_map = None
            # Check if any known short tag is present in the URL for inference
            for p, (_, short_tag) in PREFIX_MAP.items():
                if short_tag and short_tag.lower() in url_part.lower():
                    prefix_from_map = p
                    break
            download_items.append((url_part, filename_part, prefix_from_map))


    for original_url, desired_filename, suggested_prefix in download_items:
        try:
            (download_url, target_dir, final_filename,
             image_url, image_name, is_extension) = _get_download_info(
                original_url, desired_filename, suggested_prefix
            )

            if is_extension:
                # Assuming extension URLs are git repositories
                extension_repo.append((download_url, final_filename))
                print(f"📝 Added extension: {final_filename or download_url} to be cloned later.")
                continue

            if not download_url:
                print(f"{COL.R}Skipping download for: {original_url} (No valid download URL found).{COL.X}")
                continue

            # Download preview images for Civitai
            if image_url and image_name:
                print(f"🖼️ Downloading preview image: {image_name}...")
                m_download(f"{image_url} {target_dir} {image_name}", log=False) # No need for detailed log for images

            # This is the correct way to call manual_download now
            manual_download(download_url, target_dir, final_filename, image_url, image_name)

        except Exception as e:
            print(f"\n{COL.R}> Failed to process download for {original_url}: {e}{COL.X}")

    _unpack_zips()


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

    # Format the output to be compatible with the new 'download' function
    # Each entry can be 'url' or 'prefix:url[filename]'
    formatted_downloads = []
    for m in unique_models.values():
        # Infer prefix from dst_dir or use a general if not found
        inferred_prefix = None
        for p, (d_path, _) in PREFIX_MAP.items():
            if Path(m['dst_dir']) == Path(d_path):
                inferred_prefix = p
                break
        
        if inferred_prefix:
            formatted_downloads.append(f"{inferred_prefix}:{m['url']}[{m['name']}]")
        else:
            # Fallback for directly providing URL and destination
            formatted_downloads.append(f"{m['url']}[{m['name']}]")

    return base_url + ', '.join(formatted_downloads)


line = ""
line = handle_submodels(model, model_num, model_list, model_dir, line)
line = handle_submodels(vae, vae_num, vae_list, vae_dir, line)
line = handle_submodels(controlnet, controlnet_num, controlnet_list, control_dir, line)

''' File.txt - added urls '''

def _process_lines(lines):
    """Processes text lines, extracts valid URLs with tags/filenames, and ensures uniqueness."""
    processed_entries = set()  # Store (tag, clean_url) to check uniqueness
    result_urls = []

    for line in lines:
        clean_line = line.strip() # Don't lower-case the whole line yet

        # Check for tags/prefixes at the start of the line or implicitly
        current_tag = None
        for prefix, (_, short_tag) in PREFIX_MAP.items():
            if clean_line.lower().startswith(f"# {prefix}".lower()) or (short_tag and clean_line.lower().startswith(short_tag.lower())):
                current_tag = prefix
                # Remove the tag from the line to parse the URL part
                clean_line = clean_line[len(f"# {prefix}"):].strip() if clean_line.lower().startswith(f"# {prefix}".lower()) else clean_line[len(short_tag):].strip()
                break
        
        # If a line starts with a URL, treat it as a generic URL unless a prefix was already identified
        if not current_tag and (clean_line.startswith('http://') or clean_line.startswith('https://')):
            # Assume it's a generic download without a specific prefix mapping initially
            pass # current_tag remains None for generic URLs

        # Normalise the delimiters and process each URL
        normalized_line = re.sub(r'[\s,]+', ',', clean_line)
        for url_entry in normalized_line.split(','):
            url_part_and_filename = url_entry.split('#')[0].strip() # Remove comments

            # Extract explicit filename in [] if present
            explicit_filename = None
            filename_match = re.search(r'\[(.*?)\]', url_part_and_filename)
            if filename_match:
                explicit_filename = filename_match.group(1)
                url = re.sub(r'\[.*?\]', '', url_part_and_filename).strip()
            else:
                url = url_part_and_filename

            if not url.startswith('http'):
                continue

            # Clean the URL for uniqueness check
            clean_url_for_check = _clean_url(url)
            entry_key = (current_tag, clean_url_for_check)    # Uniqueness is determined by a pair (tag, URL)

            if entry_key not in processed_entries:
                formatted_url = url # Start with the original URL

                # If a prefix was identified, add it back to the formatted URL
                if current_tag:
                    formatted_url = f"{current_tag}:{formatted_url}"
                
                # Add explicit filename if it was found
                if explicit_filename:
                    formatted_url += f"[{explicit_filename}]"

                result_urls.append(formatted_url)
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
                print(f"📡 Fetching URLs from remote file: {source}")
                response = requests.get(_clean_url(source))
                response.raise_for_status()
                lines.extend(response.text.splitlines())
            except requests.RequestException as e:
                print(f"{COL.R}Error fetching remote file {source}: {e}{COL.X}")
                continue
        else:
            try:
                print(f"📁 Reading URLs from local file: {source}")
                with open(source, 'r', encoding='utf-8') as f:
                    lines.extend(f.readlines())
            except FileNotFoundError:
                print(f"{COL.R}Local file not found: {source}{COL.X}")
                continue
            except Exception as e:
                print(f"{COL.R}Error reading local file {source}: {e}{COL.X}")
                continue

    return _process_lines(lines)

# File URLs processing
urls_sources = (Model_url, Vae_url, LoRA_url, Embedding_url, Extensions_url, ADetailer_url)
# Combine urls_sources and custom_file_urls, filtering out empty ones
all_file_sources = []
for src_list in urls_sources:
    if src_list:
        all_file_sources.extend([s.strip() for s in src_list.replace(',', ' ').split()])
if custom_file_urls:
    all_file_sources.extend([f.strip() for f in custom_file_urls.replace(',', ' ').split()])
file_urls_to_process = [f"{f}.txt" if not f.endswith('.txt') else f for f in all_file_sources]

# Initialize line with submodel downloads if any
final_download_line = line # This 'line' already contains submodel downloads

# Process URLs from files and additional empowerment_output
processed_file_urls = process_file_downloads(file_urls_to_process, empowerment_output)
if processed_file_urls:
    if final_download_line:
        final_download_line += ', ' + processed_file_urls
    else:
        final_download_line = processed_file_urls

# This is the actual download call now
if detailed_download == 'on':
    print(f"\n\n{COL.Y}# ====== Detailed Download ====== #\n{COL.X}")
    download(final_download_line)
    print(f"\n{COL.Y}# =============================== #\n{COL.X}")
else:
    with capture.capture_output():
        download(final_download_line)

print('\r🏁 Download Complete!' + ' '*15)


## Install of Custom extensions
def _clone_repository(repo, repo_name, extension_dir):
    """Clones the repository to the specified directory."""
    repo_name = repo_name or repo.split('/')[-1]
    # Ensure the target directory exists before cloning
    os.makedirs(extension_dir, exist_ok=True)
    target_path = Path(extension_dir) / repo_name
    if target_path.exists():
        print(f"Skipping cloning '{repo_name}': directory already exists at {target_path}")
        return

    # Use check_call for better error handling and direct output for clarity
    try:
        print(f"Cloning {repo} into {target_path}...")
        subprocess.check_call(shlex.split(f"git clone --depth 1 --recursive {repo} {target_path}"))
        print(f"Successfully cloned {repo_name}.")
        # No need for cd and git fetch in separate commands after recursive clone to depth 1
    except subprocess.CalledProcessError as e:
        print(f"{COL.R}Error cloning {repo}: {e}{COL.X}")
    except Exception as e:
        print(f"{COL.R}An unexpected error occurred while cloning {repo}: {e}{COL.X}")


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

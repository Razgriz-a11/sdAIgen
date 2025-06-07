# ~ download.py | by ANXETY ~

from webui_utils import handle_setup_timer    # WEBUI
from CivitaiAPI import CivitAiAPI            # CivitAI API
from Manager import m_download                 # Every Download
import json_utils as js                        # JSON

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
    # url = "https://huggingface.co/NagisaNao/ANXETY/resolve/main/python31017-venv-torch251-cu121-C-fca.tar.lz4"
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
    handle_setup_timer(WEBUI, start_timer)        # Setup timer (for timer-extensions)

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
        'github.com': lambda u: u.replace('/blob/', '/raw/').split('?')[0] # Added split for github
    }
    for domain, cleaner in url_cleaners.items():
        if domain in url:
            return cleaner(url)
    return url

def _extract_filename(url, default_filename=None):
    """
    Extracts filename from URL, handling specific cases and general URLs.
    Prioritizes filename provided in URL via [filename], then Content-Disposition header,
    then infers from URL path.
    """
    parsed_url = urlparse(url)

    # 1. Try to get filename from Content-Disposition header for general URLs
    try:
        with requests.head(url, allow_redirects=True, timeout=5) as r:
            r.raise_for_status()
            if 'content-disposition' in r.headers:
                cd_header = r.headers['content-disposition']
                if 'filename=' in cd_header:
                    # Extract filename considering potential quotes and special chars
                    fname_match = re.search(r'filename\*?=(?:UTF-8\'\'|")?([^;"]+)"?', cd_header)
                    if fname_match:
                        return unquote(fname_match.group(1))
    except requests.exceptions.RequestException:
        pass # Silently fail if head request doesn't work

    # 2. Special handling for Civitai orchestration links which often contain the filename
    if 'orchestration.civitai.com' in url:
        path_name = Path(parsed_url.path).name
        if path_name and '.' in path_name: # Check if it looks like a filename
            return unquote(path_name)

    # 3. Infer filename from URL path for non-Civitai/Google Drive URLs
    if not any(d in parsed_url.netloc for d in ["civitai.com", "drive.google.com"]):
        path = Path(parsed_url.path)
        if path.name:
            # If the path name contains a dot, it likely has an extension, so use it.
            # Otherwise, use the default_filename if provided.
            if '.' in path.name:
                return unquote(path.name)
            elif default_filename:
                return default_filename
    
    # If no specific filename is found or inferred, return None to let the downloader decide or prompt.
    return default_filename


def _unpack_zips():
    """Recursively extract and delete all .zip files in PREFIX_MAP directories."""
    for dir_path, _ in PREFIX_MAP.values():
        for zip_file in Path(dir_path).rglob('*.zip'):
            with zipfile.ZipFile(zip_file, 'r') as zf:
                # Extract to a subdirectory with the same name as the zip file (without .zip)
                extract_path = zip_file.with_suffix('')
                os.makedirs(extract_path, exist_ok=True)
                zf.extractall(extract_path)
            zip_file.unlink()

# Download Core

def _process_download_link(link):
    """
    Processes a download link and returns a dictionary with parsed information.
    Returns:
        dict: {'type': 'prefixed' or 'civitai_api' or 'generic', 'url': str, 'dst_dir': str, 'filename': str or None}
    """
    original_link = link
    explicit_filename = None

    # 1. Check for filename embedded in URL (e.g., [modelname.safetensors])
    if match := re.search(r'\[(.*?)\]', original_link):
        explicit_filename = unquote(match.group(1))
        # Remove the filename tag from the URL part for further processing
        link_without_tag = re.sub(r'\[.*?\]', '', original_link).strip()
    else:
        link_without_tag = original_link.strip()

    # 2. Special handling for Civitai Model page URLs that require API (not direct orchestration links)
    if 'civitai.com/models' in link_without_tag and 'orchestration.civitai.com' not in link_without_tag:
        return {'type': 'civitai_api', 'url': _clean_url(link_without_tag), 'dst_dir': None, 'filename': explicit_filename}

    # 3. Try to parse as prefixed URL (e.g., "model:http://example.com/file")
    parts = link_without_tag.split(':', 1)
    if len(parts) > 1 and parts[0] in PREFIX_MAP:
        prefix = parts[0]
        url = _clean_url(parts[1].strip())
        dst_dir, _ = PREFIX_MAP[prefix]
        filename = explicit_filename if explicit_filename else _extract_filename(url)
        return {'type': 'prefixed', 'prefix': prefix, 'url': url, 'dst_dir': dst_dir, 'filename': filename}

    # 4. Try to parse as generic URL with specified destination and optional filename (e.g., "http://example.com/file /path/to/save filename.ext")
    space_separated_parts = link_without_tag.split(' ')
    if len(space_separated_parts) >= 2 and space_separated_parts[0].startswith('http'):
        url = _clean_url(space_separated_parts[0])
        dst_dir = space_separated_parts[1]
        filename = explicit_filename # Prioritize explicit filename from tag

        if not filename and len(space_separated_parts) > 2: # If no explicit filename from tag, check if it's given after dst_dir
            filename = space_separated_parts[2]
        
        if not filename: # If still no filename, try to infer from URL
            filename = _extract_filename(url)

        return {'type': 'generic', 'url': url, 'dst_dir': dst_dir, 'filename': filename}
    
    # 5. Fallback to generic URL with no specified destination (m_download will handle this)
    if original_link.startswith('http'):
        url = _clean_url(original_link)
        filename = explicit_filename if explicit_filename else _extract_filename(url)
        return {'type': 'generic', 'url': url, 'dst_dir': None, 'filename': filename}

    return {'type': 'invalid'}


def download(line):
    """Downloads files from comma-separated links, processes prefixes, and unpacks zips post-download."""
    for link in filter(None, map(str.strip, line.split(','))):
        parsed_info = _process_download_link(link)

        download_successful = False # Flag to track if download was initiated

        if parsed_info['type'] == 'civitai_api':
            url = parsed_info['url']
            file_name = parsed_info['filename'] # This would be the explicit filename from tag if provided
            try:
                # For Civitai API models, we need to know the target directory
                # If not provided via prefix, use a sensible default (e.g., model_dir or lora_dir etc.)
                # We can't determine this from the _process_download_link alone for civitai_api type
                # So we pass None and let manual_download try to determine it from API data or default.
                manual_download_civitai_api(url, file_name=file_name)
                download_successful = True
            except Exception as e:
                print(f"\n> Civitai API Download error for {link}: {e}")

        elif parsed_info['type'] == 'prefixed':
            prefix = parsed_info['prefix']
            url = parsed_info['url']
            dst_dir = parsed_info['dst_dir']
            filename = parsed_info['filename']

            if prefix == 'extension':
                extension_repo.append((url, filename))
                download_successful = True # Treated as "downloaded" for this context
                continue
            try:
                manual_download_generic(url, dst_dir, filename)
                download_successful = True
            except Exception as e:
                print(f"\n> Prefixed Download error for {link}: {e}")

        elif parsed_info['type'] == 'generic':
            url = parsed_info['url']
            dst_dir = parsed_info['dst_dir']
            filename = parsed_info['filename']

            if not dst_dir:
                # If no destination is specified for a generic URL, default to HOME or a general downloads folder
                dst_dir = str(HOME) # Or str(HOME / 'downloads') if you have one
                print(f"Warning: Destination directory not specified for generic URL: {url}. Defaulting to {dst_dir}.")
            try:
                manual_download_generic(url, dst_dir, filename)
                download_successful = True
            except Exception as e:
                print(f"\n> Generic Download error for {link}: {e}")
        else:
            print(f"Skipping malformed or unsupported download link: {link}")
            
        # If a download was attempted and failed, do not proceed to unpack zips for it
        # (Though _unpack_zips is called once at the end anyway, this is more for immediate feedback)
        # If download_successful is False, it means the link was skipped or failed.
        # This part of the loop processes each link. _unpack_zips runs once all links are processed.


    _unpack_zips()

def manual_download_civitai_api(url, file_name=None):
    """Handles Civitai API based downloads."""
    api = CivitAiAPI(civitai_token)
    if not (data := api.validate_download(url, file_name)):
        # If API validation fails, it's a significant issue for this type of URL.
        # We should not fall back to generic download here, as it implies the model page URL is invalid.
        raise ValueError(f"Civitai API validation failed for model URL: {url}")

    model_type, file_name = data.model_type, data.model_name
    clean_url, download_url = data.clean_url, data.download_url
    image_url, image_name = data.image_url, data.image_name

    # Determine destination directory based on model_type from Civitai API
    # This is a critical part where we map Civitai's model_type to our local directory
    dst_dir = None
    if model_type == 'Checkpoint':
        dst_dir = model_dir
    elif model_type == 'LORA':
        dst_dir = lora_dir
    elif model_type == 'TextualInversion':
        dst_dir = embed_dir
    elif model_type == 'VAE':
        dst_dir = vae_dir
    elif model_type == 'Controlnet': # Assuming Civitai returns this
        dst_dir = control_dir
    # Add other mappings as needed
    
    if not dst_dir:
        print(f"Warning: Could not map Civitai model type '{model_type}' to a known local directory. Defaulting to {model_dir}.")
        dst_dir = model_dir # Fallback if model type isn't explicitly mapped

    # Download preview images
    if image_url and image_name:
        m_download(f"{image_url} {dst_dir} {image_name}")

    format_output(clean_url, dst_dir, file_name, image_url, image_name)
    m_download(f"{download_url} {dst_dir} {file_name or ''}", log=True)


def manual_download_generic(url, dst_dir, file_name=None):
    """Handles generic HTTP/HTTPS downloads (including direct Civitai orchestration links)."""
    clean_url = url # URL is already cleaned by _process_download_link

    # If file_name is still None or an empty string, try to infer it.
    if not file_name:
        inferred_filename = _extract_filename(url)
        if inferred_filename:
            file_name = inferred_filename
        else:
            print(f"Warning: Could not infer filename for {url}. Proceeding without a specific name. `m_download` might infer it.")
            file_name = '' # Pass empty string to m_download to let it handle filename if not found

    format_output(clean_url, dst_dir, file_name, image_url=None, image_name=None)
    m_download(f"{url} {dst_dir} {file_name or ''}", log=True)


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

    unique_models = {}
    if num_selection:
        max_num = len(model_dict)
        for num in _parse_selection_numbers(num_selection, max_num):
            if 1 <= num <= max_num:
                name = list(model_dict.keys())[num - 1]
                selected.extend(model_dict[name])

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
    processed_entries = set()  # Store (tag, clean_url) to check uniqueness
    result_urls = []

    for line in lines:
        clean_line = line.strip().lower()

        # Update the current tag when detected
        for prefix, (_, short_tag) in PREFIX_MAP.items():
            if (f"# {prefix}".lower() in clean_line) or (short_tag and short_tag.lower() in clean_line):
                current_tag = prefix
                break

        if not current_tag:
            continue

        # Normalise the delimiters and process each URL
        normalized_line = re.sub(r'[\s,]+', ',', line.strip())
        for url_entry in normalized_line.split(','):
            # Check if url_entry itself is a valid URL before proceeding
            if not url_entry.strip().startswith('http'):
                continue
                
            # Extract explicit filename tag if present in url_entry
            explicit_filename = None
            match_tag = re.search(r'\[(.*?)\]', url_entry) # Use a new variable to avoid confusion
            if match_tag:
                explicit_filename = unquote(match_tag.group(1))
                # Remove the filename tag from the URL part for uniqueness check and actual download URL
                url_without_tag = re.sub(r'\[.*?\]', '', url_entry).strip()
            else:
                url_without_tag = url_entry.strip()

            clean_url_for_key = _clean_url(url_without_tag)
            
            entry_key = (current_tag, clean_url_for_key)   # Uniqueness is determined by a pair (tag, URL)

            if entry_key not in processed_entries:
                final_filename = explicit_filename
                if not final_filename: # If no explicit filename, try to infer
                    final_filename = _extract_filename(clean_url_for_key)

                formatted_url = f"{current_tag}:{clean_url_for_key}"
                if final_filename:
                    formatted_url += f"[{final_filename}]"

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
prefixed_urls = []
for p, u_list in zip(PREFIX_MAP.keys(), urls_sources): # Iterate over keys to get prefixes, and corresponding URL lists
    if u_list:
        for u_entry in u_list.replace(',', '').split():
            # If the URL already contains [filename], don't try to add it again.
            if '[' not in u_entry and ']' not in u_entry:
                clean_u = _clean_url(u_entry)
                filename = _extract_filename(clean_u)
                if filename:
                    prefixed_urls.append(f"{p}:{clean_u}[{filename}]")
                else:
                    prefixed_urls.append(f"{p}:{clean_u}")
            else:
                # If it already has a tag, just clean the URL part and keep the tag as is.
                # _process_download_link will handle extracting the filename from the tag.
                clean_u_part = re.sub(r'\[.*?\]', '', u_entry).strip()
                clean_u_part = _clean_url(clean_u_part)
                
                # Extract the filename from the tag into a variable first to avoid f-string syntax error
                match = re.search(r'\[(.*?)\]', u_entry)
                if match: # This should always be true if we are in this 'else' block
                    extracted_filename_from_tag = match.group(1)
                    prefixed_urls.append(f"{p}:{clean_u_part}[{extracted_filename_from_tag}]")
                else:
                    # Fallback in case the regex somehow fails (shouldn't happen given the outer 'else' condition)
                    prefixed_urls.append(f"{p}:{clean_u_part}")


line += ', ' + ', '.join(prefixed_urls + [process_file_downloads(file_urls, empowerment_output)])


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

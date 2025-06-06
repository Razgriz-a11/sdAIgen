# ~ download.py | by ANXETY (Remade for General Downloads) ~

from webui_utils import handle_setup_timer  # WEBUI
from CivitaiAPI import CivitAiAPI            # CivitAI API (Optional, remove if not needed)
from Manager import m_download               # Every Download
import json_utils as js                      # JSON

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
DEFAULT_DOWNLOAD_DIR = HOME / 'downloads' # New: Default download directory

# Ensure the default download directory exists
os.makedirs(DEFAULT_DOWNLOAD_DIR, exist_ok=True)

# These might need to be loaded from settings or defined
# For the purpose of this generalized script, we'll assume they exist or remove them if not strictly necessary.
# If these paths are dynamic, ensure they are properly initialized before use.
# For example, if you remove `CivitaiAPI`, then `civitai_token` isn't needed.
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

# Check and install dependencies (existing logic)
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

# Install VENV (existing logic)
current_ui = js.read(SETTINGS_PATH, 'WEBUI.current')
latest_ui = js.read(SETTINGS_PATH, 'WEBUI.latest')

# Determine whether to reinstall venv
venv_needs_reinstall = (
    not VENV.exists()
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
locals().update(settings) # This line will populate variables like model_dir, vae_dir etc.

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
    handle_setup_timer(WEBUI, start_timer)      # Setup timer (for timer-extensions)

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
# Assumes `latest_webui` and `latest_extensions` are defined (e.g., from settings.json)
# if latest_webui or latest_extensions: # Uncomment and ensure these variables are defined
#     action = 'WebUI and Extensions' if latest_webui and latest_extensions else ('WebUI' if latest_webui else 'Extensions')
#     print(f"⌚️ Update {action}...", end='')
#     with capture.capture_output():
#         ipySys('git config --global user.email "you@example.com"')
#         ipySys('git config --global user.name "Your Name"')

#         ## Update Webui
#         if latest_webui:
#             CD(WEBUI)
#             ipySys('git stash push --include-untracked')
#             ipySys('git pull --rebase')
#             ipySys('git stash pop')

#         ## Update extensions
#         if latest_extensions:
#             for entry in os.listdir(f"{WEBUI}/extensions"):
#                 dir_path = f"{WEBUI}/extensions/{entry}"
#                 if os.path.isdir(dir_path):
#                     subprocess.run(['git', 'reset', '--hard'], cwd=dir_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#                     subprocess.run(['git', 'pull'], cwd=dir_path, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#     print(f"\r✨ Update {action} Completed!")


# === FIXING EXTENSIONS ===
# Assumes WEBUI is defined
# with capture.capture_output(): # Uncomment if needed
#     ipySys("sed -i '521s/open=\\(False\\|True\\)/open=False/' {WEBUI}/extensions/Umi-AI-Wildcards/scripts/wildcard_recursive.py")


## Version switching
# Assumes `commit_hash` is defined
# if commit_hash: # Uncomment if needed
#     print('🔄 Switching to the specified version...', end='')
#     with capture.capture_output():
#         CD(WEBUI)
#         ipySys('git config --global user.email "you@example.com"')
#         ipySys('git config --global user.name "Your Name"')
#         ipySys('git reset --hard {commit_hash}')
#         ipySys('git pull origin {commit_hash}')
#     print(f"\r🔄 Switch complete! Current commit: {COL.B}{commit_hash}{COL.X}")


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
                # Ensure the local_dir exists before trying to join with 'GDrive'
                local_target_dir = Path(cfg['local_dir'])
                local_target_dir.mkdir(parents=True, exist_ok=True) # Ensure parent directories exist
                src = local_target_dir / 'GDrive' # This should likely be the actual local directory, not a subfolder named 'GDrive'
                dst = Path(GD_BASE) / cfg['gdrive_subpath'] # This is the GDrive destination

                # The original script seems to create a symlink from a 'GDrive' *subfolder* within local_dir
                # to the actual GDrive path. This might not be the most intuitive for a generic download.
                # If the goal is to make the *local_dir* itself a symlink to GDrive,
                # then `src` should be `cfg['local_dir']` and `dst` should be `os.path.join(GD_BASE, cfg['gdrive_subpath'])`
                # For now, let's stick to the original script's symlink creation logic, but clarify the paths.
                create_symlink(str(src), str(dst), log) # Convert Path objects to strings for os.path operations

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
                    link_path = Path(cfg['local_dir']) / 'GDrive' # Adjust based on how symlink was created
                    if os.path.islink(link_path):
                        os.unlink(link_path)

                print("🗑️ Symlinks removed successfully!")

            except Exception as e:
                print(f"❌ Unmount error: {str(e)}\n")

handle_gdrive(mountGDrive)

# Get XL or 1.5 models list
# These variables (model_dir, vae_dir, lora_dir, etc.) are assumed to be loaded from settings.json
# via `locals().update(settings)` earlier in the script.
# If they are not defined, you'll need to define them or ensure settings.json is properly configured.
try:
    model_files = '_xl-models-data.py' if XL_models else '_models-data.py'
    with open(f"{SCRIPTS}/{model_files}") as f:
        exec(f.read())
except NameError:
    print(f"{COL.R}Warning: XL_models not defined. Proceeding without model data.{COL.X}")
    # Define dummy variables to prevent errors if not present in settings.json
    model_list = {}
    vae_list = {}
    controlnet_list = {}
    # Define these if they aren't coming from settings.json and are needed for PREFIX_MAP
    model_dir = HOME / 'models'
    vae_dir = HOME / 'vae'
    lora_dir = HOME / 'lora'
    embed_dir = HOME / 'embeddings'
    extension_dir = HOME / 'extensions'
    adetailer_dir = HOME / 'adetailer'
    control_dir = HOME / 'controlnet'
    upscale_dir = HOME / 'upscale'
    clip_dir = HOME / 'clip'
    unet_dir = HOME / 'unet'
    vision_dir = HOME / 'vision'
    encoder_dir = HOME / 'encoder'
    diffusion_dir = HOME / 'diffusion'
    config_dir = HOME / 'config'


# Ensure all predefined directories exist
for d in [model_dir, vae_dir, lora_dir, embed_dir, extension_dir,
          adetailer_dir, control_dir, upscale_dir, clip_dir,
          unet_dir, vision_dir, encoder_dir, diffusion_dir, config_dir]:
    os.makedirs(d, exist_ok=True)


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

def _extract_filename(url_or_path):
    """
    Extracts filename from a URL or a path string.
    Handles filenames specified in square brackets `[filename.ext]` in the URL string,
    otherwise, it extracts from the URL path.
    """
    if match := re.search(r'\[(.*?)\]', url_or_path):
        return match.group(1)

    # If it's a Civitai or Google Drive URL, filename might be determined by API or default
    if any(d in urlparse(url_or_path).netloc for d in ["civitai.com", "drive.google.com"]):
        return None # Let manual_download handle this via CivitaiAPI or gdown

    # Fallback to extracting from URL path
    return Path(urlparse(url_or_path).path).name


def _unpack_zips():
    """Recursively extract and delete all .zip files in PREFIX_MAP directories."""
    for dir_path, _ in PREFIX_MAP.values():
        for zip_file in Path(dir_path).rglob('*.zip'):
            print(f"🗜️ Unpacking {zip_file.name}...")
            try:
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    # Extract to a subdirectory with the same name as the zip file (without .zip)
                    extract_path = zip_file.with_suffix('')
                    os.makedirs(extract_path, exist_ok=True)
                    zf.extractall(extract_path)
                zip_file.unlink()
                print(f"✅ Unpacked and deleted {zip_file.name}")
            except zipfile.BadZipFile:
                print(f"{COL.R}Error: {zip_file.name} is a bad zip file. Skipping.{COL.X}")
            except Exception as e:
                print(f"{COL.R}Error unpacking {zip_file.name}: {e}{COL.X}")


# Download Core

def _process_download_link(link_entry):
    """
    Processes a download link entry, parsing prefix, URL, filename, and destination directory.
    Format:
        - "prefix:url[filename.ext]"
        - "prefix:url"
        - "url /path/to/destination/filename.ext" (new)
        - "url /path/to/destination/" (new)
        - "url" (new, downloads to DEFAULT_DOWNLOAD_DIR)
    """
    link_entry = _clean_url(link_entry.strip())
    parts = shlex.split(link_entry) # Robustly split by space, handling quotes

    if len(parts) == 1:
        # Case: "url" or "prefix:url" or "prefix:url[filename]"
        full_url_string = parts[0]
        if ':' in full_url_string and full_url_string.split(':', 1)[0] in PREFIX_MAP:
            prefix, remainder = full_url_string.split(':', 1)
            url = re.sub(r'\[.*?\]', '', remainder) # Remove filename in brackets for the actual URL
            filename = _extract_filename(full_url_string) # Extract filename from the original string
            dst_dir, _ = PREFIX_MAP[prefix]
            return prefix, url, str(dst_dir), filename
        else:
            # Generic URL without prefix or specified directory
            url = full_url_string
            filename = _extract_filename(url)
            return None, url, str(DEFAULT_DOWNLOAD_DIR), filename
    elif len(parts) == 2:
        # Case: "url /path/to/destination" or "url /path/to/destination/filename.ext"
        url = parts[0]
        dest_path_str = parts[1]
        if os.path.isdir(dest_path_str) or not Path(dest_path_str).suffix: # Check if it's a directory path
            dst_dir = dest_path_str
            filename = _extract_filename(url) # Filename comes from the URL
        else: # Assumed to be "url filepath/filename.ext"
            dst_dir = str(Path(dest_path_str).parent)
            filename = Path(dest_path_str).name
        return None, url, dst_dir, filename
    elif len(parts) == 3:
        # Original format: "url dst_dir file_name"
        url = parts[0]
        dst_dir = parts[1]
        filename = parts[2]
        return None, url, dst_dir, filename

    return None, None, None, None # Invalid format


def download(line):
    """Downloads files from comma-separated links, processes prefixes, and unpacks zips post-download."""
    urls_to_process = filter(None, map(str.strip, line.split(',')))
    
    # Process extensions separately as they are handled by git clone
    extensions_to_clone = [] 
    
    for link_entry in urls_to_process:
        prefix, url, dst_dir_str, filename = _process_download_link(link_entry)

        if not url:
            print(f"{COL.R}Skipping invalid download entry: {link_entry}{COL.X}")
            continue

        dst_dir = Path(dst_dir_str)
        os.makedirs(dst_dir, exist_ok=True) # Ensure destination directory exists

        if prefix == 'extension':
            extensions_to_clone.append((url, filename)) # Filename here acts as repo_name if provided
            continue
        try:
            manual_download(url, dst_dir, filename, prefix)
        except Exception as e:
            print(f"\n> Download error for {url}: {e}")

    _unpack_zips()

    # Process extensions after all other downloads are done
    if extensions_to_clone:
        print(f"✨ Installing custom {extension_type}...", end='')
        with capture.capture_output():
            for repo_url, repo_name in extensions_to_clone:
                _clone_repository(repo_url, repo_name, extension_dir)
        print(f"\r📦 Installed '{len(extensions_to_clone)}' custom {extension_type}!")


def manual_download(url, dst_dir, file_name=None, prefix=None):
    clean_url = url
    image_url, image_name = None, None

    if 'civitai' in url:
        # Assumes civitai_token is defined (e.g., from settings.json)
        # If CivitaiAPI is not needed, this block can be removed.
        try:
            api = CivitAiAPI(civitai_token)
            if not (data := api.validate_download(url, file_name)):
                print(f"{COL.Y}Civitai API failed to validate or retrieve data for {url}. Attempting direct download.{COL.X}")
                # Fallback to direct download if Civitai API fails
                if not file_name:
                    file_name = Path(urlparse(url).path).name
                format_output(url, dst_dir, file_name)
                m_download(f"{url} {dst_dir} {file_name or ''}", log=True)
                return

            model_type, file_name = data.model_type, data.model_name
            clean_url, url = data.clean_url, data.download_url
            image_url, image_name = data.image_url, data.image_name

            # Download preview images
            if image_url and image_name:
                print(f"⬇️ Downloading preview image: {image_name}")
                m_download(f"{image_url} {dst_dir} {image_name}")
                print(f"✅ Downloaded preview: {image_name}")

        except NameError:
            print(f"{COL.R}CivitaiAPI or civitai_token not defined. Skipping Civitai-specific handling for {url}.{COL.X}")
            if not file_name:
                file_name = Path(urlparse(url).path).name
        except Exception as e:
            print(f"{COL.R}Error with Civitai API for {url}: {e}. Attempting direct download.{COL.X}")
            if not file_name:
                file_name = Path(urlparse(url).path).name

    elif any(s in url for s in ('github', 'huggingface.co')):
        # Ensure file_name has an extension if it's derived from these domains
        if file_name and '.' not in file_name:
            # Try to get extension from the URL if not present in file_name
            parsed_url = urlparse(clean_url)
            if '.' in parsed_url.path:
                file_name += f".{parsed_url.path.split('.')[-1]}"
            else: # If no extension in path, default to .bin or .safetensors if common for models
                if prefix in ['model', 'lora', 'vae']:
                    file_name += ".safetensors"
                else:
                    file_name += ".bin" # Generic binary fallback

    # If file_name is still None at this point (e.g., for Google Drive or generic URLs)
    if not file_name:
        file_name = Path(urlparse(url).path).name
        if not file_name: # If still no name from URL path, create a generic one
            file_name = f"downloaded_file_{int(time.time())}"
            print(f"{COL.Y}Warning: Could not determine filename from URL. Using {file_name}{COL.X}")

    # Formatted info output
    format_output(clean_url, dst_dir, file_name, image_url, image_name)

    # Downloading using m_download. Ensure m_download is robust for all URL types.
    # The original m_download uses aria2, gdown, etc. based on URL.
    try:
        m_download(f"{url} {dst_dir} {file_name}", log=True)
    except Exception as e:
        print(f"{COL.R}Fatal download error for {url}: {e}{COL.X}")


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
# These variables (model, model_num, vae, vae_num, controlnet, controlnet_num, etc.)
# need to be defined (e.g., as inputs to the script, or read from settings).
# For testing purposes, you might define some dummy values or set them to empty strings.
# Example:
# model = ""
# model_num = ""
# vae = ""
# vae_num = ""
# controlnet = ""
# controlnet_num = ""

# Assuming model, model_num, vae, vae_num, controlnet, controlnet_num are defined.
# If these are meant to be user inputs, you'd prompt for them or pass them as arguments.
# For a general download script, these might be less relevant unless you're still
# catering to the specific AI model download use case.
try:
    line = handle_submodels(model, model_num, model_list, model_dir, line)
    line = handle_submodels(vae, vae_num, vae_list, vae_dir, line)
    line = handle_submodels(controlnet, controlnet_num, controlnet_list, control_dir, line)
except NameError as e:
    print(f"{COL.Y}Warning: Submodel handling skipped due to undefined variables: {e}. If not intended, define them.{COL.X}")


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

        # If a tag is found, process subsequent URLs with that tag
        if current_tag:
            # Normalise the delimiters and process each URL
            normalized_line = re.sub(r'[\s,]+', ',', line.strip())
            for url_entry in normalized_line.split(','):
                url_candidate = url_entry.split('#')[0].strip()
                if not url_candidate.startswith('http'):
                    continue

                clean_url = re.sub(r'\[.*?\]', '', url_candidate)
                entry_key = (current_tag, clean_url)  # Uniqueness is determined by a pair (tag, URL)

                if entry_key not in processed_entries:
                    filename = _extract_filename(url_candidate)
                    formatted_url = f"{current_tag}:{clean_url}"
                    if filename:
                        formatted_url += f"[{filename}]"

                    result_urls.append(formatted_url)
                    processed_entries.add(entry_key)
        else: # Handle generic URLs from file without a preceding tag
            normalized_line = re.sub(r'[\s,]+', ',', line.strip())
            for url_entry in normalized_line.split(','):
                url_candidate = url_entry.split('#')[0].strip()
                if not url_candidate.startswith('http'):
                    continue
                
                # Check for uniqueness of generic URLs
                if (None, url_candidate) not in processed_entries:
                    result_urls.append(url_candidate) # Just the URL, will be handled by generic download
                    processed_entries.add((None, url_candidate)) # Mark as processed generic URL

    return ', '.join(result_urls) if result_urls else ''


def process_file_downloads(file_urls, additional_lines=None):
    """Reads URLs from files/HTTP sources."""
    lines = []

    if additional_lines:
        lines.extend(additional_lines.splitlines())

    for source in file_urls:
        if source.startswith('http'):
            try:
                print(f"🌐 Fetching URLs from: {source}")
                response = requests.get(_clean_url(source))
                response.raise_for_status()
                lines.extend(response.text.splitlines())
            except requests.RequestException as e:
                print(f"{COL.R}Error fetching URLs from {source}: {e}{COL.X}")
                continue
        else:
            try:
                print(f"📂 Reading URLs from local file: {source}")
                with open(source, 'r', encoding='utf-8') as f:
                    lines.extend(f.readlines())
            except FileNotFoundError:
                print(f"{COL.R}File not found: {source}{COL.X}")
                continue
    return _process_lines(lines)

# File URLs processing (assuming these are defined as inputs or from settings)
# For a generalized script, you might simplify how these are obtained.
# Example definition if not coming from settings:
# Model_url = ""
# Vae_url = ""
# LoRA_url = ""
# Embedding_url = ""
# Extensions_url = ""
# ADetailer_url = ""
# custom_file_urls = ""
# empowerment_output = ""

urls_sources_from_settings = []
try:
    urls_sources_from_settings = (Model_url, Vae_url, LoRA_url, Embedding_url, Extensions_url, ADetailer_url)
except NameError:
    print(f"{COL.Y}Warning: Model/VAE/LoRA URLs not defined in settings. Skipping.{COL.X}")

file_urls_from_custom = []
try:
    file_urls_from_custom = [f"{f}.txt" if not f.endswith('.txt') else f for f in custom_file_urls.replace(',', '').split()] if custom_file_urls else []
except NameError:
    print(f"{COL.Y}Warning: custom_file_urls not defined. Skipping.{COL.X}")


# p -> prefix ; u -> url | Remember: don't touch the prefix!
prefixed_urls = []
for i, u_source in enumerate(urls_sources_from_settings):
    if u_source:
        prefix_key = list(PREFIX_MAP.keys())[i] # This assumes order matches PREFIX_MAP, be careful.
        for u in u_source.replace(',', '').split():
            prefixed_urls.append(f"{prefix_key}:{u}")


# Combine all lines for download
full_line_for_download = line # From handle_submodels
if prefixed_urls:
    if full_line_for_download:
        full_line_for_download += ', '
    full_line_for_download += ', '.join(prefixed_urls)

try:
    processed_files_content = process_file_downloads(file_urls_from_custom, empowerment_output)
    if processed_files_content:
        if full_line_for_download:
            full_line_for_download += ', '
        full_line_for_download += processed_files_content
except NameError as e:
    print(f"{COL.Y}Warning: Error processing file downloads: {e}. Ensure custom_file_urls and empowerment_output are defined.{COL.X}")

if detailed_download == 'on': # Assuming `detailed_download` is defined (e.g., from settings.json)
    print(f"\n\n{COL.Y}# ====== Detailed Download ====== #\n{COL.X}")
    download(full_line_for_download)
    print(f"\n{COL.Y}# =============================== #\n{COL.X}")
else:
    with capture.capture_output():
        download(full_line_for_download)

print('\r🏁 Download Complete!' + ' '*15)


## Install of Custom extensions
def _clone_repository(repo, repo_name, extension_dir):
    """Clones the repository to the specified directory."""
    final_repo_name = repo_name or repo.split('/')[-1].replace('.git', '')
    target_path = Path(extension_dir) / final_repo_name
    if target_path.exists():
        print(f"Repository already exists at {target_path}. Skipping clone.")
        return

    print(f"Cloning {repo} to {target_path}...")
    try:
        # Use subprocess.run directly for better control and error handling
        subprocess.run(['git', 'clone', '--depth', '1', '--recursive', repo, str(target_path)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Optional: fetch to get all branches/tags if needed, but depth 1 usually means only master
        subprocess.run(['git', 'fetch'], cwd=target_path, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Successfully cloned {repo_name or repo}")
    except subprocess.CalledProcessError as e:
        print(f"{COL.R}Error cloning {repo}: {e.stderr}{COL.X}")
    except Exception as e:
        print(f"{COL.R}An unexpected error occurred during cloning {repo}: {e}{COL.X}")


extension_type = 'nodes' if UI == 'ComfyUI' else 'extensions'

# The `extension_repo` is now populated within the `download` function,
# so we don't need a separate check here. The `download` function
# now calls _clone_repository for extensions directly.


# === SPECIAL ===
## Sorting models `bbox` and `segm` | Only ComfyUI
if UI == 'ComfyUI':
    dirs = {'segm': '-seg.pt', 'bbox': None}
    for d_name in dirs:
        os.makedirs(os.path.join(adetailer_dir, d_name), exist_ok=True)

    for filename in os.listdir(adetailer_dir):
        src = os.path.join(adetailer_dir, filename)

        if os.path.isfile(src) and filename.endswith('.pt'):
            dest_dir_name = 'segm' if filename.endswith('-seg.pt') else 'bbox'
            dest = os.path.join(adetailer_dir, dest_dir_name, filename)

            if os.path.exists(dest):
                print(f"🗑️ Removing duplicate: {src}")
                os.remove(src)
            else:
                print(f"➡️ Moving {filename} to {dest_dir_name}")
                shutil.move(src, dest)


## List Models and stuff
ipyRun('run', f"{SCRIPTS}/download-result.py")

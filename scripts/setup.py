# ~ setup.py | by ANXETY ~

from IPython.display import display, HTML, clear_output
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from tqdm import tqdm
import nest_asyncio
import importlib
import argparse
import aiohttp
import asyncio
import time
import json
import sys
import os
import subprocess # Added for running shell commands

nest_asyncio.apply()  # Async support for Jupyter


# ======================== CONSTANTS =======================
HOME = Path.home()
SCR_PATH = HOME / 'ANXETY'
SETTINGS_PATH = SCR_PATH / 'settings.json'
VENV_PATH = HOME / 'venv'
MODULES_FOLDER = SCR_PATH / "modules"

# Add paths to the environment
os.environ.update({
    'home_path': str(HOME),
    'scr_path': str(SCR_PATH),
    'venv_path': str(VENV_PATH),
    'settings_path': str(SETTINGS_PATH)
})

# GitHub configuration
DEFAULT_USER = 'anxety-solo'
DEFAULT_REPO = 'sdAIgen'
DEFAULT_BRANCH = 'main'
DEFAULT_LANG = 'en'
BASE_GITHUB_URL = "https://raw.githubusercontent.com"

# Environment detection
SUPPORTED_ENVS = {
    'COLAB_GPU': 'Google Colab',
    'KAGGLE_URL_BASE': 'Kaggle'
}

# File structure configuration
FILE_STRUCTURE = {
    'CSS': ['main-widgets.css', 'download-result.css', 'auto-cleaner.css'],
    'JS': ['main-widgets.js'],
    'modules': [
        'json_utils.py', 'webui_utils.py', 'widget_factory.py',
        'CivitaiAPI.py', 'Manager.py', 'TunnelHub.py', '_season.py'
    ],
    'scripts': {
        'UIs': ['A1111.py', 'ComfyUI.py', 'Forge.py', 'Classic.py', 'ReForge.py', 'SD-UX.py'],
        '{lang}': ['widgets-{lang}.py', 'downloading-{lang}.py'],
        '': [
            'launch.py', 'download-result.py', 'auto-cleaner.py',
            '_models-data.py', '_xl-models-data.py'
        ]
    }
}


# =================== UTILITY FUNCTIONS ====================

def install_external_dependencies():
    """
    Checks for and installs required external tools (aria2c, gdown).
    This is crucial for environments like Colab/Kaggle where these might not be
    pre-installed, ensuring the script's core functionality.
    """
    print("Checking for required external tools (aria2c, gdown)...")
    try:
        # Check for aria2c and install if not found
        # Using 'which' to check if the command exists in PATH
        if subprocess.run(['which', 'aria2c'], capture_output=True).returncode != 0:
            print("aria2c not found. Attempting to install aria2...")
            # Update apt-get first to ensure latest package info
            subprocess.run(['apt-get', 'update', '-y'], check=True, capture_output=True)
            # Install aria2
            subprocess.run(['apt-get', 'install', '-y', 'aria2'], check=True, capture_output=True)
            print("aria2c installed successfully.")
        else:
            print("aria2c already installed.")

        # Check for gdown and install if not found
        if subprocess.run(['which', 'gdown'], capture_output=True).returncode != 0:
            print("gdown not found. Attempting to install gdown via pip...")
            # Use sys.executable to ensure pip associated with the current Python environment is used
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'gdown'], check=True, capture_output=True)
            print("gdown installed successfully.")
        else:
            print("gdown already installed.")

    except subprocess.CalledProcessError as e:
        print(f"Error during dependency installation: {e}")
        print(f"Command failed: {' '.join(e.cmd)}")
        if e.stdout:
            print(f"Stdout: {e.stdout.decode().strip()}")
        if e.stderr:
            print(f"Stderr: {e.stderr.decode().strip()}")
        print("Please ensure you have necessary permissions or try installing manually.")
    except Exception as e:
        print(f"An unexpected error occurred during dependency check/installation: {e}")
    print("External tool check complete.")


def _install_deps() -> bool:
    """Check if all required dependencies are installed (aria2 and gdown)."""
    try:
        from shutil import which
        required_tools = ['aria2c', 'gdown']
        return all(which(tool) is not None for tool in required_tools)
    except ImportError:
        return False

def _get_start_timer() -> int:
    """Get start timer from settings or return current time minus 5 seconds."""
    try:
        if SETTINGS_PATH.exists():
            settings = json.loads(SETTINGS_PATH.read_text())
            return settings.get("ENVIRONMENT", {}).get("start_timer", int(time.time() - 5))
    except (json.JSONDecodeError, OSError):
        pass
    return int(time.time() - 5)

def save_env_to_json(data: dict, filepath: Path) -> None:
    """Save environment data to JSON file, merging with existing content."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Load existing data if file exists
    existing_data = {}
    if filepath.exists():
        try:
            existing_data = json.loads(filepath.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    # Merge new data with existing
    merged_data = {**existing_data, **data}
    filepath.write_text(json.dumps(merged_data, indent=4))


# =================== MODULE MANAGEMENT ====================

def _clear_module_cache(modules_folder = None):
    """Clear module cache for modules in specified folder or default modules folder."""
    target_folder = Path(modules_folder) if modules_folder else MODULES_FOLDER
    target_folder = target_folder.resolve()    # Full absolute path

    for module_name, module in list(sys.modules.items()):
        if hasattr(module, "__file__") and module.__file__:
            module_path = Path(module.__file__).resolve()
            try:
                # Check if the module's path is under the target folder
                if target_folder in module_path.parents:
                    del sys.modules[module_name]
            except (ValueError, RuntimeError):
                # Handle cases where module_path.parents might not be directly comparable
                # or module object is problematic during iteration
                continue

    importlib.invalidate_caches()

def setup_module_folder(modules_folder = None):
    """Set up module folder by clearing cache and adding to sys.path."""
    target_folder = Path(modules_folder) if modules_folder else MODULES_FOLDER
    target_folder.mkdir(parents=True, exist_ok=True)

    _clear_module_cache(target_folder)

    folder_str = str(target_folder)
    if folder_str not in sys.path:
        sys.path.insert(0, folder_str)


# =================== ENVIRONMENT SETUP ====================

def detect_environment():
    """Detect runtime environment."""
    for var, name in SUPPORTED_ENVS.items():
        if var in os.environ:
            return name
    raise EnvironmentError(f"Unsupported environment. Supported: {', '.join(SUPPORTED_ENVS.values())}")

def parse_fork_arg(fork_arg):
    """Parse fork argument into user/repo."""
    if not fork_arg:
        return DEFAULT_USER, DEFAULT_REPO
    parts = fork_arg.split("/", 1)
    return parts[0], (parts[1] if len(parts) > 1 else DEFAULT_REPO)

def create_environment_data(env, lang, fork_user, fork_repo, branch):
    """Create environment data dictionary."""
    # This now reflects if the dependencies are available AFTER attempts to install them
    install_deps_status = _install_deps()
    start_timer = _get_start_timer()

    return {
        "ENVIRONMENT": {
            "env_name": env,
            "install_deps": install_deps_status,
            "fork": f"{fork_user}/{fork_repo}",
            "branch": branch,
            "lang": lang,
            "home_path": os.environ['home_path'],
            "scr_path": os.environ['scr_path'],
            "venv_path": os.environ['venv_path'],
            "settings_path": os.environ['settings_path'],
            "start_timer": start_timer,
            "public_ip": ""
        }
    }


# ===================== DOWNLOAD LOGIC =====================

def _format_lang_path(path: str, lang: str) -> str:
    """Format path with language placeholder."""
    return path.format(lang=lang) if '{lang}' in path else path

def generate_file_list(structure: Dict, base_url: str, lang: str) -> List[Tuple[str, Path]]:
    """Generate flat list of (url, path) from nested structure."""
    def walk(struct: Dict, path_parts: List[str]) -> List[Tuple[str, Path]]:
        items = []
        for key, value in struct.items():
            # Handle language-specific paths
            current_key = _format_lang_path(key, lang)
            current_path = [*path_parts, current_key] if current_key else path_parts

            if isinstance(value, dict):
                items.extend(walk(value, current_path))
            else:
                url_path = "/".join(current_path)
                for file in value:
                    # Handle language-specific files
                    formatted_file = _format_lang_path(file, lang)
                    url = f"{base_url}/{url_path}/{formatted_file}" if url_path else f"{base_url}/{formatted_file}"
                    file_path = SCR_PATH / "/".join(current_path) / formatted_file
                    items.append((url, file_path))
        return items

    return walk(structure, [])

async def download_file(session: aiohttp.ClientSession, url: str, path: Path) -> Tuple[bool, str, Path, Optional[str]]:
    """Download and save single file with error handling."""
    try:
        async with session.get(url) as resp:
            resp.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(await resp.read())
            return (True, url, path, None)
    except aiohttp.ClientResponseError as e:
        return (False, url, path, f"HTTP error {e.status}: {e.message}")
    except Exception as e:
        return (False, url, path, f"Error: {str(e)}")

async def download_files_async(lang, fork_user, fork_repo, branch, log_errors):
    """Main download executor with error logging."""
    base_url = f"{BASE_GITHUB_URL}/{fork_user}/{fork_repo}/{branch}"
    file_list = generate_file_list(FILE_STRUCTURE, base_url, lang)

    async with aiohttp.ClientSession() as session:
        tasks = [download_file(session, url, path) for url, path in file_list]
        errors = []

        # Use tqdm for progress bar during file downloads
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                           desc="Downloading files", unit="file"):
            success, url, path, error = await future
            if not success:
                errors.append((url, path, error))

        # Clear the output after download progress, useful in notebooks
        clear_output()

        if log_errors and errors:
            print("\nErrors occurred during download:")
            for url, path, error in errors:
                print(f"URL: {url}\nPath: {path}\nError: {error}\n")

# ===================== MAIN EXECUTION =====================

async def main_async(args=None):
    """
    Entry point for the script.
    Handles environment detection, dependency installation,
    file downloading, and setup.
    """
    parser = argparse.ArgumentParser(description='ANXETY Download Manager')
    parser.add_argument('--lang', default=DEFAULT_LANG, help=f"Language to be used (default: {DEFAULT_LANG})")
    parser.add_argument('--branch', default=DEFAULT_BRANCH, help=f"Branch to download files from (default: {DEFAULT_BRANCH})")
    parser.add_argument('--fork', default=None, help="Specify project fork (user or user/repo)")
    parser.add_argument('-s', '--skip-download', action="store_true", help="Skip downloading files")
    parser.add_argument('-l', "--log", action="store_true", help="Enable logging of download errors")

    # parse_known_args allows the script to run even if extra arguments are passed
    # which is common in environments like Colab/Kaggle that might add their own args.
    args, _ = parser.parse_known_args(args)

    # 1. Install necessary external system dependencies (like aria2c)
    install_external_dependencies()

    # 2. Detect the current runtime environment
    env = detect_environment()
    user, repo = parse_fork_arg(args.fork)  # GitHub: user/repo

    # 3. Download script files from GitHub
    if not args.skip_download:
        await download_files_async(args.lang, user, repo, args.branch, args.log)

    # 4. Set up Python module paths and clear cache
    setup_module_folder()

    # 5. Create environment data and save to settings file
    env_data = create_environment_data(env, args.lang, user, repo, args.branch)
    save_env_to_json(env_data, SETTINGS_PATH)

    # 6. Display relevant information after setup is complete
    # Importing _season here to ensure it's available after module setup
    try:
        from _season import display_info
        display_info(
            env=env,
            scr_folder=os.environ['scr_path'],
            branch=args.branch,
            lang=args.lang,
            fork=args.fork
        )
    except ImportError as e:
        print(f"Could not import display_info from _season.py: {e}")
        print("Please ensure _season.py was downloaded successfully.")


if __name__ == "__main__":
    # Run the main asynchronous function
    asyncio.run(main_async())

from dynaconf import Dynaconf
from pathlib import Path

# Get the path to the parent directory
BASE_DIR = str(Path(__file__).parents[2]) # root dir of project : Car_Price_Prediction_ML/

conf = Dynaconf(
    # Set ENV_VAR_PREFIX to something project-specific or keep default DYNACONF
    envvar_prefix="MYAPP", # Now env vars like MYAPP_DATA__RAW_PATH will work

    # Define the paths to your conf files from the project root
    root_path=BASE_DIR,
    settings_files=['src/configs/settings.toml', 'src/configs/.secrets.toml'],

    environments=True,             # Enable environments like [default], [production]
    env_switcher_var="ENV_FOR_MYAPP", # Variable to switch environment (default: ENV_FOR_DYNACONF => export ENV_FOR_DYNACONF=production)
    load_dotenv=True,           # Load .env file if exists
)
# `conf` object holds all configuration settings
# You can now import it in other modules: from src.config import conf

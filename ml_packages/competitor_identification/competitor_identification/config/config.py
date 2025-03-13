import yaml
from pathlib import Path


class Config:
    def __init__(self, default_config_path=None, custom_config_path=None):
        # Determine the path to the default_config.yaml relative to this module's location
        if default_config_path is None:
            default_config_path = Path(__file__).resolve().parent / 'default_config.yaml'

        # Load the default configuration
        self.config = self.load_config(default_config_path)

        # If a custom configuration path is provided, load and update the configuration
        if custom_config_path:
            self.update_config(custom_config_path)

    def load_config(self, config_path):
        """Load the configuration file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise Exception(f"Configuration file {config_path} not found.")
        except yaml.YAMLError as e:
            raise Exception(f"Error parsing YAML file {config_path}: {e}")

    def update_config(self, config_path):
        """Update the configuration with values from a custom config file."""
        try:
            with open(config_path, 'r') as file:
                custom_config = yaml.safe_load(file)
            self.config.update(custom_config)
        except FileNotFoundError:
            raise Exception(f"Configuration file {config_path} not found.")
        except yaml.YAMLError as e:
            raise Exception(f"Error parsing YAML file {config_path}: {e}")

    def get(self, key, default=None):
        """Retrieve a value from the configuration."""
        return self.config.get(key, default)

    def set(self, key, value):
        """Set a value in the configuration."""
        self.config[key] = value
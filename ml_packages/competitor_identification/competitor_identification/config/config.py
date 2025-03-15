import yaml
from pathlib import Path


class Config:
    def __init__(self, default_config_path=None, custom_config_path=None):
        """
        Initializes the Config object by loading the default configuration.
        If a custom configuration file is provided, it updates the default configuration.

        Parameters:
        - default_config_path (str or Path, optional): Path to the default configuration file.
        - custom_config_path (str or Path, optional): Path to the custom configuration file.
        """
        try:
            # Determine the path to the default_config.yaml relative to this module's location
            if default_config_path is None:
                default_config_path = Path(__file__).resolve().parent / 'default_config.yaml'

            # Load the default configuration
            self.config = self.load_config(default_config_path)

            # If a custom configuration path is provided, load and update the configuration
            if custom_config_path:
                self.update_config(custom_config_path)
        except Exception as e:
            raise RuntimeError(f"Error initializing Config: {e}")

    def load_config(self, config_path):
        """
        Loads a YAML configuration file.

        Parameters:
        - config_path (str or Path): Path to the configuration file.

        Returns:
        - dict: The loaded configuration.
        """
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise Exception(f"Configuration file {config_path} not found.")
        except yaml.YAMLError as e:
            raise Exception(f"Error parsing YAML file {config_path}: {e}")

    def update_config(self, config_path):
        """
        Updates the existing configuration with values from a custom YAML file.

        Parameters:
        - config_path (str or Path): Path to the custom configuration file.
        """
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
from .config.config import Config


class CompetitorIdentification:
    def __init__(self, custom_config_path=None):
        """
        Initialize the CustomerChurnPredictor with configuration.

        Args:
        - custom_config_path (str): Path to the custom configuration file (optional).
        """
        # Initialize the Config class with the custom configuration (if provided)
        self.config = Config(custom_config_path=custom_config_path)

    def get_config(self):
        """
        Retrieve the configuration object.
        """
        return self.config
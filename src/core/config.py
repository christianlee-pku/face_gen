import os
import importlib.util

class Config:
    """
    Configuration class that supports reading from python files.
    Supports recursive attribute access for nested dictionaries.
    """
    def __init__(self, cfg_dict=None):
        if cfg_dict is None:
            cfg_dict = dict()
        self._cfg_dict = cfg_dict

    @staticmethod
    def fromfile(filename):
        """
        Load a config from a file.
        """
        filename = str(filename)
        if not filename.endswith('.py'):
             raise IOError('Only py config files are supported now.')
        
        if not os.path.isfile(filename):
             raise FileNotFoundError(f"Config file not found: {filename}")

        # Import the file as a module
        spec = importlib.util.spec_from_file_location("temp_config_module", filename)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        cfg_dict = {
            name: value
            for name, value in mod.__dict__.items()
            if not name.startswith('__')
        }
        
        return Config(cfg_dict)

    def __getattr__(self, name):
        value = self._cfg_dict.get(name)
        if isinstance(value, dict):
            return Config(value)
        return value

    def __getitem__(self, name):
        value = self._cfg_dict[name]
        if isinstance(value, dict):
            return Config(value)
        return value

    def __repr__(self):
        return f"Config(path={self._cfg_dict})"

    def get(self, key, default=None):
        value = self._cfg_dict.get(key, default)
        if isinstance(value, dict):
            return Config(value)
        return value
    
    @property
    def _cfg_dict_copy(self):
        return self._cfg_dict.copy()
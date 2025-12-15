class Registry:
    """
    A simple registry to map strings to classes/functions.
    Similar to MMEngine's Registry but lightweight.
    """
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + f'(name={self._name}, '
        format_str += f'items={list(self._module_dict.keys())})'
        return format_str

    def get(self, key):
        """Get the registered module by key."""
        return self._module_dict.get(key, None)

    def register_module(self, name=None, module=None, force=False):
        """Register a module."""
        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module

        # Use as a decorator
        def _register(module):
            self._register_module(module=module, module_name=name, force=force)
            return module

        return _register

    def _register_module(self, module, module_name=None, force=False):
        if not hasattr(module, "__name__"):
            raise TypeError(f"Module {module} must have a __name__ attribute")

        if module_name is None:
            module_name = module.__name__

        if not force and module_name in self._module_dict:
            raise KeyError(f"{module_name} is already registered in {self._name}")

        self._module_dict[module_name] = module

    def build(self, cfg, *args, **kwargs):
        """Build an instance from a config dict."""
        # Unpack Config object if necessary
        if hasattr(cfg, '_cfg_dict'):
            cfg = cfg._cfg_dict
            
        if not isinstance(cfg, dict):
            raise TypeError(f"cfg must be a dict, but got {type(cfg)}")
        if 'type' not in cfg:
            raise KeyError(f"cfg must contain the key 'type', but got {cfg}")

        args_cfg = cfg.copy()
        obj_type = args_cfg.pop('type')

        if isinstance(obj_type, str):
            obj_cls = self.get(obj_type)
            if obj_cls is None:
                raise KeyError(f"{obj_type} is not in the {self._name} registry")
        elif isinstance(obj_type, type):
            obj_cls = obj_type
        else:
            raise TypeError(f"type must be a str or type, but got {type(obj_type)}")

        try:
            return obj_cls(*args, **kwargs, **args_cfg)
        except Exception as e:
            raise type(e)(f"Failed to build {obj_type} from {cfg}: {e}") from e

# Initialize Registries
DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')
MODELS = Registry('model')
HOOKS = Registry('hook')
METRICS = Registry('metric')

import unittest
import tempfile
import os
from src.core.registry import Registry
from src.core.config import Config

class TestRegistry(unittest.TestCase):
    def test_registry_register_and_get(self):
        TEST_REGISTRY = Registry("test")
        
        @TEST_REGISTRY.register_module()
        class TestClass:
            pass
            
        self.assertEqual(TEST_REGISTRY.get("TestClass"), TestClass)
        
    def test_registry_build(self):
        TEST_REGISTRY = Registry("test_build")
        
        @TEST_REGISTRY.register_module()
        class TestBuildClass:
            def __init__(self, arg1):
                self.arg1 = arg1
                
        cfg = dict(type='TestBuildClass', arg1='value')
        obj = TEST_REGISTRY.build(cfg)
        
        self.assertIsInstance(obj, TestBuildClass)
        self.assertEqual(obj.arg1, 'value')

class TestConfig(unittest.TestCase):
    def test_config_loading(self):
        with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            f.write("key = 'value'\n")
            f.write("num = 123\n")
            fname = f.name
            
        try:
            cfg = Config.fromfile(fname)
            self.assertEqual(cfg.key, 'value')
            self.assertEqual(cfg.num, 123)
        finally:
            os.remove(fname)

if __name__ == '__main__':
    unittest.main()


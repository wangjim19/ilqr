import importlib
import pdb

def load_config(filepath):
	module = importlib.import_module(filepath)
	config = getattr(module, 'Config')
	return config
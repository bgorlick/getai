# getai/api/__init__.py - Initialization module for the getai.api package.

from getai.api.datasets import DatasetAPI
from getai.api.models import ModelAPI
from getai.api.utils import UtilsAPI

# Exposing class methods as module-level functions
search_datasets = DatasetAPI.search_datasets
download_dataset = DatasetAPI.download_dataset
search_models = ModelAPI.search_models
download_model = ModelAPI.download_model
get_hf_token = UtilsAPI.get_hf_token
hf_login = UtilsAPI.hf_login

__all__ = [
    "search_datasets",
    "download_dataset",
    "search_models",
    "download_model",
    "get_hf_token",
    "hf_login",
    "DatasetAPI",
    "ModelAPI",
    "UtilsAPI",
]

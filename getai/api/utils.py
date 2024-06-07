# getai/api/utils.py - This module contains utility functions for the API.

from functools import lru_cache
from typing import Optional
from getai.core.utils import CoreUtils


class UtilsAPI:
    @staticmethod
    @lru_cache(maxsize=None)
    def get_hf_token(token: Optional[str] = None) -> str:
        """Retrieve the Hugging Face token using caching for efficiency."""
        return token or CoreUtils.get_hf_token()

    @staticmethod
    def hf_login():
        """Log in using Hugging Face CLI."""
        CoreUtils.hf_login()


# Expose class methods as module-level functions
get_hf_token = UtilsAPI.get_hf_token
hf_login = UtilsAPI.hf_login

__all__ = [
    "UtilsAPI",
    "get_hf_token",
    "hf_login",
]

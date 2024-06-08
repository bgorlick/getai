# getai/api/models.py - GetAI API methods for searching and downloading models.

from functools import lru_cache
from typing import Optional
from pathlib import Path


class ModelAPI:

    @staticmethod
    @lru_cache(maxsize=None)
    def _get_hf_token(token: Optional[str] = None) -> str:
        from getai.core.utils import get_hf_token

        return token or get_hf_token()

    @staticmethod
    async def search_models(
        query: str,
        hf_token: Optional[str] = None,
        max_connections: int = 5,
        **kwargs,
    ):
        from getai.api.utils import get_hf_token
        from getai.core.model_search import AsyncModelSearch

        hf_token = hf_token or get_hf_token()

        searcher = AsyncModelSearch(
            query=query,
            max_connections=max_connections,
            hf_token=hf_token,
        )
        await searcher.search_models(query, **kwargs)

    @staticmethod
    async def download_model(
        identifier: str,
        branch: str = "main",
        hf_token: Optional[str] = None,
        max_connections: int = 5,
        output_dir: Optional[Path] = None,
        **kwargs,
    ):
        from getai.api.utils import get_hf_token
        from getai.core.model_downloader import AsyncModelDownloader

        hf_token = hf_token or get_hf_token()
        output_dir = output_dir or Path.home() / ".getai" / "models"

        downloader = AsyncModelDownloader(
            output_dir=output_dir,
            max_connections=max_connections,
            hf_token=hf_token,
        )
        await downloader.download_model(identifier, branch, **kwargs)

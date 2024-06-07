# getai/api/datasets.py - This module contains the API methods for searching and downloading datasets.

from pathlib import Path
from typing import Optional
from getai.api.utils import get_hf_token


class DatasetAPI:

    @staticmethod
    async def search_datasets(
        query: str,
        hf_token: Optional[str] = None,
        max_connections: int = 5,
        output_dir: Optional[Path] = None,
        **kwargs,
    ):
        from getai.core.dataset_search import AsyncDatasetSearch

        hf_token = hf_token or get_hf_token()
        output_dir = output_dir or Path.home() / ".getai" / "datasets" / query

        searcher = AsyncDatasetSearch(
            query=query,
            output_dir=output_dir,
            max_connections=max_connections,
            hf_token=hf_token,
        )
        await searcher.display_dataset_search_results()

    @staticmethod
    async def download_dataset(
        identifier: str,
        hf_token: Optional[str] = None,
        max_connections: int = 5,
        output_dir: Optional[Path] = None,
        **kwargs,
    ):
        from getai.core.dataset_downloader import AsyncDatasetDownloader

        hf_token = hf_token or get_hf_token()
        output_dir = output_dir or Path.home() / ".getai" / "datasets" / identifier

        downloader = AsyncDatasetDownloader(
            output_dir=output_dir,
            max_connections=max_connections,
            hf_token=hf_token,
        )
        await downloader.download_dataset_info(
            dataset_id=identifier,
            **{
                key: value
                for key, value in kwargs.items()
                if key in downloader.get_expected_kwargs()
            },
        )

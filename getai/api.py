"""
Helper README:

This module provides a simplified interface to interact with the getai tool,
allowing users to search and download datasets and models from Hugging Face.

Usage Examples:
---------------
1. Search for datasets with default settings:
    ```python
    from getai import search_datasets
    import asyncio

    async def main():
        await search_datasets("text classification")

    asyncio.run(main())
    ```

2. Download a specific dataset:
    ```python
    from getai import download_dataset
    import asyncio

    async def main():
        await download_dataset("dataset-id")

    asyncio.run(main())
    ```

3. Search for models with custom settings:
    ```python
    from getai import search_models
    import asyncio

    async def main():
        await search_models("bert", max_connections=10)

    asyncio.run(main())
    ```

4. Download a model specifying a branch:
    ```python
    from getai import download_model
    import asyncio

    async def main():
        await download_model("bert-base-uncased", branch="v1.0")

    asyncio.run(main())
    ```

Additional Configurations:
---------------------------
- max_connections: int (default: 5) - Maximum number of concurrent connections.
- output_dir: str - Directory to save downloaded files.
- hf_token: str - Hugging Face token for authentication.
"""

# api.py

# native libs first
from pathlib import Path
import logging

# relative import local libs
from .dataset_downloader import AsyncDatasetDownloader
from .dataset_search import AsyncDatasetSearch
from .model_downloader import AsyncModelDownloader
from .model_search import AsyncModelSearch
from .utils import get_hf_token, get_dataset_output_folder
from .session_manager import SessionManager


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def search_datasets(
    query, hf_token=None, max_connections=5, output_dir=None, **kwargs
):
    hf_token = hf_token or get_hf_token()
    output_dir = Path(output_dir) if output_dir else get_dataset_output_folder(query)

    async with SessionManager(
        max_connections=max_connections, hf_token=hf_token
    ) as manager:
        session = await manager.get_session()
        searcher = AsyncDatasetSearch(
            query=query,
            filtered_datasets=[],
            total_datasets=0,
            output_dir=output_dir,
            max_connections=max_connections,
            hf_token=hf_token,
            session=session,
        )
        await searcher.search_datasets(query, **kwargs)

        downloader = AsyncDatasetDownloader(
            session=session, max_connections=max_connections, output_dir=output_dir
        )
        for dataset in searcher.data["filtered_datasets"]:
            await downloader.download_dataset_info(dataset["id"], **kwargs)


async def download_dataset(
    identifier, hf_token=None, max_connections=5, output_dir=None, **kwargs
):
    hf_token = hf_token or get_hf_token()
    output_dir = (
        Path(output_dir) if output_dir else get_dataset_output_folder(identifier)
    )

    async with SessionManager(
        max_connections=max_connections, hf_token=hf_token
    ) as manager:
        session = await manager.get_session()
        downloader = AsyncDatasetDownloader(
            session=session, max_connections=max_connections, output_dir=output_dir
        )
        await downloader.download_dataset_info(identifier, **kwargs)


async def search_models(query, hf_token=None, max_connections=5, **kwargs):
    hf_token = hf_token or get_hf_token()

    async with SessionManager(
        max_connections=max_connections, hf_token=hf_token
    ) as manager:
        session = await manager.get_session()
        searcher = AsyncModelSearch(
            query=query,
            max_connections=max_connections,
            session=session,
            hf_token=hf_token,
        )
        await searcher.search_models(query, **kwargs)

        downloader = AsyncModelDownloader(
            session=session,
            max_connections=max_connections,
            output_dir=Path.home() / ".getai" / "models",
        )
        for model in searcher.filtered_models:
            await downloader.download_model(model["id"], branch="main", **kwargs)


async def download_model(
    identifier,
    branch="main",
    hf_token=None,
    max_connections=5,
    output_dir=None,
    **kwargs
):
    hf_token = hf_token or get_hf_token()
    output_dir = Path(output_dir) if output_dir else Path.home() / ".getai" / "models"

    async with SessionManager(
        max_connections=max_connections, hf_token=hf_token
    ) as manager:
        session = await manager.get_session()
        downloader = AsyncModelDownloader(
            session=session, max_connections=max_connections, output_dir=output_dir
        )
        await downloader.download_model(identifier, branch, **kwargs)

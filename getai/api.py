""" api.py for GetAI - Contains the core API functions for searching and downloading datasets and models. """

from pathlib import Path
import logging

from getai.utils import get_hf_token
from getai.session_manager import SessionManager

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define default directories
DEFAULT_MODEL_DIR = Path.home() / ".getai" / "models"
DEFAULT_DATASET_DIR = Path.home() / ".getai" / "datasets"


async def search_datasets(
    query, hf_token=None, max_connections=5, output_dir=None, **kwargs
):
    """Search datasets on Hugging Face based on a query."""
    hf_token = hf_token or get_hf_token()
    output_dir = Path(output_dir) if output_dir else DEFAULT_DATASET_DIR / query

    from getai.dataset_search import AsyncDatasetSearch
    from getai.dataset_downloader import AsyncDatasetDownloader

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
            await downloader.download_dataset_info(
                dataset["id"],
                **{
                    key: value
                    for key, value in kwargs.items()
                    if key in downloader.get_expected_kwargs()
                }
            )


async def download_dataset(
    identifier, hf_token=None, max_connections=5, output_dir=None, **kwargs
):
    """Download a dataset from Hugging Face by its identifier."""
    hf_token = hf_token or get_hf_token()
    output_dir = Path(output_dir) if output_dir else DEFAULT_DATASET_DIR / identifier

    from getai.dataset_downloader import AsyncDatasetDownloader

    async with SessionManager(
        max_connections=max_connections, hf_token=hf_token
    ) as manager:
        session = await manager.get_session()
        downloader = AsyncDatasetDownloader(
            session=session, max_connections=max_connections, output_dir=output_dir
        )
        await downloader.download_dataset_info(
            identifier,
            **{
                key: value
                for key, value in kwargs.items()
                if key in downloader.get_expected_kwargs()
            }
        )


async def search_models(query, hf_token=None, max_connections=5, **kwargs):
    """Search models on Hugging Face based on a query."""
    hf_token = hf_token or get_hf_token()

    from getai.model_search import AsyncModelSearch
    from getai.model_downloader import AsyncModelDownloader

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
            output_dir=DEFAULT_MODEL_DIR,
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
    """Download a model from Hugging Face by its identifier and branch."""
    hf_token = hf_token or get_hf_token()
    output_dir = Path(output_dir) if output_dir else DEFAULT_MODEL_DIR

    from getai.model_downloader import AsyncModelDownloader

    async with SessionManager(
        max_connections=max_connections, hf_token=hf_token
    ) as manager:
        session = await manager.get_session()
        downloader = AsyncModelDownloader(
            session=session, max_connections=max_connections, output_dir=output_dir
        )
        await downloader.download_model(identifier, branch, **kwargs)

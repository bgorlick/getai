# dataset_downloader.py

""" dataset_downloader.py - Asynchronous dataset downloader for Hugging Face datasets API. """

from pathlib import Path
import asyncio
import aiohttp
import aiofiles
from aiohttp import ClientSession
from rainbow_tqdm import tqdm
import logging

from .dataset_search import AsyncDatasetSearch
from .utils import get_dataset_output_folder

logging.basicConfig(
    format='%(name)s - %(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

BASE_URL = "https://huggingface.co"


class AsyncDatasetDownloader:

    def __init__(self, output_dir=None, max_connections=5, hf_token=None):
        self.output_dir = output_dir
        self.max_connections = max_connections
        self.hf_token = hf_token
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=None)
        
    async def __aenter__(self):
        self.session = ClientSession(headers={'Authorization': f'Bearer {self.hf_token}'})
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_session()

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def download_dataset_info(self, dataset_id, revision=None, full=False, output_folder=None):
        try:
            url = f"{BASE_URL}/api/datasets/{dataset_id}"
            if revision:
                url += f"/revision/{revision}"

            params = {"full": str(full).lower()}
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    dataset_info = await response.json()
                    logger.info(f"Dataset info for {dataset_id}:")
                    logger.info(dataset_info)
                    if output_folder is None:
                        output_folder = get_dataset_output_folder(dataset_id, self.output_dir)
                    await self.download_dataset_files(dataset_id, revision, output_folder)
                else:
                    logging.error(f"Error fetching dataset info: HTTP {response.status}")
        except Exception as e:
            logging.exception(f"Error downloading dataset info: {e}")

    async def download_dataset_files(self, dataset_id, revision=None, output_folder=None):
        try:
            url = f"{BASE_URL}/api/datasets/{dataset_id}/tree/{revision or 'main'}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    file_tree = await response.json()
                    output_folder.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Downloading dataset files to: {output_folder}")
                    tasks = []
                    for file in file_tree:
                        file_url = f"{BASE_URL}/datasets/{dataset_id}/resolve/{revision or 'main'}/{file['path']}"
                        tasks.append(self.download_dataset_file(file_url, output_folder / file['path']))
                    await asyncio.gather(*tasks)
                else:
                    logging.error(f"Error fetching dataset file tree: HTTP {response.status}")
        except Exception as e:
            logging.exception(f"Error downloading dataset files: {e}")
        
    async def download_dataset_file(self, url, file_path):
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    total_size = response.content_length or 0
                    output_path = Path(file_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    async with aiofiles.open(output_path, 'wb') as f:
                        progress_bar = tqdm(total=total_size, desc=output_path.name, unit='iB', unit_scale=True, ncols=100)
                        async for chunk in response.content.iter_chunked(1024):
                            await f.write(chunk)
                            progress_bar.update(len(chunk))
                        progress_bar.close()
                    logging.info(f"Downloaded dataset file: {output_path}")
                else:
                    logging.error(f"Error downloading dataset file: HTTP {response.status}")
        except Exception as e:
            logging.exception(f"Error downloading dataset file: {e}")

    async def get_dataset_tags(self):
        try:
            url = f"{BASE_URL}/api/datasets-tags-by-type"
            async with self.session.get(url) as response:
                if response.status == 200:
                    tags = await response.json()
                    logger.info("Available dataset tags:")
                    logger.info(tags)
                else:
                    logging.error(f"Error fetching dataset tags: HTTP {response.status}")
        except Exception as e:
            logging.exception(f"Error fetching dataset tags: {e}")

    async def search_datasets(self, query, author=None, filter=None, sort=None, direction=None, limit=None, full=False):
        try:
            url = f"{BASE_URL}/api/datasets"
            params = {
                "search": query,
                "author": author,
                "filter": filter,
                "sort": sort,
                "direction": direction,
                "limit": str(limit) if limit is not None else None,
                "full": str(full).lower()
            }
            params = {k: v for k, v in params.items() if v is not None}  
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    search_results = await response.json()
                    total_datasets = len(search_results)
                    if total_datasets > 0:
                        async with AsyncDatasetSearch(
                            query, search_results, total_datasets,
                            self.output_dir, self.max_connections, self.hf_token
                        ) as search:
                            await search.get_dataset_page_size()
                            await search.display_dataset_search_results(self)
                    else:
                        logger.info(f"No datasets found for the search query '{query}'.")
                else:
                    logging.error(f"Error searching for datasets: HTTP {response.status}")
        except Exception as e:
            logging.exception(f"Error searching for datasets: {e}")

""" dataset_downloader.py - GetAI Asynchronous Dataset Downloader. """

import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import hashlib
import asyncio
import aiohttp
import aiofiles
from aiohttp import ClientSession
from rainbow_tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

logging.basicConfig(
    format="%(name)s - %(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://huggingface.co"


class AsyncDatasetDownloader:
    """Asynchronous downloader for Hugging Face datasets."""

    def __init__(
        self,
        session: ClientSession,
        output_dir: Optional[Path] = None,
        max_connections: int = 5,
        hf_token: Optional[str] = None,
    ):
        """Initialize the downloader with settings."""
        self.session = session
        self.output_dir: Path = (
            output_dir if output_dir else Path.home() / ".getai" / "datasets"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_connections: int = max_connections
        self.hf_token: str = hf_token if hf_token else ""

    async def download_dataset_info(
        self,
        dataset_id: str,
        revision: Optional[str] = None,
        full: bool = False,
        output_folder: Optional[Path] = None,
    ):
        """Download dataset info from Hugging Face."""
        if not self.session:
            raise RuntimeError("Session is not initialized")

        try:
            url = f"{BASE_URL}/api/datasets/{dataset_id}"
            if revision:
                url += f"/revision/{revision}"

            params = {"full": str(full).lower()}
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    dataset_info = await response.json()
                    logger.info("Dataset info for %s:", dataset_id)
                    logger.info("%s", dataset_info)
                    output_folder = output_folder or self.output_dir / dataset_id
                    output_folder.mkdir(parents=True, exist_ok=True)
                    await self.download_dataset_files(
                        dataset_id, revision, output_folder, dataset_info
                    )
                    await self.validate_checksums_and_sizes(output_folder, dataset_info)
                else:
                    logger.error(
                        "Error fetching dataset info: HTTP %s", response.status
                    )
        except aiohttp.ClientError as e:
            logger.exception("Client error while downloading dataset info: %s", e)
        except asyncio.TimeoutError as e:
            logger.exception("Timeout error while downloading dataset info: %s", e)
        except Exception as e:
            logger.exception("Unexpected error while downloading dataset info: %s", e)

    async def download_dataset_files(
        self,
        dataset_id: str,
        revision: Optional[str] = None,
        output_folder: Optional[Path] = None,
        dataset_info: Optional[Dict[str, Any]] = None,
    ):
        """Download dataset files from Hugging Face."""
        if not self.session:
            raise RuntimeError("Session is not initialized")

        output_folder = output_folder or self.output_dir / dataset_id
        output_folder.mkdir(parents=True, exist_ok=True)

        try:
            url = f"{BASE_URL}/api/datasets/{dataset_id}/tree/{revision or 'main'}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    file_tree = await response.json()
                    logger.info("Downloading dataset files to: %s", output_folder)
                    tasks = []
                    lfs_files = []

                    for file in file_tree:
                        file_url = f"{BASE_URL}/datasets/{dataset_id}/resolve/{revision or 'main'}/{file['path']}"
                        if file["path"].endswith(".parquet"):
                            lfs_files.append(file)
                        else:
                            tasks.append(
                                self.download_dataset_file(
                                    file_url, output_folder / file["path"]
                                )
                            )
                    await asyncio.gather(*tasks)

                    if dataset_info and "siblings" in dataset_info:
                        logger.info("Detected LFS metadata, attempting LFS download.")
                        for sibling in dataset_info["siblings"]:
                            if sibling["rfilename"].endswith(".parquet"):
                                lfs_url = f"{BASE_URL}/datasets/{dataset_id}/resolve/{revision or 'main'}/{sibling['rfilename']}"
                                await self.download_git_lfs_file(
                                    lfs_url, output_folder / sibling["rfilename"]
                                )

                    metadata = await self.parse_json_files_for_metadata(output_folder)
                    for lfs_file in lfs_files:
                        lfs_metadata = metadata.get(lfs_file["path"])
                        if lfs_metadata:
                            lfs_url = f"https://huggingface.co/{dataset_id}/resolve/{revision or 'main'}/{lfs_file['path']}"
                            await self.download_file_from_url(
                                lfs_url, output_folder / lfs_file["path"]
                            )
                        else:
                            logger.error(
                                "Missing LFS metadata for %s", lfs_file["path"]
                            )

                    source_datasets = metadata.get("source_datasets", [])
                    for source_url in source_datasets:
                        await self.download_file_from_url(
                            source_url, output_folder / Path(source_url).name
                        )
                else:
                    logger.error(
                        "Error fetching dataset file tree: HTTP %s", response.status
                    )
        except aiohttp.ClientError as e:
            logger.exception("Client error while downloading dataset files: %s", e)
        except asyncio.TimeoutError as e:
            logger.exception("Timeout error while downloading dataset files: %s", e)
        except Exception as e:
            logger.exception("Unexpected error while downloading dataset files: %s", e)

    async def download_dataset_file(self, url: str, file_path: Path):
        """Download a single dataset file."""
        if not self.session:
            raise RuntimeError("Session is not initialized")

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    total_size = response.content_length or 0
                    output_path = Path(file_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    async with aiofiles.open(output_path, "wb") as f:
                        progress_bar = tqdm(
                            total=total_size,
                            desc=output_path.name,
                            unit="iB",
                            unit_scale=True,
                            ncols=100,
                        )
                        async for chunk in response.content.iter_chunked(1024):
                            await f.write(chunk)
                            progress_bar.update(len(chunk))
                        progress_bar.close()
                    logger.info("Downloaded dataset file: %s", output_path)
                elif response.status == 404:
                    logger.info("Attempting to use LFS for %s", url)
                    await self.download_git_lfs_file(url, file_path)
                else:
                    logger.error(
                        "Error downloading dataset file: HTTP %s", response.status
                    )
        except aiohttp.ClientError as e:
            logger.exception("Client error while downloading dataset file: %s", e)
        except asyncio.TimeoutError as e:
            logger.exception("Timeout error while downloading dataset file: %s", e)
        except Exception as e:
            logger.exception("Unexpected error while downloading dataset file: %s", e)

    @retry(
        stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def download_git_lfs_file(self, url: str, file_path: Path):
        """Download a Git LFS file."""
        if not self.session:
            raise RuntimeError("Session is not initialized")

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    size = response.content_length or 0

                    if size > 0:
                        await self.download_file_from_url(url, file_path)
                    else:
                        logger.error("Missing size in LFS metadata.")
                        await self.download_file_from_url(url, file_path)
                else:
                    logger.error(
                        "Error fetching Git LFS file info: HTTP %s", response.status
                    )
        except aiohttp.ClientError as e:
            logger.exception("Client error while downloading Git LFS file: %s", e)
        except asyncio.TimeoutError as e:
            logger.exception("Timeout error while downloading Git LFS file: %s", e)
        except Exception as e:
            logger.exception("Unexpected error while downloading Git LFS file: %s", e)

    @retry(
        stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def download_file_from_url(self, url: str, file_path: Path):
        """Download a file from a URL."""
        if not self.session:
            raise RuntimeError("Session is not initialized")

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    total_size = response.content_length or 0
                    output_path = Path(file_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    async with aiofiles.open(output_path, "wb") as f:
                        progress_bar = tqdm(
                            total=total_size,
                            desc=output_path.name,
                            unit="iB",
                            unit_scale=True,
                            ncols=100,
                        )
                        async for chunk in response.content.iter_chunked(1024):
                            await f.write(chunk)
                            progress_bar.update(len(chunk))
                        progress_bar.close()
                    logger.info("Downloaded file from URL: %s", output_path)
                else:
                    logger.error(
                        "Error downloading file from URL: HTTP %s", response.status
                    )
        except aiohttp.ClientError as e:
            logger.exception("Client error while downloading file from URL: %s", e)
        except asyncio.TimeoutError as e:
            logger.exception("Timeout error while downloading file from URL: %s", e)
        except Exception as e:
            logger.exception("Unexpected error while downloading file from URL: %s", e)

    async def parse_json_files_for_metadata(self, directory: Path) -> Dict[str, Any]:
        """Parse all JSON files in the directory for LFS metadata."""
        logger.info("Parsing JSON files in %s for metadata.", directory)
        metadata: Dict[str, Any] = {}

        for json_file in directory.glob("*.json"):
            try:
                async with aiofiles.open(json_file, "r") as f:
                    content = await f.read()
                    data = json.loads(content)
                    logger.info("Parsed JSON file %s: %s", json_file, data)

                    flat_data = self.flatten_json(data)

                    for key, value in flat_data.items():
                        if "splits" in key:
                            parts = key.split(".")
                            split_key = f"{parts[0]}-{parts[1]}"
                            file_info = {
                                "name": parts[1],
                                "num_bytes": flat_data.get(f"{key}.num_bytes"),
                                "num_examples": flat_data.get(f"{key}.num_examples"),
                                "dataset_name": flat_data.get(f"{key}.dataset_name"),
                            }
                            metadata[split_key] = file_info
                        elif (
                            "download_size" in key
                            or "dataset_size" in key
                            or "size_in_bytes" in key
                        ):
                            dataset_key = key.split(".")[0]
                            if dataset_key not in metadata:
                                metadata[dataset_key] = {}
                            metadata[dataset_key].update(
                                {
                                    "download_size": flat_data.get(
                                        f"{dataset_key}.download_size"
                                    ),
                                    "dataset_size": flat_data.get(
                                        f"{dataset_key}.dataset_size"
                                    ),
                                    "size_in_bytes": flat_data.get(
                                        f"{dataset_key}.size_in_bytes"
                                    ),
                                }
                            )
                        elif "source_datasets" in key:
                            if "source_datasets" not in metadata:
                                metadata["source_datasets"] = []
                            urls = value.split(",")
                            metadata["source_datasets"].extend(
                                url.strip() for url in urls
                            )

            except Exception as e:
                logger.error("Error parsing JSON file %s: %s", json_file, e)

        logger.info("Final parsed metadata:")
        for key, value in metadata.items():
            if isinstance(value, list):
                logger.info("Key: %s, Value: [List with %d items]", key, len(value))
                for i, item in enumerate(value):
                    logger.info("  List item %d: %s", i, item)
            elif isinstance(value, dict):
                logger.info("Key: %s, Value: [Dict with %d items]", key, len(value))
                for subkey, subvalue in value.items():
                    logger.info("  Dict key: %s, Dict value: %s", subkey, subvalue)
            else:
                logger.info("Key: %s, Value: %s", key, value)

        return metadata

    def flatten_json(
        self, json_data: Dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> Dict[str, Any]:
        """Flatten a nested JSON object."""
        items: List[Tuple[str, Any]] = []
        for k, v in json_data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_json(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for i, sub_item in enumerate(v):
                    if isinstance(sub_item, dict):
                        items.extend(
                            self.flatten_json(
                                sub_item, f"{new_key}[{i}]", sep=sep
                            ).items()
                        )
                    else:
                        items.append((f"{new_key}[{i}]", sub_item))
            else:
                items.append((new_key, v))
        return dict(items)

    async def validate_checksums_and_sizes(
        self, directory: Path, dataset_info: Dict[str, Any]
    ):
        """Validate checksums and sizes of downloaded files using dataset info."""
        if "siblings" not in dataset_info:
            logger.info(
                "No checksum data found in dataset metadata. Recommend users to manually verify file integrity."
            )
            return

        checksums = {
            sibling["rfilename"]: sibling.get("checksum")
            for sibling in dataset_info["siblings"]
        }
        sizes = {
            sibling["rfilename"]: {
                "dataset_size": sibling.get("dataset_size"),
                "size_in_bytes": sibling.get("size_in_bytes"),
            }
            for sibling in dataset_info["siblings"]
        }
        missing_checksums = [
            file for file, checksum in checksums.items() if checksum is None
        ]

        if missing_checksums:
            logger.warning(
                "Missing checksums for files: %s", ", ".join(missing_checksums)
            )
            logger.info("Recommend users to manually verify file integrity.")
        else:
            logger.info("All files have checksums available for validation.")

        for filename, expected_checksum in checksums.items():
            file_path = directory / filename
            if file_path.exists():
                try:
                    actual_checksum = await self.calculate_file_checksum(file_path)
                    if actual_checksum == expected_checksum:
                        logger.info("Checksum passed for %s", file_path)
                    else:
                        logger.warning(
                            "Checksum failed for %s: expected %s, got %s",
                            file_path,
                            expected_checksum,
                            actual_checksum,
                        )
                except FileNotFoundError as fnf_error:
                    logger.error(
                        "FileNotFoundError while validating checksum for %s: %s",
                        file_path,
                        fnf_error,
                    )
                except IOError as io_error:
                    logger.error(
                        "IOError while validating checksum for %s: %s",
                        file_path,
                        io_error,
                    )
            else:
                logger.warning("File not found for checksum validation: %s", file_path)

        for filename, size_info in sizes.items():
            file_path = directory / filename
            if file_path.exists():
                try:
                    actual_size = file_path.stat().st_size
                    if (
                        size_info["size_in_bytes"]
                        and actual_size == size_info["size_in_bytes"]
                    ):
                        logger.info(
                            "Size match for %s: %s bytes", file_path, actual_size
                        )
                    else:
                        logger.warning(
                            "Size mismatch for %s: expected %s bytes, got %s bytes",
                            file_path,
                            size_info["size_in_bytes"],
                            actual_size,
                        )
                except FileNotFoundError as fnf_error:
                    logger.error(
                        "FileNotFoundError while validating size for %s: %s",
                        file_path,
                        fnf_error,
                    )
                except IOError as io_error:
                    logger.error(
                        "IOError while validating size for %s: %s", file_path, io_error
                    )
            else:
                logger.warning("File not found for size validation: %s", file_path)

    async def calculate_file_checksum(
        self, file_path: Path, algorithm: str = "sha256"
    ) -> str:
        """Calculate the checksum of a file."""
        try:
            hasher = hashlib.new(algorithm)
            async with aiofiles.open(file_path, "rb") as f:
                while True:
                    data = await f.read(4096)
                    if not data:
                        break
                    hasher.update(data)
            return hasher.hexdigest()
        except FileNotFoundError as fnf_error:
            logger.error("FileNotFoundError: %s", fnf_error)
            raise
        except IOError as io_error:
            logger.error("IOError: %s", io_error)
            raise
        except Exception as ex:
            logger.error("Unexpected error in calculate_file_checksum: %s", ex)
            raise

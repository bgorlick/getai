# getai/core/model_downloader.py: Async Downloads models from the Hugging Face Hub.

import base64
import datetime
import hashlib
import logging
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import asyncio
from aiofiles import open as aio_open
import aiofiles


BASE_URL = "https://huggingface.co"
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

file_size_pattern = re.compile(
    r'<a class="[^"]*" title="Download file"[^>]*>([\d.]+ [GMK]B)'
)
search_cache: Dict[str, List[Dict]] = {}


class AsyncModelDownloader:
    """Downloads models using async methods."""

    def __init__(
        self,
        max_retries: int = 5,
        output_dir: Optional[Path] = None,
        max_connections: int = 7,
        hf_token: Optional[str] = None,
    ):
        """Initialize downloader with settings."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.output_dir = (
            output_dir if output_dir else Path.home() / ".getai" / "models"
        )
        self.max_retries = max_retries
        self.max_connections = max_connections
        self.token = hf_token
        self.branch_sizes: Dict[str, int] = {}
        self.file_locks: Dict[Path, asyncio.Lock] = {}

    async def download_model(
        self, model_id: str, branch: str, clean: bool = False, check: bool = False
    ) -> None:
        """Download and optionally verify a model."""
        print(f"Downloading model '{model_id}' from branch '{branch}'")

        from getai.core import SessionManager

        async with SessionManager(
            max_connections=self.max_connections, hf_token=self.token
        ) as manager:
            session = await manager.get_session()
            links, sha256, is_lora, is_llamacpp = (
                await self.get_download_links_from_huggingface(
                    session, model_id, branch
                )
            )
            output_folder = self.get_output_folder(
                model_id, branch, is_lora, is_llamacpp
            )
            await self.download_model_files(
                session, model_id, branch, links, dict(sha256), output_folder
            )
            if check:
                await self.check_model_files(
                    model_id, branch, links, dict(sha256), output_folder
                )

    async def get_download_links_from_huggingface(
        self,
        session,
        model: str,
        branch: str,
        text_only: bool = False,
        specific_file: Optional[str] = None,
    ) -> Tuple[List[str], List[Tuple[str, str]], bool, bool]:
        """Fetch model download links from Hugging Face."""
        page = f"/api/models/{model}/tree/{branch}"
        cursor = b""
        links: List[str] = []
        sha256: List[Tuple[str, str]] = []
        classifications: List[str] = []
        has_pytorch = False
        has_pt = False
        has_gguf = False
        has_safetensors = False
        is_lora = False

        while True:
            url = f"{BASE_URL}{page}" + (f"?cursor={cursor.decode()}" if cursor else "")
            self.logger.debug("Making request to: %s", url)

            async with session.get(url) as response:
                response.raise_for_status()
                content = await response.json()

                if not content:
                    break

                for i, item in enumerate(content):
                    fname = item.get("path", "")
                    if specific_file and fname != specific_file:
                        continue

                    if not is_lora and fname.endswith(
                        ("adapter_config.json", "adapter_model.bin")
                    ):
                        is_lora = True

                    is_pytorch = re.match(r"(pytorch|adapter|gptq)_model.*\.bin", fname)
                    is_safetensors = re.match(r".*\.safetensors", fname)
                    is_pt = re.match(r".*\.pt", fname)
                    is_gguf = re.match(r".*\.gguf", fname)
                    is_tiktoken = re.match(r".*\.tiktoken", fname)
                    is_tokenizer = (
                        re.match(r"(tokenizer|ice|spiece).*\.model", fname)
                        or is_tiktoken
                    )
                    is_text = re.match(r".*\.(txt|json|py|md)", fname) or is_tokenizer

                    if any(
                        (
                            is_pytorch,
                            is_safetensors,
                            is_pt,
                            is_gguf,
                            is_tokenizer,
                            is_text,
                        )
                    ):
                        if "lfs" in item:
                            sha256.append((fname, item["lfs"]["oid"]))

                        if is_text:
                            links.append(
                                f"https://huggingface.co/{model}/resolve/{branch}/{fname}"
                            )
                            classifications.append("text")
                            continue

                        if not text_only:
                            links.append(
                                f"https://huggingface.co/{model}/resolve/{branch}/{fname}"
                            )
                            if is_safetensors:
                                has_safetensors = True
                                classifications.append("safetensors")
                            elif is_pytorch:
                                has_pytorch = True
                                classifications.append("pytorch")
                            elif is_pt:
                                has_pt = True
                                classifications.append("pt")
                            elif is_gguf:
                                has_gguf = True
                                classifications.append("gguf")

                cursor = (
                    base64.b64encode(
                        f'{{"file_name":"{content[-1]["path"]}"}}'.encode()
                    )
                    + b":50"
                )
                cursor = base64.b64encode(cursor)
                cursor = cursor.replace(b"=", b"%3D")

        if (has_pytorch or has_pt) and has_safetensors:
            links = [
                link
                for link, classification in zip(links, classifications)
                if classification not in ("pytorch", "pt")
            ]

        if has_gguf and specific_file is None:
            has_q4km = any("q4_k_m" in link.lower() for link in links)
            if has_q4km:
                links = [link for link in links if "q4_k_m" in link.lower()]
            else:
                links = [link for link in links if not link.lower().endswith(".gguf")]

        is_llamacpp = has_gguf and specific_file is not None

        return links, sha256, is_lora, is_llamacpp

    def get_output_folder(
        self,
        identifier: str,
        branch: str,
        is_lora: bool = False,
        is_llamacpp: bool = False,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Determine folder for model storage."""
        base_folder = output_dir or Path("models")
        output_folder = f"{'_'.join(identifier.split('/')[-2:])}"
        if branch != "main":
            output_folder += f"_{branch}"
        if is_lora:
            output_folder += "_lora"
        elif is_llamacpp:
            output_folder += "_llamacpp"
        return base_folder / output_folder

    async def download_model_files(
        self,
        session,
        model: str,
        branch: str,
        links: List[str],
        sha256_dict: Dict[str, Optional[str]],
        output_folder: Path,
    ):
        """Download all model files."""
        output_folder.mkdir(parents=True, exist_ok=True)
        metadata = (
            f"url: https://huggingface.co/{model}\n"
            f"branch: {branch}\n"
            f'download date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
        )
        sha256_str = "\n".join(
            [f" {sha256} {filename}" for filename, sha256 in sha256_dict.items()]
        )
        if sha256_str:
            metadata += f"sha256sum:\n{sha256_str}"
        metadata += "\n"
        metadata_path = output_folder / "huggingface-metadata.txt"
        async with aiofiles.open(metadata_path, "w") as metafile:
            await metafile.write(metadata)

        semaphore = asyncio.Semaphore(self.max_connections)

        async def download_file(link: str):
            """Download a single file with improved semaphore management."""
            async with semaphore:
                filename = Path(link).name
                file_hash: Optional[str] = sha256_dict.get(filename)

                # Log the download start
                self.logger.info(f"Starting download for {filename}")

                if file_hash is None:
                    print(
                        f"Warning: No SHA256 hash found for {filename}. Downloading without sha256 verification."
                    )
                await self._download_model_file(session, link, output_folder, file_hash)

        tasks = [asyncio.ensure_future(download_file(link)) for link in links]
        # Ensure all tasks are completed before proceeding
        await asyncio.gather(*tasks)
        self.logger.info("All download tasks completed")

    async def _download_model_file(
        self,
        session,
        url: str,
        output_folder: Path,
        file_hash: Optional[str],
    ):
        """Download and save a model file with support for resuming downloads."""
        from rainbow_tqdm import tqdm

        filename = Path(url.rsplit("/", 1)[1])
        output_path = output_folder / filename

        # Check if the file already exists and get its size
        if output_path.exists():
            file_size = output_path.stat().st_size
        else:
            file_size = 0

        headers = {}
        if file_size > 0:
            headers["Range"] = f"bytes={file_size}-"
            self.logger.info(
                f"Resuming download for '{filename}' from byte {file_size}"
            )

        # Retry logic for downloading the file
        max_attempts = self.max_retries
        attempt = 0
        while attempt < max_attempts:
            try:
                async with session.get(url, headers=headers) as response:
                    if response.status == 416:
                        # Handle the case where the file is already completely downloaded
                        self.logger.info("'%s' is already fully downloaded.", filename)
                        return

                    response.raise_for_status()
                    total_size = response.content_length or 0
                    # If resuming, adjust total_size to reflect the remaining bytes
                    if "Content-Range" in response.headers:
                        content_range = response.headers["Content-Range"]
                        total_size = int(content_range.split("/")[1]) - file_size

                    mode = "ab" if file_size > 0 else "wb"
                    # Ensure correct handling of file path with aiofiles.open in binary mode
                    async with aiofiles.open(output_path, mode) as f:
                        progress_bar = tqdm(
                            total=total_size,
                            desc=filename.name,
                            unit="iB",
                            unit_scale=True,
                            ncols=100,
                            initial=file_size,  # Start the progress bar from the current file size
                        )
                        async for chunk in response.content.iter_chunked(
                            1024 * 10
                        ):  # Use larger chunk size for efficiency
                            await f.write(chunk)
                            progress_bar.update(len(chunk))
                        progress_bar.close()

                break
            except Exception as e:
                attempt += 1
                self.logger.warning(f"Attempt {attempt} failed: {e}")
                if attempt >= max_attempts:
                    self.logger.error(
                        "Failed to download '%s' after %d attempts",
                        filename,
                        max_attempts,
                    )
                    raise

    async def check_model_files(
        self,
        model: str,
        branch: str,
        links: List[str],
        sha256_dict: Dict[str, str],
        output_folder: Path,
    ):
        """Validate downloaded files with checksums."""
        validated = True
        for link in links:
            filename = Path(link).name
            file_path = output_folder / filename
            if not file_path.exists():
                print(f"The following file is missing: {file_path}")
                validated = False
                continue
        for filename, expected_hash in sha256_dict.items():
            file_path = output_folder / filename
            if not file_path.exists():
                print(f"The following file is missing: {file_path}")
                validated = False
                continue
            actual_hash = await self.calculate_file_sha256(file_path)
            if actual_hash != expected_hash:
                print(
                    f"Checksum failed for {file_path}: expected {expected_hash}, got {actual_hash}"
                )
                validated = False
            else:
                print(f"Checksum validated: {file_path}")
        if validated:
            print(
                f"[+] Validated all files and checksums for model: {model}, branch: {branch}!"
            )
        else:
            print(
                f"[-] Invalid files or checksums for model: {model}, branch: {branch}. You may need to rerun the download process with the --clean flag."
            )

    async def calculate_file_sha256(self, file_path: Path) -> str:
        """Calculate SHA256 hash for a file."""
        chunk_size = 1024 * 1024
        sha256 = hashlib.sha256()
        async with aio_open(file_path, "rb") as file:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()

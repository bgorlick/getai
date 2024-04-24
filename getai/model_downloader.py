import asyncio
import base64
import datetime
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional, Union, Dict, BinaryIO

import aiofiles
import aiohttp
from aiofiles import open as aio_open

from aiohttp import ClientSession, TCPConnector
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from rainbow_tqdm import tqdm

from .utils import convert_to_bytes, interactive_branch_selection
import chunk

BASE_URL = "https://huggingface.co"

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)

file_size_pattern = re.compile(r'<a class="[^"]*" title="Download file"[^>]*>([\d.]+ [GMK]B)')


class AsyncSearch:

    def __init__(self, query, filtered_models, total_models, branch_sizes, session, model_downloader):
        self.query = query
        self.filtered_models = filtered_models
        self.total_models = total_models
        self.branch_sizes = branch_sizes
        self.session = session
        self.model_downloader = model_downloader
        self.page_size = 20
        self.filtered_model_ids = set(model['id'] for model in filtered_models)
        self.main_search_models = filtered_models.copy()
        self.search_history = [(filtered_models.copy(), 1)]
        self.prefetched_pages = set()
    
    async def get_page_size(self):
        while True:
            print("Enter the desired number of results per page (default: 20), or press Enter to continue:")
            page_size_input = await asyncio.get_event_loop().run_in_executor(None, input, "> ")
            if page_size_input.strip():
                try:
                    self.page_size = int(page_size_input)
                    break
                except ValueError:
                    print("Invalid input. Please enter a valid integer.")
            else:
                break

    async def display_search_results(self):
        while True:
            total_pages = (len(self.filtered_models) + self.page_size - 1) // self.page_size
            current_page = self.search_history[-1][1]
            while True:
                await self.display_current_page(current_page, total_pages)
                user_input = await self.get_user_input(current_page, total_pages)

                if user_input.lower() == 'n' and current_page < total_pages:
                    current_page += 1
                    self.search_history[-1] = (self.filtered_models, current_page)
                elif user_input.lower() == 'p' and current_page > 1:
                    current_page -= 1
                    self.search_history[-1] = (self.filtered_models, current_page)
                elif user_input.lower() == 'f':
                    await self.filter_search_results()
                    break
                elif user_input.lower() == 's':
                    await self.sort_search_results()
                    break
                elif user_input.lower() == 'r':
                    if len(self.search_history) > 1:
                        self.search_history.pop()
                        self.filtered_models, current_page = self.search_history[-1]
                        self.total_models = len(self.filtered_models)
                        break
                    else:
                        print("You are already at the main search results.")
                elif user_input.isdigit() and 1 <= int(user_input) <= len(self.get_current_models(current_page)):
                    selected_model = self.get_current_models(current_page)[int(user_input) - 1]
                    branches = await self.model_downloader.get_model_branches(selected_model['id'])
                    if len(branches) > 1:
                        print(f"Multiple branches detected for model: {selected_model['id']}")
                        selected_branch = await self.model_downloader.select_branch_interactive(branches)
                        await self.model_downloader.download_model(selected_model['id'], selected_branch, False, False)
                    else:
                        await self.model_downloader.download_model(selected_model['id'], 'main', False, False)
                    break
                else:
                    print("Invalid input. Please try again.")

            await self.handle_user_choice()

    async def display_current_page(self, current_page, total_pages):
        start_index = (current_page - 1) * self.page_size
        end_index = min(start_index + self.page_size, len(self.filtered_models))
        current_models = self.filtered_models[start_index:end_index]
        print(f"Search results for '{self.query}' (Page {current_page} of {total_pages}, Total: {self.total_models}):")
        for i, model in enumerate(current_models, start=1):
            model_id, model_name, author = model['id'], model.get('modelName', model['id']), model.get('author', 'N/A')
            # Fetch branch file sizes for the current model
            branch_sizes = await self.model_downloader.get_branch_file_sizes(model_id)
            if branch_sizes:
                model_size = sum(branch_sizes.values()) / (1024 ** 3)
                size_str = f"{model_size:.2f} GB"
            else:
                size_str = 'N/A'
            print(f"{i}. \033[94m{model_name}\033[0m by \033[93m{author}\033[0m (\033[96m{model_id}\033[0m) \033[93m(Size: {size_str})\033[0m")
        print("Enter 'n' for next page, 'p' for previous page, 'f' to filter, 's' to sort, 'r' for previous results, or the model # to download.")
        
    async def prefetch_branch_sizes(self, current_page, total_pages):
        prefetch_threshold = 4  # Prefetch threshold in pages
        prefetch_pages = min(prefetch_threshold, total_pages - current_page)

        if prefetch_pages > 0:
            start_index = current_page * self.page_size
            end_index = min(start_index + self.page_size * prefetch_pages, len(self.filtered_models))
            models_to_prefetch = self.filtered_models[start_index:end_index]

            # Check if the pages to prefetch have already been fetched and cached
            pages_to_prefetch = set(range(current_page + 1, current_page + prefetch_pages + 1))
            pages_not_cached = pages_to_prefetch - self.prefetched_pages

            if pages_not_cached:
                # Filter models that haven't been prefetched yet
                models_not_prefetched = [model for model in models_to_prefetch if model['id'] not in self.branch_sizes]

                if models_not_prefetched:
                    prefetched_branch_sizes = await self.model_downloader.get_branch_file_sizes_for_models(models_not_prefetched, quiet=True)
                    self.branch_sizes.update(prefetched_branch_sizes)

                # Update the set of prefetched pages
                self.prefetched_pages.update(pages_not_cached)
            
    async def get_user_input(self, current_page, total_pages):
        current_models = self.get_current_models(current_page)
        model_completer = WordCompleter([str(i) for i in range(1, len(current_models) + 1)] + ['n', 'p', 'f', 's', 'r'])
        prompt_session = PromptSession(completer=model_completer)
        user_input = await prompt_session.prompt_async('Enter your choice: ')
        return user_input

    def get_current_models(self, current_page):
        start_index = (current_page - 1) * self.page_size
        end_index = min(start_index + self.page_size, len(self.filtered_models))
        current_models = self.filtered_models[start_index:end_index]
        return current_models

    async def handle_user_choice(self):
        print("Enter 'b' to see the filtered search results, 'm' for the main search results, or 'q' to quit.")
        user_input = await asyncio.get_event_loop().run_in_executor(None, input, "> ")
        if user_input.lower() == 'b':
            return
        elif user_input.lower() == 'm':
            self.filtered_models = self.main_search_models.copy()
            self.total_models = len(self.filtered_models)
            self.search_history = [(self.main_search_models.copy(), 1)]
            self.prefetched_pages.clear()  
        elif user_input.lower() == 'q':
            raise StopAsyncIteration
        else:
            print("Invalid input. Please try again.")

    async def filter_search_results(self):
        print("Enter the filter keyword:")
        filter_keyword = await asyncio.get_event_loop().run_in_executor(None, input, "> ")
        logging.debug(f"Filter keyword: {filter_keyword}")
        self.filtered_model_ids = set(model['id'] for model in self.filtered_models if filter_keyword.lower() in model['id'].lower())
        logging.debug(f"Filtered model IDs: {self.filtered_model_ids}")
        self.filtered_models = [model for model in self.filtered_models if model['id'] in self.filtered_model_ids]
        logging.debug(f"Filtered models: {self.filtered_models}")
        self.total_models = len(self.filtered_models)
        self.search_history.append((self.filtered_models.copy(), 1))
        self.prefetched_pages.clear()  
        if not self.filtered_models:
            print(f"No models found for the filter keyword '{filter_keyword}'.")

    async def sort_search_results(self):
        print("Enter the sort criteria (e.g., 'downloads', 'likes', 'lastModified'):")
        sort_criteria = await asyncio.get_event_loop().run_in_executor(None, input, "> ")
        reverse = True
        if sort_criteria.startswith('-'):
            sort_criteria = sort_criteria[1:]
            reverse = False
        self.filtered_models.sort(key=lambda x: x.get(sort_criteria, 0), reverse=reverse)
        self.search_history.append((self.filtered_models.copy(), 1))
        self.prefetched_pages.clear()  
        print(f"Search results sorted by '{sort_criteria}' in {'descending' if reverse else 'ascending'} order.")


class AsyncModelDownloader:

    def __init__(self, max_retries=5, output_dir=None, max_connections=5, hf_token=None):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.output_dir = output_dir
        self.max_retries = max_retries
        self.max_connections = max_connections
        self.token = hf_token
        self.connector = None
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=None)
        self.branch_sizes = {}
        self.file_locks = {}  

    async def __aenter__(self):
        self.connector = TCPConnector(limit=self.max_connections)
        headers = {'Authorization': f'Bearer {self.token}'} if self.token else {}
        self.session = ClientSession(connector=self.connector, headers=headers)
        self.logger.debug(f"Using token for authorization: {self.token is not None}")
        self.logger.debug(f"Session created with headers: {headers}")
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb):
        await self.close()

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None
        if self.connector:
            await self.connector.close()
            self.connector = None

    async def download_selected_model(self, model_id):
        if self.session:
            branches = await self.get_model_branches(model_id)
            selected_branch = await self.select_branch_interactive(branches)
            print(f"Downloading {model_id} from branch '\033[96m{selected_branch}\033[0m'...")
            await self.download_model(model_id, selected_branch, False, False)
        else:
            print(f"No active session available to download model {model_id}.")

    async def download_model(self, identifier, branch, clean, check):
        if self.session:
            branch_sizes = await self.get_branch_file_sizes(identifier)
            branches = await self.get_model_branches(identifier)
            branch_to_use = await self.select_branch(branches, branch, branch_sizes)
            links, sha256_list, is_lora, is_llamacpp = await self.get_download_links_from_huggingface(identifier, branch_to_use, text_only=False)
            output_folder = self.get_output_folder(identifier, branch_to_use, is_lora, is_llamacpp, self.output_dir)
            sha256_dict = dict(sha256_list)
            await self.download_model_files(identifier, branch_to_use, links, sha256_dict, output_folder)
            if check:
                await self.check_model_files(identifier, branch_to_use, links, sha256_dict, output_folder)
        else:
            print(f"No active session available to download model {identifier}.")

    async def http_get(self, session, url: str, temp_file: BinaryIO, resume_size=0, headers=None):
        headers = headers or {}
        if resume_size > 0:
            headers["Range"] = f"bytes={resume_size}-"
        async with session.get(url, headers=headers) as r:
            content_length = int(r.headers.get("Content-Length", 0))
            total = resume_size + content_length
            progress = tqdm(
                unit="B",
                unit_scale=True,
                total=total,
                initial=resume_size,
                desc=f"Downloading {Path(url).name}",
            )
            async for chunk in r.content.iter_chunked(1024):
                if chunk: 
                    progress.update(len(chunk))
                    temp_file.write(chunk)
            progress.close()

    async def download_model_file(self, session, url, output_folder, file_hash):
        filename = Path(url.rsplit('/', 1)[1])
        output_path = output_folder / filename

        # Create a lock for the file if it doesn't exist
        if output_path not in self.file_locks:
            self.file_locks[output_path] = asyncio.Lock()

        # Acquire the lock before accessing the file
        async with self.file_locks[output_path]:
            if output_path.exists():
                if file_hash is not None:
                    current_hash = await self.calculate_file_sha256(output_path)
                    if current_hash == file_hash:
                        self.logger.debug(f"'{filename}' exists and matches expected SHA256 hash; skipping.")
                        return
                    else:
                        self.logger.debug(f"'{filename}' exists but SHA256 hash doesn't match; resuming download.")
                        self.logger.debug(f"Expected SHA256: {file_hash}")
                        self.logger.debug(f"Actual SHA256: {current_hash}")
                        resume_size = output_path.stat().st_size
                else:
                    self.logger.debug(f"'{filename}' exists but no expected SHA256 hash provided; skipping.")
                    return
            else:
                resume_size = 0

            async with aiofiles.open(output_path, "ab") as temp_file:
                await self.http_get(session, url, temp_file, resume_size=resume_size)
                
    async def search_models(self, query, model_id_filter=None, rfilename_filter=None):
        async with self.session as session:
            url = f"{BASE_URL}/api/models"
            params = {
                "search": query,
                "sort": "downloads",
                "direction": "-1",
                "limit": 1000,
                "full": "true"
            }
            retry_count = 0
            while retry_count < self.max_retries:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data
                        total_models = len(models)
                        filtered_models = self.filter_models(models, model_id_filter, rfilename_filter)
                        if len(filtered_models) > 0:
                            search = AsyncSearch(query, filtered_models, total_models, {}, session, self)
                            await search.get_page_size()
                            await search.display_search_results()
                        else:
                            print(f"No models found for the search query '{query}'.")
                        break
                    elif response.status == 503:
                        retry_count += 1
                        if retry_count < self.max_retries:
                            delay = 2 ** retry_count
                            print(f"Received 503 error. Retrying in {delay} seconds...")
                            await asyncio.sleep(delay)
                        else:
                            print("Max retries exceeded. Skipping the request.")
                            break
                    else:
                        print(f"Error searching for models: HTTP {response.status}")
                        break

    def filter_models(self, models, model_id_filter, rfilename_filter):
        filtered_models = []
        for model in models:
            if model_id_filter and model_id_filter.lower() not in model.get("id", "").lower():
                continue
            if rfilename_filter:
                siblings = model.get("siblings", [])
                if not any(rfilename_filter.lower() in sibling.get("rfilename", "").lower() for sibling in siblings):
                    continue
            filtered_models.append(model)
        return filtered_models

    async def select_branch_interactive(self, branches):
        if len(branches) > 1:
            print("Available branches:")
            for i, branch in enumerate(branches, start=1):
                print(f"{i}. \033[96m{branch}\033[0m")
            branch_completer = WordCompleter([str(i) for i in range(1, len(branches) + 1)])
            branch_session = PromptSession(completer=branch_completer)
            selected_branch_index = await branch_session.prompt_async('Enter the number of the branch to download: ')
            if selected_branch_index.isdigit() and 1 <= int(selected_branch_index) <= len(branches):
                return branches[int(selected_branch_index) - 1]
        return 'main'

    async def select_branch(self, branches, branch_arg, branch_sizes):
        if branch_arg:
            if isinstance(branch_arg, str):
                return branch_arg if branch_arg in branches else 'main'
            else:
                branches = sorted(branches, key=lambda x: (x != 'main'))
                cyan_start = '\033[96m'
                cyan_end = '\033[0m'
                yellow_start = '\033[93m'
                yellow_end = '\033[0m'
                branches_info = [
                    f"{cyan_start}{branch}{cyan_end} ({yellow_start}{branch_sizes.get(branch, 0) / (1024**3):.2f} GB{yellow_end})"
                    for branch in branches
                ]
                print(f"Available Branches: [{', '.join(branches_info)}]")
                if self.session:
                    return await interactive_branch_selection(branches)
                else:
                    return 'main'
        return 'main'

    async def get_model_branches(self, model):
        url = f"{BASE_URL}/{model}/tree/main"
        if self.session:
            async with self.session.get(url) as response:
                if response.status == 200:
                    text = await response.text()
                    pattern = r'refs/heads/([^&]+)'
                    branch_matches = re.findall(pattern, text)
                    return list(set(branch_matches))
                else:
                    print(f"Error fetching file tree: HTTP {response.status}")
        return []

    def get_output_folder(self, identifier, branch, is_lora=False, is_llamacpp=False, output_dir=None):
        base_folder = output_dir or 'models'
        output_folder = f"{'_'.join(identifier.split('/')[-2:])}"
        if branch != 'main':
            output_folder += f'_{branch}'
        if is_lora:
            output_folder += '_lora'
        elif is_llamacpp:
            output_folder += '_llamacpp'
        return Path(base_folder) / output_folder

    async def get_download_links_from_huggingface(self, model, branch, text_only=False, specific_file=None):
        if self.session:
            page = f"/api/models/{model}/tree/{branch}"
            cursor = b""
            links = []
            sha256 = []
            classifications = []
            has_pytorch = False
            has_pt = False
            has_gguf = False
            has_safetensors = False
            is_lora = False
            while True:
                url = f"{BASE_URL}{page}" + (f"?cursor={cursor.decode()}" if cursor else "")
                self.logger.debug(f"Making request to: {url}")
                self.logger.debug(f"Request headers: {self.session._default_headers}")
                async with self.session.get(url) as response:
                    response.raise_for_status()
                    content = await response.text()
                    content_dict = json.loads(content)
                    if len(content_dict) == 0:
                        break
                    for i, content in enumerate(content_dict):
                        fname = content['path']
                        if specific_file not in [None, ''] and fname != specific_file:
                            continue
                        if not is_lora and fname.endswith(('adapter_config.json', 'adapter_model.bin')):
                            is_lora = True
                        is_pytorch = re.match(r"(pytorch|adapter|gptq)_model.*\.bin", fname)
                        is_safetensors = re.match(r".*\.safetensors", fname)
                        is_pt = re.match(r".*\.pt", fname)
                        is_gguf = re.match(r'.*\.gguf', fname)
                        is_tiktoken = re.match(r".*\.tiktoken", fname)
                        is_tokenizer = re.match(r"(tokenizer|ice|spiece).*\.model", fname) or is_tiktoken
                        is_text = re.match(r".*\.(txt|json|py|md)", fname) or is_tokenizer
                        if any((is_pytorch, is_safetensors, is_pt, is_gguf, is_tokenizer, is_text)):
                            if 'lfs' in content_dict[i]:
                                sha256.append([fname, content_dict[i]['lfs']['oid']])
                            if is_text:
                                links.append(f"https://huggingface.co/{model}/resolve/{branch}/{fname}")
                                classifications.append('text')
                                continue
                            if not text_only:
                                links.append(f"https://huggingface.co/{model}/resolve/{branch}/{fname}")
                                if is_safetensors:
                                    has_safetensors = True
                                    classifications.append('safetensors')
                                elif is_pytorch:
                                    has_pytorch = True
                                    classifications.append('pytorch')
                                elif is_pt:
                                    has_pt = True
                                    classifications.append('pt')
                                elif is_gguf:
                                    has_gguf = True
                                    classifications.append('gguf')
                    cursor = base64.b64encode(f'{{"file_name":"{content_dict[-1]["path"]}"}}'.encode()) + b':50'
                    cursor = base64.b64encode(cursor)
                    cursor = cursor.replace(b'=', b'%3D')
            if has_pytorch or has_pt and has_safetensors:
                links = [link for link, classification in zip(links, classifications) if classification not in ('pytorch', 'pt')]
            if has_gguf and specific_file is None:
                has_q4km = any('q4_k_m' in link.lower() for link in links)
                if has_q4km:
                    links = [link for link in links if 'q4_k_m' in link.lower()]
                else:
                    links = [link for link in links if not link.lower().endswith('.gguf')]
            is_llamacpp = has_gguf and specific_file is not None
            return links, sha256, is_lora, is_llamacpp
        return [], [], False, False

    async def download_model_files(self, model, branch, links, sha256_dict, output_folder):
        output_folder.mkdir(parents=True, exist_ok=True)
        metadata = f'url: https://huggingface.co/{model}\n' \
                f'branch: {branch}\n' \
                f'download date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
        sha256_str = '\n'.join([f' {sha256} {filename}' for filename, sha256 in sha256_dict.items()])
        if sha256_str:
            metadata += f'sha256sum:\n{sha256_str}'
        metadata += '\n'
        metadata_path = output_folder / 'huggingface-metadata.txt'
        async with aiofiles.open(metadata_path, 'w') as metafile:
            await metafile.write(metadata)
        
        semaphore = asyncio.Semaphore(self.max_connections)

        async def download_file(link):
            async with semaphore:
                filename = Path(link).name
                if filename in sha256_dict:
                    file_hash = sha256_dict[filename]
                else:
                    print(f"Warning: No SHA256 hash found for {filename}. Downloading without sha256 verification.")
                    file_hash = None
                await self.download_model_file(self.session, link, output_folder, file_hash)
        
        if self.session:
            tasks = [asyncio.ensure_future(download_file(link)) for link in links]
            await asyncio.gather(*tasks)

    async def check_model_files(self, model, branch, links, sha256_dict, output_folder):
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
                print(f"Checksum failed for {file_path}: expected {expected_hash}, got {actual_hash}")
                validated = False
            else:
                print(f"Checksum validated: {file_path}")
        if validated:
            print(f'[+] Validated all files and checksums for model: {model}, branch: {branch}!')
        else:
            print(f'[-] Invalid files or checksums for model: {model}, branch: {branch}. You may need to rerun the download process with the --clean flag.')

    async def calculate_file_sha256(self, file_path):
        chunk_size = 1024 * 1024
        sha256 = hashlib.sha256()
        async with aio_open(file_path, 'rb') as file:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()

    async def get_branch_file_sizes_for_models(self, models, quiet=False):
        branch_sizes = {}
        for model in models:
            model_id = model['id']
            if model_id not in branch_sizes:
                branch_sizes[model_id] = await self.get_branch_file_sizes(model_id, quiet=quiet)
        return branch_sizes

    async def get_branch_file_sizes(self, model, quiet=False):
        if not quiet:
            print(f"Fetching file sizes for {model}...")
        branches = await self.get_model_branches(model)
        branch_sizes = {}
        if self.session:
            for branch in branches:
                page_url = f"{BASE_URL}/{model}/tree/{branch}"
                try:
                    async with self.session.get(page_url) as response:
                        if response.status == 200:
                            text = await response.text()
                            file_sizes = file_size_pattern.findall(text)
                            total_size = sum(convert_to_bytes(size) for size in file_sizes)
                            branch_sizes[branch] = total_size
                        else:
                            if not quiet:
                                print(f"Failed to fetch page for branch: {branch}")
                except Exception as e:
                    if not quiet:
                        print(f"Error fetching branch file sizes for {model}, branch {branch}: {e}")
                    continue
        return branch_sizes

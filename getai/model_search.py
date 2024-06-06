""" model_search.py - GetAI model search and download functions. """

import asyncio
import logging
import re
from datetime import datetime
from typing import (
    Optional,
    Dict,
    List,
    Tuple,
    Set,
    Any,  # pylint: disable=unused-import noqa: F401
)
from aiohttp import ClientSession
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter

# Import API functions
from getai import download_model
from getai.utils import interactive_branch_selection, convert_to_bytes

BASE_URL = "https://huggingface.co"
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

file_size_pattern = re.compile(
    r'<a class="[^"]*" title="Download file"[^>]*>([\d.]+ [GMK]B)'
)

search_cache: Dict[str, List[Dict]] = {}


class AsyncModelSearch:
    """Class to asynchronously search and manage Hugging Face models."""

    def __init__(
        self,
        query: str,
        session: ClientSession,
        max_connections: int = 10,
        hf_token: Optional[str] = None,
        **kwargs: Any,  # Accepting additional keyword arguments for flexibility
    ):
        """Initialize AsyncModelSearch with query and session."""
        self.query = query
        self.page_size = 20
        self.filtered_models: List[Dict] = []
        self.total_models: int = 0
        self.branch_sizes: Dict[str, Dict[str, int]] = {}
        self.filtered_model_ids: Set[str] = set()
        self.main_search_models: List[Dict] = []
        self.search_history: List[Tuple[List[Dict], int]] = []
        self.prefetched_pages: Set[int] = set()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        self.token = hf_token
        self.max_connections = max_connections
        self.session = session
        self.semaphore = asyncio.Semaphore(self.max_connections)

    async def search_models(
        self,
        query: Optional[str] = None,
        model_id_filter: Optional[str] = None,
        rfilename_filter: Optional[str] = None,
    ) -> None:
        """Search models based on the query and filters."""
        if not query:
            query = self.query

        if query in search_cache:
            models = search_cache[query]
        else:
            if self.session is None:
                self.logger.error("Session is not initialized.")
                return

            url = f"{BASE_URL}/api/models"
            params = {
                "search": query,
                "sort": "downloads",
                "direction": "-1",
                "limit": 1000,
                "full": "true",
            }
            retry_count = 0
            models = []
            while retry_count < self.max_connections:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data
                        search_cache[query] = models
                        break
                    elif response.status == 503:
                        retry_count += 1
                        if retry_count < self.max_connections:
                            delay = 2**retry_count
                            self.logger.warning(
                                "Received 503 error. Retrying in %s seconds...", delay
                            )
                            await asyncio.sleep(delay)
                        else:
                            self.logger.error(
                                "Max retries exceeded. Skipping the request."
                            )
                            return
                    else:
                        self.logger.error(
                            "Error searching for models: HTTP %s", response.status
                        )
                        return

        self.total_models = len(models)
        self.filtered_models = self.filter_models(
            models, model_id_filter, rfilename_filter
        )
        if self.filtered_models:
            self.main_search_models = self.filtered_models.copy()
            self.search_history = [(self.filtered_models.copy(), 1)]
            await self.display_search_results()
        else:
            self.logger.info("No models found for the search query '%s'.", query)

    def filter_models(
        self,
        models: List[Dict],
        model_id_filter: Optional[str],
        rfilename_filter: Optional[str],
    ) -> List[Dict]:
        """Filter models by ID and filename."""
        filtered_models = []
        for model in models:
            if (
                model_id_filter
                and model_id_filter.lower() not in model.get("id", "").lower()
            ):
                continue
            if rfilename_filter:
                siblings = model.get("siblings", [])
                if not any(
                    rfilename_filter.lower() in sibling.get("rfilename", "").lower()
                    for sibling in siblings
                ):
                    continue
            filtered_models.append(model)
        return filtered_models

    async def display_search_results(self):
        """Display the search results interactively."""
        total_pages = (len(self.filtered_models) + self.page_size - 1) // self.page_size
        current_page = self.search_history[-1][1]

        asyncio.create_task(self.prefetch_branch_sizes(current_page, current_page + 1))

        while True:
            await self.display_current_page(current_page, total_pages)

            if current_page == 1:
                asyncio.create_task(
                    self.prefetch_remaining_branch_sizes(current_page, total_pages)
                )

            user_input = await self.get_user_input(current_page)
            if user_input.lower() == "n" and current_page < total_pages:
                current_page += 1
                self.search_history[-1] = (self.filtered_models.copy(), current_page)
            elif user_input.lower() == "p" and current_page > 1:
                current_page -= 1
                self.search_history[-1] = (self.filtered_models.copy(), current_page)
            elif user_input.lower() == "f":
                await self.filter_search_results()
                break
            elif user_input.lower() == "s":
                await self.sort_search_results()
                break
            elif user_input.lower() == "r":
                if len(self.search_history) > 1:
                    self.search_history.pop()
                    self.filtered_models, current_page = self.search_history[-1]
                    self.total_models = len(self.filtered_models)
                else:
                    self.logger.info("You are already at the main search results.")
            elif user_input.isdigit() and 1 <= int(user_input) <= len(
                self.get_current_models(current_page)
            ):
                selected_model = self.get_current_models(current_page)[
                    int(user_input) - 1
                ]
                branches = await self.get_model_branches(selected_model["id"])
                selected_branch = await self.select_branch_interactive(branches)
                await download_model(
                    identifier=selected_model["id"],
                    branch=selected_branch,
                    hf_token=self.token,
                    max_connections=self.max_connections,
                )
            else:
                self.logger.error("Invalid input. Please try again.")

    async def display_current_page(self, current_page: int, total_pages: int):
        """Display the models for the current page."""
        start_index = (current_page - 1) * self.page_size
        end_index = min(start_index + self.page_size, len(self.filtered_models))
        current_models = self.filtered_models[start_index:end_index]

        self.logger.info(
            "Search results for '%s' (Page %d of %d, Total: %d):",
            self.query,
            current_page,
            total_pages,
            self.total_models,
        )
        for i, model in enumerate(current_models, start=1):
            model_id, model_name, author = (
                model["id"],
                model.get("modelName", model["id"]),
                model.get("author", "N/A"),
            )
            size_str = await self.get_model_size_str(model_id)
            last_modified = model.get("lastModified", "N/A")
            if last_modified != "N/A":
                try:
                    last_modified_dt = datetime.fromisoformat(last_modified[:-1])
                    last_modified = last_modified_dt.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    last_modified = "Invalid date format"
            print(
                f"{i}. \033[94m{model_name}\033[0m by \033[94m{author}\033[0m "
                f"(\033[96m{model_id}\033[0m) (\033[93mSize: {size_str}\033[0m) (\033[97mLast updated: {last_modified}\033[0m)"
            )
        print(
            "Enter 'n' for next page, 'p' for previous page, 'f' to filter, 's' to sort, 'r' for previous results, or the model # to download."
        )

    async def get_model_size_str(self, model_id: str) -> str:
        """Get the size of the model as a string."""
        if model_id not in self.branch_sizes:
            async with self.semaphore:
                branch_sizes = await self.get_branch_file_sizes(model_id)
            self.branch_sizes[model_id] = branch_sizes
        else:
            branch_sizes = self.branch_sizes[model_id]

        if branch_sizes:
            model_size = sum(branch_sizes.values()) / (1024**3)
            return f"{model_size:.2f} GB"
        return "N/A"

    async def prefetch_branch_sizes(self, start_page: int, end_page: int):
        """Prefetch the branch sizes for the models on specified pages."""
        start_index = (start_page - 1) * self.page_size
        end_index = min(end_page * self.page_size, len(self.filtered_models))
        models_to_prefetch = self.filtered_models[start_index:end_index]

        tasks = [
            self.get_branch_file_sizes(model["id"]) for model in models_to_prefetch
        ]

        results = await asyncio.gather(*tasks)

        for model, sizes in zip(models_to_prefetch, results):
            self.branch_sizes[model["id"]] = sizes
            self.update_display_with_new_size(model["id"])

        self.prefetched_pages.update(range(start_page, end_page))

    async def prefetch_remaining_branch_sizes(
        self, current_page: int, total_pages: int
    ):
        """Prefetch branch sizes for remaining models beyond current page."""
        prefetch_threshold = 2
        prefetch_pages = min(prefetch_threshold, total_pages - current_page)
        if prefetch_pages > 0:
            start_index = current_page * self.page_size
            end_index = min(
                start_index + self.page_size * prefetch_pages, len(self.filtered_models)
            )
            models_to_prefetch = self.filtered_models[start_index:end_index]

            pages_to_prefetch = set(
                range(current_page + 1, current_page + prefetch_pages + 1)
            )
            pages_not_cached = pages_to_prefetch - self.prefetched_pages

            if pages_not_cached:
                models_not_prefetched = [
                    model
                    for model in models_to_prefetch
                    if model["id"] not in self.branch_sizes
                ]
                if models_not_prefetched:
                    tasks = [
                        self.get_branch_file_sizes(model["id"])
                        for model in models_not_prefetched
                    ]
                    prefetched_branch_sizes = await asyncio.gather(*tasks)
                    for model, sizes in zip(
                        models_not_prefetched, prefetched_branch_sizes
                    ):
                        self.branch_sizes[model["id"]] = sizes

                self.prefetched_pages.update(pages_not_cached)

    async def get_user_input(self, current_page: int) -> str:
        """Get user input for the current page."""
        current_models = self.get_current_models(current_page)
        model_completer = WordCompleter(
            [str(i) for i in range(1, len(current_models) + 1)]
            + ["n", "p", "f", "s", "r"]
        )
        prompt_session: PromptSession = PromptSession(completer=model_completer)
        return await prompt_session.prompt_async("Enter your choice: ")

    def get_current_models(self, current_page: int) -> List[Dict]:
        """Get the models for the current page."""
        start_index = (current_page - 1) * self.page_size
        end_index = min(start_index + self.page_size, len(self.filtered_models))
        return self.filtered_models[start_index:end_index]

    async def filter_search_results(self):
        """Filter search results based on user input."""
        print("Enter the filter keyword:")
        filter_keyword = await asyncio.get_event_loop().run_in_executor(
            None, input, "> "
        )
        self.filtered_model_ids = {
            model["id"]
            for model in self.filtered_models
            if filter_keyword.lower() in model["id"].lower()
        }
        self.filtered_models = [
            model
            for model in self.filtered_models
            if model["id"] in self.filtered_model_ids
        ]
        self.total_models = len(self.filtered_models)
        self.search_history.append((self.filtered_models.copy(), 1))
        self.prefetched_pages.clear()
        if not self.filtered_models:
            print(f"No models found for the filter keyword '{filter_keyword}'.")

    async def sort_search_results(self):
        """Sort search results based on user input criteria."""
        print("Enter the sort criteria (e.g., 'downloads', 'likes', 'lastModified'):")
        sort_criteria = await asyncio.get_event_loop().run_in_executor(
            None, input, "> "
        )
        reverse = True
        if sort_criteria.startswith("-"):
            sort_criteria = sort_criteria[1:]
            reverse = False
        self.filtered_models.sort(
            key=lambda x: x.get(sort_criteria, 0), reverse=reverse
        )
        self.search_history.append((self.filtered_models.copy(), 1))
        self.prefetched_pages.clear()
        print(
            f"Search results sorted by '{sort_criteria}' in {'descending' if reverse else 'ascending'} order."
        )

    async def select_branch_interactive(self, branches: List[str]) -> str:
        """Select a branch interactively from the list."""
        if len(branches) > 1:
            print("Available branches:")
            for i, branch in enumerate(branches, start=1):
                print(f"{i}. \033[96m{branch}\033[0m")
            branch_completer = WordCompleter(
                [str(i) for i in range(1, len(branches) + 1)]
            )
            prompt_session: PromptSession = PromptSession(completer=branch_completer)
            selected_branch_index = await prompt_session.prompt_async(
                "Enter the number of the branch to download: "
            )
            if selected_branch_index.isdigit() and 1 <= int(
                selected_branch_index
            ) <= len(branches):
                return branches[int(selected_branch_index) - 1]
        return "main"

    async def select_branch(
        self,
        branches: List[str],
        branch_arg: Optional[str],
    ) -> str:
        """Select a branch based on user input or default to main."""
        if branch_arg:
            if isinstance(branch_arg, str):
                return branch_arg if branch_arg in branches else "main"
            else:
                branches = sorted(branches, key=lambda x: (x != "main"))

                cyan_start = "\033[96m"
                cyan_end = "\033[0m"
                yellow_start = "\033[93m"
                yellow_end = "\033[0m"

                branches_info = []
                for branch in branches:
                    size = self.branch_sizes.get(branch, 0)
                    if isinstance(size, int):
                        size_gb = size / (1024**3)
                    else:
                        size_gb = 0.0
                    branches_info.append(
                        f"{cyan_start}{branch}{cyan_end} ({yellow_start}{size_gb:.2f} GB{yellow_end})"
                    )

                print(f"Available Branches: [{', '.join(branches_info)}]")
                return await interactive_branch_selection(branches)
        return "main"

    async def get_model_branches(self, model: str) -> List[str]:
        """Get a list of branches for the given model."""
        if self.session is None:
            self.logger.error("Session is not initialized.")
            return []

        url = f"{BASE_URL}/{model}/tree/main"
        async with self.session.get(url) as response:
            if response.status == 200:
                text = await response.text()
                pattern = r"refs/heads/([^&]+)"
                branch_matches = re.findall(pattern, text)
                return list(set(branch_matches))
            else:
                self.logger.error("Error fetching file tree: HTTP %s", response.status)
        return []

    def update_display_with_new_size(self, model_id: str):
        """Update the display with new size information."""
        # can be used for debugging
        # print(f"Updated size for model {model_id}: {self.branch_sizes[model_id]}")
        pass

    async def get_branch_file_sizes_for_models(
        self, models: List[Dict], quiet: bool = False
    ) -> Dict[str, Dict[str, int]]:
        """Get file sizes for branches of multiple models."""
        branch_sizes: Dict[str, Dict[str, int]] = {}
        for model in models:
            model_id = model["id"]
            if model_id not in branch_sizes:
                branch_sizes[model_id] = await self.get_branch_file_sizes(
                    model_id, quiet=quiet
                )
        return branch_sizes

    async def get_branch_file_sizes(
        self, model: str, quiet: bool = False
    ) -> Dict[str, int]:
        """Get file sizes for all branches of a model."""
        if not quiet:
            self.logger.debug("Fetching file sizes for %s...", model)
        branches = await self.get_model_branches(model)
        branch_sizes: Dict[str, int] = {}
        if self.session:
            for branch in branches:
                page_url = f"{BASE_URL}/{model}/tree/{branch}"
                try:
                    async with self.session.get(page_url) as response:
                        if response.status == 200:
                            text = await response.text()
                            file_sizes = file_size_pattern.findall(text)
                            total_size = sum(
                                convert_to_bytes(size) for size in file_sizes
                            )
                            branch_sizes[branch] = total_size
                        else:
                            if not quiet:
                                self.logger.info(
                                    "Failed to fetch page for branch: %s", branch
                                )
                except Exception as e:
                    if not quiet:
                        self.logger.info(
                            "Error fetching branch file sizes for %s, branch %s: %s",
                            model,
                            branch,
                            e,
                        )
                    continue
        return branch_sizes

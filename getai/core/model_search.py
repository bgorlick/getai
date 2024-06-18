import asyncio
import logging
import re
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Set, Any
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from getai.core.session_manager import SessionManager
from getai import api

BASE_URL = "https://huggingface.co"
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.ERROR)

file_size_pattern = re.compile(
    r'<a class="[^"]*" title="Download file"[^>]*>([\d.]+ [GMK]B)'
)

search_cache: Dict[str, List[Dict]] = {}


class AsyncModelSearch:
    """Class to asynchronously search and manage Hugging Face models."""

    def __init__(
        self,
        query: str,
        max_connections: int = 10,
        hf_token: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize AsyncModelSearch with query."""
        self.query = query
        self.page_size = 20
        self.filtered_models: List[Dict] = []
        self.total_models: int = 0
        self.branch_sizes: Dict[str, Dict[str, int]] = {}
        self.model_branch_info: Dict[str, Dict[str, Any]] = {}
        self.filtered_model_ids: Set[str] = set()
        self.main_search_models: List[Dict] = []
        self.search_history: List[Tuple[List[Dict], int]] = []
        self.prefetched_pages: Set[int] = set()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.NOTSET)
        self.filter_flag = False

        self.token = hf_token
        self.max_connections = max_connections
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
            async with SessionManager(
                max_connections=self.max_connections, hf_token=self.token
            ) as manager:
                session = await manager.get_session()
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
                    async with session.get(url, params=params) as response:
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
                                    "Received 503 error. Retrying in %s seconds...",
                                    delay,
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
            elif user_input.lower() == "s":
                await self.sort_search_results()
            elif user_input.lower() == "r":
                if len(self.search_history) > 1:
                    self.search_history.pop()
                    self.filtered_models, current_page = self.search_history[-1]
                    self.total_models = len(self.filtered_models)
                else:
                    self.logger.info("You are already at the main search results.")
            elif user_input.lower() == "none":
                self.filtered_models = self.main_search_models
                self.filter_flag = False
                await self.display_search_results()
            elif user_input.lower() == "q":
                break
            elif user_input.isdigit() and 1 <= int(user_input) <= len(
                self.get_current_models(current_page)
            ):
                selected_model = self.get_current_models(current_page)[
                    int(user_input) - 1
                ]
                branches = await self.get_model_branches(selected_model["id"])
                selected_branch = await self.select_branch_interactive(
                    branches, selected_model["id"]
                )
                await api.download_model(
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

            branches_info = ""
            if self.model_branch_info.get(model_id, {}).get("has_branches", False):
                branches_count = len(self.model_branch_info[model_id]["branches"])
                branches_info = f" | Branches: {branches_count}"
            # ascii grey
            ascii_grey_bold = "\033[90m\033[1m"
            ascii_bold_green = "\033[92m\033[1m"
            ascii_bold_magenta = "\033[95m\033[1m"
            abwhite = "\033[97m\033[1m"
            qrst = "\033[0m"

            print(
                f"{i}. \033[96m{model_name}\033[0m by \033[94m{author}\033[0m | "
                f"(\033[93mSize: {size_str}\033[0m{branches_info}) "
                f"(\033[97m{last_modified}\033[0m)\n"
                f"{ascii_grey_bold}{'-' * 100}\033[0m"
            )
        print(
            f"{ascii_bold_green}getai search commands{ascii_bold_magenta}> {abwhite} #{qrst} download model, {abwhite}'n'{qrst} (next), {abwhite}'p'{qrst} (prev), {abwhite}'f'{qrst} (filter), {abwhite}'s'{qrst} (sort), {abwhite}'r'{qrst} (results), {abwhite}'none'{qrst} (all), {abwhite} 'q'{qrst} to quit.\033[0m"
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
            self.model_branch_info[model["id"]] = {
                "has_branches": len(sizes) > 1,
                "branches": list(sizes.keys()),
            }
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
                        self.model_branch_info[model["id"]] = {
                            "has_branches": len(sizes) > 1,
                            "branches": list(sizes.keys()),
                        }

                self.prefetched_pages.update(pages_not_cached)

    async def get_user_input(self, current_page: int) -> str:
        """Get user input for the current page."""
        current_models = self.get_current_models(current_page)
        model_completer = WordCompleter(
            [str(i) for i in range(1, len(current_models) + 1)]
            + ["n", "p", "f", "s", "r", "none"]
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
        print(
            "Enter a filter query to filter the results (or type 'none' to show all results):"
        )
        filter_keyword = await asyncio.get_event_loop().run_in_executor(
            None, input, "> "
        )
        if filter_keyword.lower() == "none":
            self.filtered_models = self.main_search_models
            self.filter_flag = False
        else:
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
            self.filter_flag = True
        self.total_models = len(self.filtered_models)
        self.search_history.append((self.filtered_models.copy(), 1))
        self.prefetched_pages.clear()
        if not self.filtered_models:
            print(f"No models found for the filter keyword '{filter_keyword}'.")
        await self.display_search_results()

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
        await self.display_search_results()

    async def select_branch_interactive(
        self, branches: List[str], model_id: str
    ) -> str:
        """Select a branch interactively from the list."""
        if len(branches) > 1:
            print("Available branches:")
            for i, branch in enumerate(branches, start=1):
                size = self.branch_sizes.get(model_id, {}).get(branch, 0)
                size_gb = size / (1024**3) if isinstance(size, int) else 0.0
                print(f"{i}. \033[96m{branch}\033[0m (Size: {size_gb:.2f} GB)")
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
        from getai.core import interactive_branch_selection

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
                    size = self.branch_sizes.get(branch, {}).get(branch, 0)
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
        async with SessionManager(
            max_connections=self.max_connections, hf_token=self.token
        ) as manager:
            session = await manager.get_session()
            url = f"{BASE_URL}/{model}/tree/main"
            async with session.get(url) as response:
                if response.status == 200:
                    text = await response.text()
                    pattern = r"refs/heads/([^&]+)"
                    branch_matches = re.findall(pattern, text)
                    return list(set(branch_matches))
                else:
                    self.logger.error(
                        "Error fetching file tree: HTTP %s", response.status
                    )
        return []

    def update_display_with_new_size(self, model_id: str):
        """Update the display with new size information."""
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
        from getai.core import convert_to_bytes

        if not quiet:
            self.logger.debug("Fetching file sizes for %s...", model)
        branches = await self.get_model_branches(model)
        branch_sizes: Dict[str, int] = {}
        async with SessionManager(
            max_connections=self.max_connections, hf_token=self.token
        ) as manager:
            session = await manager.get_session()
            for branch in branches:
                page_url = f"{BASE_URL}/{model}/tree/{branch}"
                try:
                    async with session.get(page_url) as response:
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

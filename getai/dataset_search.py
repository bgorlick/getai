""" dataset_search.py - Asynchronous dataset search for Hugging Face datasets API. """

from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import aiohttp
import asyncio
from aiohttp import ClientSession
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter

logging.basicConfig(
    format="%(name)s - %(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://huggingface.co"


class AsyncDatasetSearch:
    """Asynchronous dataset search for Hugging Face datasets API."""

    def __init__(
        self,
        query: str,
        filtered_datasets: List[Dict[str, Any]],
        total_datasets: int,
        output_dir: Path,
        max_connections: int,
        hf_token: Optional[str],
        session: ClientSession,
    ):
        self.config = {
            "query": query,
            "total_datasets": total_datasets,
            "output_dir": output_dir,
            "max_connections": max_connections,
            "hf_token": hf_token,
            "session": session,
            "page_size": 20,
            "timeout": aiohttp.ClientTimeout(total=None),
        }
        self.data = {
            "filtered_datasets": self.sort_by_last_modified(filtered_datasets),
            "filtered_dataset_ids": {dataset["id"] for dataset in filtered_datasets},
            "main_search_datasets": filtered_datasets.copy(),
            "search_history": [(filtered_datasets.copy(), 1)],
        }
        self.session = session

    @staticmethod
    def sort_by_last_modified(datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort the datasets by lastModified date in descending order."""
        return sorted(datasets, key=lambda x: x.get("lastModified", ""), reverse=True)

    async def search_datasets(self, query: str, **kwargs):
        """Search datasets based on a query."""
        if not self.session:
            raise RuntimeError("Session is not initialized")

        params = {
            "search": query,
            "limit": 50,
        }
        params.update(kwargs)

        valid_types = (str, int, float)
        params = {
            k: (str(v).lower() if isinstance(v, bool) else v)
            for k, v in params.items()
            if isinstance(v, valid_types + (bool,))
        }

        search_url = f"{BASE_URL}/api/datasets"
        async with self.session.get(search_url, params=params) as response:
            if response.status == 200:
                datasets = await response.json()
                sorted_datasets = self.sort_by_last_modified(datasets)
                self.data["filtered_datasets"] = sorted_datasets
                self.data["main_search_datasets"] = sorted_datasets.copy()
                self.data["search_history"] = [(sorted_datasets.copy(), 1)]
                self.config["total_datasets"] = len(sorted_datasets)
            else:
                raise RuntimeError(f"Failed to fetch datasets: {response.status}")

    async def display_dataset_search_results(self):
        """Display dataset search results and handle user interaction for downloading."""
        await self.search_datasets(self.config["query"])

        while True:
            total_pages = (
                len(self.data["filtered_datasets"]) + self.config["page_size"] - 1
            ) // self.config["page_size"]
            current_page = self.data["search_history"][-1][1]

            while True:
                await self.display_current_dataset_page(current_page, total_pages)
                user_input = await self.get_dataset_user_input(
                    current_page, total_pages
                )

                if user_input.lower() == "n" and current_page < total_pages:
                    current_page += 1
                    self.data["search_history"][-1] = (
                        self.data["filtered_datasets"],
                        current_page,
                    )
                elif user_input.lower() == "p" and current_page > 1:
                    current_page -= 1
                    self.data["search_history"][-1] = (
                        self.data["filtered_datasets"],
                        current_page,
                    )
                elif user_input.lower() == "f":
                    await self.filter_dataset_search_results()
                    break
                elif user_input.lower() == "s":
                    await self.sort_dataset_search_results()
                    break
                elif user_input.lower() == "r":
                    if len(self.data["search_history"]) > 1:
                        self.data["search_history"].pop()
                        (
                            self.data["filtered_datasets"],
                            current_page,
                        ) = self.data[
                            "search_history"
                        ][-1]
                        self.config["total_datasets"] = len(
                            self.data["filtered_datasets"]
                        )
                        break
                    else:
                        logger.info("You are already at the main search results.")
                elif user_input.isdigit() and 1 <= int(user_input) <= len(
                    self.get_current_datasets(current_page)
                ):
                    selected_dataset = self.get_current_datasets(current_page)[
                        int(user_input) - 1
                    ]
                    output_folder = (
                        Path(self.config["output_dir"]) / selected_dataset["id"]
                    )
                    output_folder.mkdir(parents=True, exist_ok=True)

                    # Delayed import to avoid circular import issue
                    from getai.api import download_dataset

                    # Call download_dataset from getai.api
                    await download_dataset(
                        identifier=selected_dataset["id"],
                        hf_token=self.config["hf_token"],
                        max_connections=self.config["max_connections"],
                        output_dir=output_folder,
                    )
                    break
                else:
                    logger.error("Invalid input. Please try again.")

            await self.handle_dataset_user_choice()

    async def display_current_dataset_page(self, current_page, total_pages):
        """Display the current page of dataset search results."""
        start_index = (current_page - 1) * self.config["page_size"]
        end_index = min(
            start_index + self.config["page_size"], len(self.data["filtered_datasets"])
        )
        current_datasets = self.data["filtered_datasets"][start_index:end_index]
        logger.info(
            "Search results for '%s' (Page %d of %d, Total: %d):",
            self.config["query"],
            current_page,
            total_pages,
            self.config["total_datasets"],
        )
        for i, dataset in enumerate(current_datasets, start=1):
            dataset_id, dataset_name = dataset["id"], dataset.get(
                "dataset_name", dataset["id"]
            )
            last_modified = dataset.get("lastModified", "N/A")
            if last_modified != "N/A":
                last_modified = datetime.fromisoformat(last_modified[:-1]).strftime(
                    "%Y-%m-%d %H:%M"
                )
            logger.info(
                "%d. \033[94m%s\033[0m (\033[96m%s\033[0m) \033[93m(Last updated: %s)\033[0m",
                i,
                dataset_name,
                dataset_id,
                last_modified,
            )
        logger.info(
            "Enter 'n' for the next page, 'p' for the previous page, 'f' to filter, 's' to sort, 'r' to return to previous search results, or the dataset number to download."
        )

    async def get_dataset_user_input(self, current_page, total_pages):
        """Get user input for dataset interaction."""
        current_datasets = self.get_current_datasets(current_page)
        dataset_completer = WordCompleter(
            [str(i) for i in range(1, len(current_datasets) + 1)]
            + ["n", "p", "f", "s", "r"]
        )
        prompt_session = PromptSession(completer=dataset_completer)
        user_input = await prompt_session.prompt_async("Enter your choice: ")
        return user_input

    def get_current_datasets(self, current_page):
        """Get the current datasets for the specified page."""
        start_index = (current_page - 1) * self.config["page_size"]
        end_index = min(
            start_index + self.config["page_size"], len(self.data["filtered_datasets"])
        )
        current_datasets = self.data["filtered_datasets"][start_index:end_index]
        return current_datasets

    async def handle_dataset_user_choice(self):
        """Handle user choice for dataset interaction."""
        logger.info(
            "Enter 'b' to go back to the filtered search results, 'm' to return to the main search results, or 'q' to quit."
        )
        user_input = await asyncio.get_event_loop().run_in_executor(None, input, "> ")

        if user_input.lower() == "b":
            return
        elif user_input.lower() == "m":
            self.data["filtered_datasets"] = self.sort_by_last_modified(
                self.data["main_search_datasets"].copy()
            )
            self.config["total_datasets"] = len(self.data["filtered_datasets"])
            self.data["search_history"] = [
                (self.data["main_search_datasets"].copy(), 1)
            ]
        elif user_input.lower() == "q":
            raise StopAsyncIteration
        else:
            logger.error("Invalid input. Please try again.")

    async def filter_dataset_search_results(self):
        """Filter dataset search results based on user input."""
        logger.info("Enter the filter keyword:")
        filter_keyword = await asyncio.get_event_loop().run_in_executor(
            None, input, "> "
        )
        self.data["filtered_dataset_ids"] = {
            dataset["id"]
            for dataset in self.data["filtered_datasets"]
            if filter_keyword.lower() in dataset["id"].lower()
        }
        self.data["filtered_datasets"] = self.sort_by_last_modified(
            [
                dataset
                for dataset in self.data["filtered_datasets"]
                if dataset["id"] in self.data["filtered_dataset_ids"]
            ]
        )
        self.config["total_datasets"] = len(self.data["filtered_datasets"])
        self.data["search_history"].append((self.data["filtered_datasets"].copy(), 1))
        if not self.data["filtered_datasets"]:
            logging.warning(
                "No datasets found for the filter keyword '%s'.", filter_keyword
            )

    async def sort_dataset_search_results(self):
        """Sort dataset search results based on user input."""
        logger.info(
            "Enter the sort criteria (e.g., 'downloads', 'likes', 'lastModified'):"
        )
        sort_criteria = await asyncio.get_event_loop().run_in_executor(
            None, input, "> "
        )
        reverse = True
        if sort_criteria.startswith("-"):
            sort_criteria = sort_criteria[1:]
            reverse = False
        self.data["filtered_datasets"].sort(
            key=lambda x: x.get(sort_criteria, 0), reverse=reverse
        )
        self.data["search_history"].append((self.data["filtered_datasets"].copy(), 1))
        logger.info(
            "Search results sorted by '%s' in %s order.",
            sort_criteria,
            "descending" if reverse else "ascending",
        )

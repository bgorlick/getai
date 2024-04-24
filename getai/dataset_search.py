# dataset_search.py

""" dataset_search.py - Asynchronous dataset search for Hugging Face datasets API. """

from datetime import datetime
from pathlib import Path
import asyncio
import aiohttp
from aiohttp import ClientSession
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
import logging

from .utils import get_dataset_output_folder

logging.basicConfig(
    format='%(name)s - %(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

BASE_URL = "https://huggingface.co"


class AsyncDatasetSearch:

    def __init__(self, query, filtered_datasets, total_datasets, output_dir, max_connections, hf_token):
        self.query = query
        self.filtered_datasets = filtered_datasets
        self.total_datasets = total_datasets
        self.output_dir = output_dir
        self.max_connections = max_connections
        self.hf_token = hf_token
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=None)
        self.page_size = 20
        self.filtered_dataset_ids = set(dataset['id'] for dataset in filtered_datasets)
        self.main_search_datasets = filtered_datasets.copy()
        self.search_history = [(filtered_datasets.copy(), 1)]

    async def __aenter__(self):
        self.session = ClientSession(headers={'Authorization': f'Bearer {self.hf_token}'})
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_session()

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def get_dataset_page_size(self):
        while True:
            try:
                print("Enter the desired number of results per page (default: 20), or press Enter to continue:")
                page_size_input = await asyncio.get_event_loop().run_in_executor(None, input, "> ")
                if page_size_input.strip():
                    self.page_size = int(page_size_input)
                    break
                else:
                    break
            except ValueError:
                logger.error("Invalid input. Please enter a valid integer.")

    async def display_dataset_search_results(self, downloader):
        while True:
            total_pages = (len(self.filtered_datasets) + self.page_size - 1) // self.page_size
            current_page = self.search_history[-1][1]

            while True:
                await self.display_current_dataset_page(current_page, total_pages)
                user_input = await self.get_dataset_user_input(current_page, total_pages)

                if user_input.lower() == 'n' and current_page < total_pages:
                    current_page += 1
                    self.search_history[-1] = (self.filtered_datasets, current_page)
                elif user_input.lower() == 'p' and current_page > 1:
                    current_page -= 1
                    self.search_history[-1] = (self.filtered_datasets, current_page)
                elif user_input.lower() == 'f':
                    await self.filter_dataset_search_results()
                    break
                elif user_input.lower() == 's':
                    await self.sort_dataset_search_results()
                    break
                elif user_input.lower() == 'r':
                    if len(self.search_history) > 1:
                        self.search_history.pop()
                        self.filtered_datasets, current_page = self.search_history[-1]
                        self.total_datasets = len(self.filtered_datasets)
                        break
                    else:
                        logger.info("You are already at the main search results.")
                elif user_input.isdigit() and 1 <= int(user_input) <= len(self.get_current_datasets(current_page)):
                    selected_dataset = self.get_current_datasets(current_page)[int(user_input) - 1]
                    output_folder = get_dataset_output_folder(selected_dataset['id'], self.output_dir)
                    await downloader.download_dataset_info(selected_dataset['id'], output_folder=output_folder)
                    break
                else:
                    logger.error("Invalid input. Please try again.")

            await self.handle_dataset_user_choice()

    async def display_current_dataset_page(self, current_page, total_pages):
        start_index = (current_page - 1) * self.page_size
        end_index = min(start_index + self.page_size, len(self.filtered_datasets))
        current_datasets = self.filtered_datasets[start_index:end_index]
        logger.info(f"Search results for '{self.query}' (Page {current_page} of {total_pages}, Total: {self.total_datasets}):")
        for i, dataset in enumerate(current_datasets, start=1):
            dataset_id, dataset_name = dataset['id'], dataset.get('dataset_name', dataset['id'])
            last_modified = dataset.get('lastModified', 'N/A')
            if last_modified != 'N/A':
                last_modified = datetime.fromisoformat(last_modified[:-1]).strftime('%Y-%m-%d %H:%M')
            logger.info(f"{i}. \033[94m{dataset_name}\033[0m (\033[96m{dataset_id}\033[0m) \033[93m(Last updated: {last_modified})\033[0m")
        logger.info("Enter 'n' for the next page, 'p' for the previous page, 'f' to filter, 's' to sort, 'r' to return to previous search results, or the dataset number to download.")
        
    async def get_dataset_user_input(self, current_page, total_pages):
        current_datasets = self.get_current_datasets(current_page)
        dataset_completer = WordCompleter([str(i) for i in range(1, len(current_datasets) + 1)] + ['n', 'p', 'f', 's', 'r'])
        prompt_session = PromptSession(completer=dataset_completer)
        user_input = await prompt_session.prompt_async('Enter your choice: ')
        return user_input

    def get_current_datasets(self, current_page):
        start_index = (current_page - 1) * self.page_size
        end_index = min(start_index + self.page_size, len(self.filtered_datasets))
        current_datasets = self.filtered_datasets[start_index:end_index]
        return current_datasets

    async def handle_dataset_user_choice(self):
        logger.info("Enter 'b' to go back to the filtered search results, 'm' to return to the main search results, or 'q' to quit.")
        user_input = await asyncio.get_event_loop().run_in_executor(None, input, "> ")
        if user_input.lower() == 'b':
            return
        elif user_input.lower() == 'm':
            self.filtered_datasets = self.main_search_datasets.copy()
            self.total_datasets = len(self.filtered_datasets)
            self.search_history = [(self.main_search_datasets.copy(), 1)]
        elif user_input.lower() == 'q':
            raise StopAsyncIteration
        else:
            logging.error("Invalid input. Please try again.")

    async def filter_dataset_search_results(self):
        logger.info("Enter the filter keyword:")
        filter_keyword = await asyncio.get_event_loop().run_in_executor(None, input, "> ")
        self.filtered_dataset_ids = set(dataset['id'] for dataset in self.filtered_datasets if filter_keyword.lower() in dataset['id'].lower())
        self.filtered_datasets = [dataset for dataset in self.filtered_datasets if dataset['id'] in self.filtered_dataset_ids]
        self.total_datasets = len(self.filtered_datasets)
        self.search_history.append((self.filtered_datasets.copy(), 1))
        if not self.filtered_datasets:
            logging.warning(f"No datasets found for the filter keyword '{filter_keyword}'.")

    async def sort_dataset_search_results(self):
        logger.info("Enter the sort criteria (e.g., 'downloads', 'likes', 'lastModified'):")
        sort_criteria = await asyncio.get_event_loop().run_in_executor(None, input, "> ")
        reverse = True
        if sort_criteria.startswith('-'):
            sort_criteria = sort_criteria[1:]
            reverse = False
        self.filtered_datasets.sort(key=lambda x: x.get(sort_criteria, 0), reverse=reverse)
        self.search_history.append((self.filtered_datasets.copy(), 1))
        logger.info(f"Search results sorted by '{sort_criteria}' in {'descending' if reverse else 'ascending'} order.")

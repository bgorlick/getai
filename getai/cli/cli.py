import argparse
from pathlib import Path
import asyncio
import logging
from aiohttp import ClientError

from getai.api import search_datasets, download_dataset, search_models, download_model

from getai.cli.utils import CLIUtils
from getai.core.dataset_search import AsyncDatasetSearch


# Configure logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_common_arguments(parser: argparse.ArgumentParser):
    """Add common arguments for dataset and model parsers."""
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--max-connections", type=int, default=5, help="Max connections"
    )


def define_subparsers(parser: argparse.ArgumentParser):
    """Define top-level subparsers for search and download commands."""
    subparsers = parser.add_subparsers(dest="mode", help="Mode of operation")

    # Search mode
    search_parser = subparsers.add_parser(
        "search", help="Search for models or datasets"
    )
    search_subparsers = search_parser.add_subparsers(
        dest="search_mode", help="Search mode"
    )

    # Search datasets
    search_datasets_parser = search_subparsers.add_parser(
        "datasets", help="Search for datasets"
    )
    search_datasets_parser.add_argument(
        "query", type=str, help="Search query for datasets"
    )
    search_datasets_parser.add_argument("--author", type=str, help="Filter by author")
    search_datasets_parser.add_argument(
        "--filter-criteria", type=str, help="Filter criteria"
    )
    search_datasets_parser.add_argument("--sort", type=str, help="Sort by")
    search_datasets_parser.add_argument("--direction", type=str, help="Sort direction")
    search_datasets_parser.add_argument("--limit", type=int, help="Limit results")
    search_datasets_parser.add_argument(
        "--full", action="store_true", help="Full dataset info"
    )
    add_common_arguments(search_datasets_parser)

    # Search models
    search_models_parser = search_subparsers.add_parser(
        "models", help="Search for models"
    )
    search_models_parser.add_argument("query", type=str, help="Search query for models")
    add_common_arguments(search_models_parser)

    # Download mode
    download_parser = subparsers.add_parser(
        "download", help="Download models or datasets"
    )
    download_subparsers = download_parser.add_subparsers(
        dest="download_mode", help="Download mode"
    )

    # Download dataset
    download_dataset_parser = download_subparsers.add_parser(
        "dataset", help="Download a dataset"
    )
    download_dataset_parser.add_argument(
        "identifier", type=str, help="Dataset identifier"
    )
    download_dataset_parser.add_argument(
        "--revision", type=str, help="Dataset revision"
    )
    download_dataset_parser.add_argument(
        "--full", action="store_true", help="Full dataset info"
    )
    add_common_arguments(download_dataset_parser)

    # Download model
    download_model_parser = download_subparsers.add_parser(
        "model", help="Download a model"
    )
    download_model_parser.add_argument("identifier", type=str, help="Model identifier")
    download_model_parser.add_argument(
        "--branch", type=str, default="main", help="Model branch"
    )
    download_model_parser.add_argument(
        "--clean", action="store_true", help="Start from scratch"
    )
    download_model_parser.add_argument(
        "--check", action="store_true", help="Check files"
    )
    add_common_arguments(download_model_parser)


async def main():
    """Main function for the GetAI CLI"""
    parser = argparse.ArgumentParser(description="GetAI CLI")
    define_subparsers(parser)
    parser.add_argument(
        "--hf-login", action="store_true", help="Log in using Hugging Face CLI"
    )
    args = parser.parse_args()

    logger.info("Parsed arguments: %s", args)

    if args.hf_login:
        logger.info("Logging in to Hugging Face CLI")
        CLIUtils.hf_login()
        return

    hf_token = CLIUtils.get_hf_token()

    if not args.mode:
        logger.error("Invalid mode. Please specify a valid mode.")
        parser.print_help()
        return

    try:
        if args.mode == "search":
            if args.search_mode == "datasets":
                logger.info("Searching datasets with query: %s", args.query)
                search_instance = AsyncDatasetSearch(
                    query=args.query,
                    output_dir=args.output_dir or Path.home() / ".getai" / "datasets",
                    max_connections=args.max_connections,
                    hf_token=hf_token,
                )
                await search_instance.display_dataset_search_results()
            elif args.search_mode == "models":
                logger.info("Searching models with query: %s", args.query)
                await search_models(
                    query=args.query,
                    hf_token=hf_token,
                    max_connections=args.max_connections,
                )
            else:
                logger.error(
                    "Invalid search subcommand. Please specify 'datasets' or 'models'."
                )
                parser.print_help()

        elif args.mode == "download":
            if args.download_mode == "dataset":
                logger.info("Downloading dataset: %s", args.identifier)
                await download_dataset(
                    identifier=args.identifier,
                    revision=args.revision,
                    full=args.full,
                    output_dir=args.output_dir,
                    max_connections=args.max_connections,
                )
            elif args.download_mode == "model":
                logger.info("Downloading model: %s", args.identifier)
                await download_model(
                    identifier=args.identifier,
                    branch=args.branch,
                    hf_token=hf_token,
                    max_connections=args.max_connections,
                    output_dir=args.output_dir,
                    clean=args.clean,
                    check=args.check,
                )
            else:
                logger.error(
                    "Invalid download subcommand. Please specify 'dataset' or 'model'."
                )
                parser.print_help()

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Closing operation...")
    except ClientError as e:
        logger.error("HTTP error during operation: %s", e)
    except asyncio.CancelledError:
        logger.info("Task cancelled during operation.")
    except ValueError as e:
        logger.error("Value error during operation: %s", e)
    except Exception as e:
        logger.error("Unexpected error during operation: %s", e)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Closing operation...")
    finally:
        pending_tasks = asyncio.all_tasks(loop)
        loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
        loop.close()

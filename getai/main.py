import argparse
import asyncio
import logging
from aiohttp import ClientError
from getai.api import search_datasets, download_dataset, search_models, download_model
from getai.utils import get_hf_token

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main function for the GetAI CLI"""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", help="Mode of operation")

    # Model mode
    model_parser = subparsers.add_parser("model", help="Download a model")
    model_parser.add_argument(
        "identifier", type=str, help="Model identifier on Hugging Face"
    )
    model_parser.add_argument(
        "--branch", nargs="?", const="main", default="main", help="Branch name"
    )
    model_parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory to save the model"
    )
    model_parser.add_argument(
        "--max-connections", type=int, default=5, help="Max connections for downloads"
    )
    model_parser.add_argument(
        "--clean", action="store_true", help="Start download from scratch"
    )
    model_parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the checksums of files after download",
    )

    # Dataset mode
    dataset_parser = subparsers.add_parser("dataset", help="Download a dataset")
    dataset_parser.add_argument(
        "identifier", type=str, help="Dataset identifier on Hugging Face"
    )
    dataset_parser.add_argument("--revision", type=str, help="Revision of the dataset")
    dataset_parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory to save the dataset"
    )
    dataset_parser.add_argument(
        "--max-connections", type=int, default=5, help="Max connections for downloads"
    )
    dataset_parser.add_argument(
        "--full", action="store_true", help="Fetch full dataset information"
    )

    # Search mode
    search_parser = subparsers.add_parser(
        "search", help="Search for models or datasets"
    )
    search_subparsers = search_parser.add_subparsers(
        dest="search_mode", help="Search mode"
    )

    # Model search mode
    model_search_parser = search_subparsers.add_parser(
        "model", help="Search for models"
    )
    model_search_parser.add_argument("query", type=str, help="Search query for models")
    model_search_parser.add_argument(
        "--max-connections", type=int, default=5, help="Max connections for searching"
    )

    # Dataset search mode
    dataset_search_parser = search_subparsers.add_parser(
        "dataset", help="Search for datasets"
    )
    dataset_search_parser.add_argument(
        "query", type=str, help="Search query for datasets"
    )
    dataset_search_parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory to save the dataset"
    )
    dataset_search_parser.add_argument(
        "--max-connections", type=int, default=5, help="Max connections for downloads"
    )
    dataset_search_parser.add_argument(
        "--author", type=str, help="Filter datasets by author or organization"
    )
    dataset_search_parser.add_argument(
        "--filter-criteria", type=str, help="Filter datasets based on tags"
    )
    dataset_search_parser.add_argument(
        "--sort", type=str, help="Property to use when sorting datasets"
    )
    dataset_search_parser.add_argument(
        "--direction", type=str, help="Direction to sort datasets"
    )
    dataset_search_parser.add_argument(
        "--limit", type=int, help="Limit the number of datasets fetched"
    )
    dataset_search_parser.add_argument(
        "--full", action="store_true", help="Fetch full dataset information"
    )

    # Token update
    parser.add_argument(
        "--update-token", type=str, help="Update the Hugging Face token"
    )

    args = parser.parse_args()
    hf_token = get_hf_token(update_token=args.update_token)
    if args.update_token:
        logger.info("Hugging Face token updated successfully.")

    if args.mode not in ["model", "dataset", "search"]:
        logger.error("Invalid mode. Please specify 'model', 'dataset', or 'search'.")
        return

    try:
        if args.mode == "search":
            if not args.search_mode:
                logger.error("Please specify the search mode (model or dataset).")
                return

            if args.search_mode == "model":
                await search_models(
                    query=args.query,
                    hf_token=hf_token,
                    max_connections=args.max_connections,
                )
            else:
                await search_datasets(
                    query=args.query,
                    hf_token=hf_token,
                    max_connections=args.max_connections,
                    output_dir=args.output_dir,
                    author=args.author,
                    filter_criteria=args.filter_criteria,
                    sort=args.sort,
                    direction=args.direction,
                    limit=args.limit,
                    full=args.full,
                )
        elif args.mode == "dataset":
            await download_dataset(
                identifier=args.identifier,
                hf_token=hf_token,
                max_connections=args.max_connections,
                output_dir=args.output_dir,
                revision=args.revision,
                full=args.full,
            )
        else:  # args.mode == 'model'
            await download_model(
                identifier=args.identifier,
                branch=args.branch,
                hf_token=hf_token,
                max_connections=args.max_connections,
                output_dir=args.output_dir,
                clean=args.clean,
                check=args.check,
            )
    except KeyboardInterrupt:
        logger.info("\nKeyboardInterrupt received. Closing operation...")
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
        logger.info("\nKeyboardInterrupt received. Closing operation...")
    finally:
        pending_tasks = asyncio.all_tasks(loop)
        loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
        loop.close()

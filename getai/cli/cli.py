import argparse
from pathlib import Path
import asyncio
import logging
from aiohttp import ClientError
import getai.api as api
from getai.cli.utils import CLIUtils

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path.home() / ".getai" / "models"
DEFAULT_MAX_CONNECTIONS = 5
DEFAULT_BRANCH = "main"


def add_common_arguments(parser: argparse.ArgumentParser):
    """Add common arguments for dataset and model parsers."""
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--max-connections",
        type=int,
        default=DEFAULT_MAX_CONNECTIONS,
        help="Max connections",
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
        "datasets", aliases=["dataset"], help="Search for datasets"
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
        "models", aliases=["model"], help="Search for models"
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
        "dataset", aliases=["datasets"], help="Download a dataset"
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
        "model", aliases=["models"], help="Download a model"
    )
    download_model_parser.add_argument("identifier", type=str, help="Model identifier")
    download_model_parser.add_argument(
        "--branch", type=str, default=DEFAULT_BRANCH, help="Model branch"
    )
    download_model_parser.add_argument(
        "--clean", action="store_true", help="Start from scratch"
    )
    download_model_parser.add_argument(
        "--check", action="store_true", help="Check files"
    )
    add_common_arguments(download_model_parser)

    # Alias model/models to download
    download_parser_model = subparsers.add_parser(
        "model", aliases=["models"], help="Download a model"
    )
    download_parser_model.add_argument("identifier", type=str, help="Model identifier")
    download_parser_model.add_argument(
        "--branch", type=str, default=DEFAULT_BRANCH, help="Model branch"
    )
    download_parser_model.add_argument(
        "--clean", action="store_true", help="Start from scratch"
    )
    download_parser_model.add_argument(
        "--check", action="store_true", help="Check files"
    )
    download_parser_model.set_defaults(mode="download", download_mode="model")


def set_defaults(args):
    """Set default values for output_dir, max_connections, and branch if not provided."""
    if not hasattr(args, "max_connections") or args.max_connections is None:
        args.max_connections = DEFAULT_MAX_CONNECTIONS

    if not hasattr(args, "output_dir") or args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR

    if not hasattr(args, "branch") or args.branch is None:
        args.branch = DEFAULT_BRANCH


async def main():
    """Main function for the GetAI CLI"""
    parser = argparse.ArgumentParser(description="GetAI CLI")
    define_subparsers(parser)
    parser.add_argument(
        "--hf-login", action="store_true", help="Log in using Hugging Face CLI"
    )
    args = parser.parse_args()

    logger.debug("Parsed arguments: %s", args)

    if args.hf_login:
        logger.info("Logging in to Hugging Face CLI")
        CLIUtils.hf_login()
        return

    hf_token = CLIUtils.get_hf_token()

    if not args.mode:
        logger.error("Invalid mode. Please specify a valid mode.")
        parser.print_help()
        return

    # Set default values for max_connections, output_dir, and branch
    set_defaults(args)

    # Log final values of the arguments
    logger.debug(
        "Final arguments: mode=%s, search_mode=%s, download_mode=%s, identifier=%s, branch=%s, output_dir=%s, max_connections=%s, hf_token=%s",
        args.mode,
        getattr(args, "search_mode", None),
        getattr(args, "download_mode", None),
        getattr(args, "identifier", None),
        args.branch,
        args.output_dir,
        args.max_connections,
        hf_token,
    )

    try:
        if args.mode == "search":
            if args.search_mode in ["datasets", "dataset"]:
                logger.info("Searching datasets with query: %s", args.query)
                await api.search_datasets(
                    query=args.query,
                    hf_token=hf_token,
                    max_connections=args.max_connections,
                    output_dir=args.output_dir,
                )
            elif args.search_mode in ["models", "model"]:
                logger.info("Searching models with query: %s", args.query)
                await api.search_models(
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
            if args.download_mode in ["dataset", "datasets"]:
                logger.info("Downloading dataset: %s", args.identifier)
                await api.download_dataset(
                    identifier=args.identifier,
                    revision=args.revision,
                    full=args.full,
                    hf_token=hf_token,
                    max_connections=args.max_connections,
                    output_dir=args.output_dir,
                )
            elif args.download_mode in ["model", "models"]:
                logger.info("Downloading model: %s", args.identifier)
                await api.download_model(
                    identifier=args.identifier,
                    branch=args.branch,
                    clean=args.clean,
                    check=args.check,
                    hf_token=hf_token,
                    max_connections=args.max_connections,
                    output_dir=args.output_dir,
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

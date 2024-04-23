# GetAI - v0.0.1 - Asynchronous AI Model and Dataset Downloader - github.com/bgorlick/getai/
""" main.py - Contains the main function for the GetAI CLI - Asynchronous AI Model and Dataset Downloader """
import argparse
import asyncio
import logging
from .utils import get_hf_token
from .dataset_downloader import AsyncDatasetDownloader
from .model_downloader import AsyncModelDownloader

logging.basicConfig(level=logging.INFO)


async def close_model_downloader(model_downloader):
    try:
        await model_downloader.close()
    except Exception as e:
        print(f"Error closing model_downloader: {e}")


async def close_dataset_downloader(dataset_downloader):
    try:
        await dataset_downloader.close_dataset_downloader()
    except Exception as e:
        print(f"Error closing dataset_downloader: {e}")


async def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation')

    # Model mode
    model_parser = subparsers.add_parser('model', help='Download a model')
    model_parser.add_argument('identifier', type=str, help='Model identifier on Hugging Face')
    model_parser.add_argument('--branch', nargs='?', const=True, default=False, help='Enable branch selection or specify a branch name')
    model_parser.add_argument('--output-dir', type=str, help='Directory to save the model', default=None)
    model_parser.add_argument('--max-retries', type=int, default=5, help='Max retries for downloads')
    model_parser.add_argument('--max-connections', type=int, default=5, help='Max simultaneous connections for downloads')
    model_parser.add_argument('--clean', action='store_true', help='Start download from scratch')
    model_parser.add_argument('--check', action='store_true', help='Validate the checksums of files after download')

    # Dataset mode
    dataset_parser = subparsers.add_parser('dataset', help='Download a dataset')
    dataset_parser.add_argument('identifier', type=str, help='Dataset identifier on Hugging Face')
    dataset_parser.add_argument('--revision', type=str, help='Revision of the dataset')
    dataset_parser.add_argument('--output-dir', type=str, help='Directory to save the dataset', default=None)
    dataset_parser.add_argument('--max-retries', type=int, default=5, help='Max retries for downloads')
    dataset_parser.add_argument('--max-connections', type=int, default=5, help='Max simultaneous connections for downloads')
    dataset_parser.add_argument('--full', action='store_true', help='Fetch full dataset information')

    # Search mode
    search_parser = subparsers.add_parser('search', help='Search for models or datasets')
    search_subparsers = search_parser.add_subparsers(dest='search_mode', help='Search mode')

    # Model search mode
    model_search_parser = search_subparsers.add_parser('model', help='Search for models')
    model_search_parser.add_argument('query', type=str, help='Search query for models')

    # Dataset search mode
    dataset_search_parser = search_subparsers.add_parser('dataset', help='Search for datasets')
    dataset_search_parser.add_argument('query', type=str, help='Search query for datasets')
    dataset_search_parser.add_argument('--author', type=str, help='Filter datasets by author or organization')
    dataset_search_parser.add_argument('--filter', type=str, help='Filter datasets based on tags')
    dataset_search_parser.add_argument('--sort', type=str, help='Property to use when sorting datasets')
    dataset_search_parser.add_argument('--direction', type=str, help='Direction in which to sort datasets')
    dataset_search_parser.add_argument('--limit', type=int, help='Limit the number of datasets fetched')
    dataset_search_parser.add_argument('--full', action='store_true', help='Fetch full dataset information')

    # Token update
    parser.add_argument("--update-token", type=str, help="Update the Hugging Face token")

    args = parser.parse_args()

    hf_token = get_hf_token(update_token=args.update_token)
    if args.update_token:
        print("Hugging Face token updated successfully.")

    if args.mode not in ['model', 'dataset', 'search']:
        logging.error("Invalid mode. Please specify 'model', 'dataset', or 'search'.")
        return

    if args.mode == 'search':
        if not args.search_mode:
            logging.error("Please specify the search mode (model or dataset).")
            return

        if args.search_mode == 'model':
            async with AsyncModelDownloader(
                max_retries=5,
                output_dir=None,
                max_connections=5,
                hf_token=hf_token
            ) as model_downloader:
                try:
                    await model_downloader.search_models(args.query)
                except KeyboardInterrupt:
                    print("\nKeyboardInterrupt received. Closing model downloader...")
                except Exception as e:
                    print(f"Error during model search: {e}")
        else:
            async with AsyncDatasetDownloader(
                max_retries=5,
                output_dir=None,
                max_connections=5,
                hf_token=hf_token
            ) as dataset_downloader:
                try:
                    await dataset_downloader.search_datasets(
                        query=args.query,
                        author=args.author,
                        filter=args.filter,
                        sort=args.sort,
                        direction=args.direction,
                        limit=args.limit,
                        full=args.full
                    )
                except KeyboardInterrupt:
                    print("\nKeyboardInterrupt received. Closing dataset downloader...")
                except Exception as e:
                    print(f"Error during dataset search: {e}")
    elif args.mode == 'dataset':
        async with AsyncDatasetDownloader(
            max_retries=args.max_retries,
            output_dir=args.output_dir,
            max_connections=args.max_connections,
            hf_token=hf_token
        ) as dataset_downloader:
            try:
                await dataset_downloader.download_dataset_info(args.identifier, args.revision, args.full)
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt received. Closing dataset downloader...")
            except Exception as e:
                print(f"Error during dataset download: {e}")
    else:  # args.mode == 'model'
        async with AsyncModelDownloader(
            max_retries=args.max_retries,
            output_dir=args.output_dir,
            max_connections=args.max_connections,
            hf_token=hf_token
        ) as model_downloader:
            try:
                await model_downloader.download_model(args.identifier, args.branch, args.clean, args.check)
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt received. Closing model downloader...")
            except Exception as e:
                print(f"Error during model download: {e}")


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Closing downloader...")
    finally:
        pending_tasks = asyncio.all_tasks(loop)
        loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
        loop.close()

from getai import download_dataset
import asyncio

async def download_dataset_example():
    await download_dataset(
        identifier="stanfordnlp/imdb",
        hf_token="None",
        max_connections=5,
        output_dir="datasets/stardfordnlp/imdb"
    )

if __name__ == "__main__":
    asyncio.run(download_dataset_example())

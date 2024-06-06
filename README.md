
# GetAI - The Easiest to Use AI Model Search & Download API and Interactive Console Tool

![PyPI](https://img.shields.io/pypi/v/getai)
![License](https://img.shields.io/github/license/bgorlick/getai)
![Build Status](https://img.shields.io/github/actions/workflow/status/bgorlick/getai/build.yml)

GetAI is a powerful API library and command-line tool for AI Models and Datasets. It simplifies the process of searching, downloading, and exploring AI models and datasets from various sources like Hugging Face and other platforms. With GetAI, you can easily find and download the models and datasets you need with a simple import statement and minimal lines of code, without the hassle of navigating through multiple websites and repositories.

- **Easy to Download and Use**
  - Many tools force you into their controlled ecosystem. GetAI liberates you and your AI agents.
  - Install: `pip install getai`
  - Search Models: `getai search model <query>`
  - Search Datasets: `getai search dataset <query`
  - Download a Model: `getai model author/model_name`  Working Example: `getai model meta-llama/Llama-2-7b-hf`
 
- **Two lines of code to add AI Model Search or Download) to Your Project**
  ```python
  from getai import search_datasets, download_dataset; import asyncio
  asyncio.run(search_datasets("sentiment analysis", hf_token=None, max_connections=5, output_dir="datasets"))
  ```
- **Interactive Search for Models and Datasets**
  - Powerful fully interactive console UX for search (showing sizes, branches, last updated, and more)
  - Accelerates navigation and model discovery

- **Detailed Information**
  - Displays sizes and last modified dates
  - Enables sorting results in various ways

- **Future-Ready Design**
  - Quickly find the most current models and datasets
  - Designed to simplify integration with your AI agents in searching and downloading datasets and models for fully autonomous projects

## Table of Contents
- [Why GetAI?](#why-getai)
- [Features of GetAI](#features-of-getai)
- [Installation](#installation)
- [Usage](#usage)
  - [Search for AI Models](#search-for-ai-models)
  - [Search for Datasets](#search-for-datasets)
  - [Download a Model](#download-a-model)
  - [Download a Dataset](#download-a-dataset)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Support and Feedback](#support-and-feedback)
- [Sources of Inspiration](#sources-of-inspiration)
- [Using GetAI as a Library](#using-getai-as-a-library)

## Why GetAI?

The advent of large language models has led many companies to release open-source versions of their pre-trained foundation models. This has enabled anyone to download and run AI models locally, rather than relying on third-party API services. However, the process of finding, downloading, and setting up these models can be cumbersome and not always straightforward, especially for new users.

GetAI aims to simplify this process, not only for human developers but also for AI agents that need a simple tool to search and download models and datasets easily. With GetAI, you can quickly find the models and datasets you need, download them asynchronously, and start exploring and using them in your projects.

## Features of GetAI

- **Asynchronous Downloads**: GetAI allows you to download models asynchronously, making efficient use of network resources and saving you time.
- **Searching for AI Models**: You can easily search for AI models using various filters such as name, last updated date, and other attributes. GetAI provides a user-friendly interface to find the models that best suit your needs.
- **Searching for Datasets**: GetAI also enables you to search for datasets by name and download them easily. You can quickly find and access the datasets you require for training or evaluating your AI models.
- **Multiple Sources**: GetAI supports downloading models and datasets from multiple sources, including Hugging Face, TensorFlow Hub, and other platforms. You can access a wide range of resources from different providers through a single tool.
- **Flexible Configuration**: GetAI allows you to configure sources and authentication through a *`config.yaml`* file (default location: *`/home/.getai/config.yaml`*). You can easily set up your credentials and preferences to streamline your workflow.
- **Interactive CLI**: GetAI provides an easy-to-use command-line interface with interactive features such as branch selection and progress display. You can navigate through the available options and monitor the download progress seamlessly.
- **API Access**: Easily integrate GetAI functionalities into your own applications using the provided API.

## Installation

You can install *`getai`* using pip:

```bash
pip install getai
```

## Usage

GetAI provides a simple and intuitive command-line interface. Here are some examples of how you can use GetAI:

### Search for AI Models

```bash
getai search model <query> [--author <author>] [--filter <filter>] [--sort <sort>] [--direction <direction>] [--limit <limit>] [--full]
```

This command allows you to search for AI models based on the provided query. You can use various options to refine your search results:
- *`--author`*: Filter models by author or organization.
- *`--filter`*: Filter models based on tags.
- *`--sort`*: Property to use when sorting models.
- *`--direction`*: Direction in which to sort models.
- *`--limit`*: Limit the number of models fetched.
- *`--full`*: Fetch full model information.

Example:

```bash
getai search model "text-generation" --sort downloads --direction -1 --limit 10
```

Sample output:
```
Search results for 'text-generation' (Page 1 of 1, Total: 10):
1. gpt2 by OpenAI (openai/gpt2) (Size: 548.09 MB)
2. distilgpt2 by HuggingFace (distilgpt2) (Size: 353.75 MB)
3. gpt2-large by OpenAI (openai/gpt2-large) (Size: 1.50 GB)
...
Enter 'n' for the next page, 'p' for the previous page, 'f' to filter, 's' to sort, 'r' to return to previous search results, or the model number to download.
```

### Search for Datasets

```bash
getai search dataset <query> [--author <author>] [--filter <filter>] [--sort <sort>] [--direction <direction>] [--limit <limit>] [--full]
```

This command enables you to search for datasets based on the provided query. You can use various options to refine your search results:
- *`--author`*: Filter datasets by author or organization.
- *`--filter`*: Filter datasets based on tags.
- *`--sort`*: Property to use when sorting datasets.
- *`--direction`*: Direction in which to sort datasets.
- *`--limit`*: Limit the number of datasets fetched.
- *`--full`*: Fetch full dataset information.

Example:

```bash
getai search dataset "sentiment analysis" --filter language:en --sort downloads --direction -1 --limit 5
```

Sample output:
```
Search results for 'sentiment analysis' (Page 1 of 1, Total: 5):
1. imdb by andrew-maas (andrew-maas/imdb) (Size: 80.23 MB)
2. twitter_sentiment by nlp-with-deeplearning (nlp-with-deeplearning/twitter_sentiment) (Size: 63.15 MB)
3. sst2 by glue (glue/sst2) (Size: 7.09 MB)
...
Enter 'n' for the next page, 'p' for the previous page, 'f' to filter, 's' to sort, 'r' to return to previous search results, or the dataset number to download.
```

### Download a Model

```bash
getai model <identifier> [--branch <branch>] [--output-dir <output-dir>] [--max-retries <max-retries>] [--max-connections <max-connections>] [--clean] [--check]
```

This command allows you to download a specific model by providing its identifier. You can use various options to customize the download process:
- *`--branch`*: Specify a branch name or enable branch selection.
- *`--output-dir`*: Directory to save the model.
- *`--max-retries`*: Max retries for downloads.
- *`--max-connections`*: Max simultaneous connections for downloads.
- *`--clean`*: Start download from scratch.
- *`--check`*: Validate the checksums of files after download.

Example:

```bash
getai model meta-llama/Llama-2-7b-hf --branch main --output-dir models/gpt2 --max-retries 3 --max-connections 5
```

### Download a Dataset

```bash
getai dataset <identifier> [--revision <revision>] [--output-dir <output-dir>] [--max-retries <max-retries>] [--max-connections <max-connections>] [--full]
```

This command enables you to download a specific dataset by providing its identifier. You can use various options to customize the download process:
- *`--revision`*: Revision of the dataset.
- *`--output-dir`*: Directory to save the dataset.
- *`--max-retries`*: Max retries for downloads.
- *`--max-connections`*: Max simultaneous connections for downloads.
- *`--full`*: Fetch full dataset information.

Example:

```bash
getai dataset glue/sst2 --revision main --output-dir datasets/sst2 --max-retries 3 --max-connections 5
```

For more detailed usage instructions and additional options, please refer to the GetAI documentation.

## Configuration

GetAI uses a *`config.yaml`* file to store configuration settings such as API tokens and other preferences. By default, the configuration file is located at *`/home/.getai/config.yaml`*. However, we recommend using the *`huggingface-cli login`* command to securely set up your Hugging Face token or setting the *`HF_TOKEN`* environment variable.

Example of setting the environment variable:

```bash
export HF_TOKEN=your_huggingface_token_here
```

Here's an example of a *`config.yaml`* file:

```yaml
hf_token: your_huggingface_token_here
```

Replace *`your_huggingface_token_here`* with your actual Hugging Face token.

## Using GetAI as a Library

GetAI isn't just a command-line tool; it's also a powerful Python library for searching and downloading datasets and models from Hugging Face. This guide shows you how to leverage GetAI's capabilities programmatically in your Python applications.

### Installation

First, install the GetAI package:

```bash
pip install getai
```

### Usage Examples

Below are examples of how to use the main functions provided by GetAI. These examples demonstrate how to search for and download datasets and models programmatically.

#### Searching for Datasets

Let's say you're working on a project related to sentiment analysis and you want to find relevant datasets. Here's how you can do it:

```python
from getai import search_datasets
import asyncio

async def search_datasets_example():
    await search_datasets(
        query="sentiment analysis",
        hf_token="your_huggingface_token",
        max_connections=5,
        output_dir="datasets"
    )

if __name__ == "__main__":
    asyncio.run(search_datasets_example())
```

#### Downloading a Dataset

Once you've found the dataset you need, downloading it is simple. For example, to download the SST-2 dataset from the GLUE benchmark:

```python
from getai import download_dataset
import asyncio

async def download_dataset_example():
    await download_dataset(
        identifier="glue/sst2",
        hf_token="your_huggingface_token",
        max_connections=5,
        output_dir="datasets/sst2"
    )

if __name__ == "__main__":
    asyncio.run(download_dataset_example())
```

#### Searching for Models

Imagine you need a model for text generation. Here's how you can search for it:

```python
from getai import search_models
import asyncio

async def search_models_example():
    await search_models(
        query="text-generation",
        hf_token="your_huggingface_token",
        max_connections=5
    )

if __name__ == "__main__":
    asyncio.run(search_models_example())
```

#### Downloading a Model

After finding the model, downloading it is straightforward. For instance, to download the GPT-2 model:

```python
from getai import download_model
import asyncio

async def download_model_example():
    await download_model(
        identifier="openai/gpt2",
        branch="main",
        hf_token="your_huggingface_token",
        max_connections=5,
        output_dir="models/gpt2"
    )

if __name__ == "__main__":
    asyncio.run(download_model_example())
```

Replace *`your_huggingface_token`* with your actual Hugging Face token.

### Detailed Function Usage

To give you a deeper understanding, here are the detailed descriptions and usages of the core functions provided by GetAI.

#### search_datasets

```python
async def search_datasets(
    query, hf_token=None, max_connections=5, output_dir=None, **kwargs
):
    """
    Search datasets on Hugging Face based on a query.
    
    Args:
        query (str): The search query.
        hf_token (str): Hugging Face token.
        max_connections (int): Maximum number of concurrent connections.
        output_dir (Path): Directory to save search results.
        **kwargs: Additional keyword arguments for filtering search results.
    """
```

#### download_dataset

```python
async def download_dataset(
    identifier, hf_token=None, max_connections=5, output_dir=None, **kwargs
):
    """
    Download a dataset from Hugging Face by its identifier.
    
    Args:
        identifier (str): The dataset identifier.
        hf_token (str): Hugging Face token.
        max_connections (int): Maximum number of concurrent connections.
        output_dir (Path): Directory to save the dataset.
        **kwargs: Additional keyword arguments for dataset download.
    """
```

#### search_models

```python
async def search_models(
    query, hf_token=None, max_connections=5, **kwargs
):
    """
    Search models on Hugging Face based on a query.
    
    Args:
        query (str): The search query.
        hf_token (str): Hugging Face token.
        max_connections (int): Maximum number of concurrent connections.
        **kwargs: Additional keyword arguments for filtering search results.
    """
```

#### download_model

```python
async def download_model(
    identifier, branch="main", hf_token=None, max_connections=5, output_dir=None, **kwargs
):
    """
    Download a model from Hugging Face by its identifier and branch.
    
    Args:
        identifier (str): The model identifier.
        branch (str): The branch name.
        hf_token (str): Hugging Face token.
        max_connections (int): Maximum number of concurrent connections.
        output_dir (Path): Directory to save the model.
        **kwargs: Additional keyword arguments for model download.
    """
```

## Contributing

Contributions to GetAI are welcome! If you would like to contribute to the project, please follow the guidelines outlined in the *`CONTRIBUTING.md`* file. You can help improve GetAI by reporting issues, suggesting new features, or submitting pull requests.

## License

GetAI is released under the MIT License with attribution to the author, Ben Gorlick (github.com/bgorlick). Please see the *`LICENSE`* file for more details.

(c) 2023-2024 Ben Gorlick github.com/bgorlick

## Support and Feedback

If you encounter any issues, have questions, or would like to provide feedback, please open an issue on the GetAI GitHub repository. We appreciate your input and will do our best to assist you.

Thank you for using GetAI! We hope it simplifies your workflow and enhances your experience with AI models and datasets.

## Sources of Inspiration

This project started as an attempt to create a completely asynchronous port of oobagooba's [text-generation-webui](https://github.com/oobabooga/text-generation-webui) model downloading script. His script at the time operated with a multithreaded design and I wanted to explore building an asynchronous version. Credits go entirely to him for the initial approach, pagination methods, and parsing logic for a variety of the file types.

## Thank you to everyone who continues to advance the AI and ML development!

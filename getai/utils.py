""" utils.py - Contains utility functions for the GetAI CLI, AsyncModelDownloader, and AsyncDatasetDownloader classes."""

import os
from pathlib import Path
import logging
import yaml
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def convert_to_bytes(size_str):
    """Convert size string like '2.3 GB' or '200 MB' to bytes."""
    try:
        size_units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
        size, unit = size_str.split()
        return int(float(size) * size_units[unit])
    except Exception as e:
        logging.exception("Error converting size to bytes: %s", e)
        raise


async def interactive_branch_selection(branches):
    """Prompt user to select a branch interactively from a list."""
    try:
        branch_completer = WordCompleter(branches, ignore_case=True)
        session = PromptSession(completer=branch_completer)
        selected_branch = await session.prompt_async("Select a branch [Press TAB]: ")
        return selected_branch if selected_branch in branches else "main"
    except Exception as e:
        logging.exception("Error during interactive branch selection: %s", e)
        raise


def get_hf_token(update_token=None):
    """Retrieve or update Hugging Face token securely."""
    try:
        getai_config_file = Path.home() / ".getai" / "getai_config.yaml"

        if update_token:
            logging.info("Updating Hugging Face token in ~/.getai/getai_config.yaml.")
            getai_config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(getai_config_file, "w", encoding="utf-8") as f:
                yaml.dump({"hf_token": update_token}, f)
            return update_token

        hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            logging.info("Using Hugging Face token from environment variable.")
            return hf_token

        hf_token_file = Path.home() / ".huggingface" / "token"
        if hf_token_file.exists():
            with open(hf_token_file, "r", encoding="utf-8") as f:
                logging.info("Using Hugging Face token from ~/.huggingface/token.")
                return f.read().strip()

        if getai_config_file.exists():
            with open(getai_config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                logging.info(
                    "Using Hugging Face token from ~/.getai/getai_config.yaml."
                )
                return config.get("hf_token")

        logging.warning("No Hugging Face token found. Prompting user for input.")
        hf_token = input("Enter your Hugging Face token: ")

        logging.info("Saving Hugging Face token to ~/.getai/getai_config.yaml.")
        getai_config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(getai_config_file, "w", encoding="utf-8") as f:
            yaml.dump({"hf_token": hf_token}, f)

        return hf_token
    except Exception as e:
        logging.exception("Error retrieving or updating Hugging Face token: %s", e)
        raise

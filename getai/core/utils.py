# getai/core/utils.py - GetAI utility functions for the core functionality.

import os
from pathlib import Path
import logging
import subprocess

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.ERROR
)


class CoreUtils:
    @staticmethod
    def convert_to_bytes(size_str):
        """Convert size string like '2.3 GB' or '200 MB' to bytes."""
        try:
            size_units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
            size, unit = size_str.split()
            return int(float(size) * size_units[unit])
        except Exception as e:
            logging.exception("Error converting size to bytes: %s", e)
            raise

    @staticmethod
    async def interactive_branch_selection(branches):
        """Prompt user to select a branch interactively from a list."""
        try:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.completion import WordCompleter

            branch_completer = WordCompleter(branches, ignore_case=True)
            session = PromptSession(completer=branch_completer)
            selected_branch = await session.prompt_async(
                "Select a branch [Press TAB]: "
            )
            return selected_branch if selected_branch in branches else "main"
        except Exception as e:
            logging.exception("Error during interactive branch selection: %s", e)
            raise

    @staticmethod
    def get_hf_token():
        """Retrieve the Hugging Face token securely from environment variables or the CLI."""
        try:
            hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")
            if hf_token:
                logging.info("Using Hugging Face token from environment variable.")
                return hf_token

            hf_token_file = Path.home() / ".huggingface" / "token"
            if hf_token_file.exists():
                with open(hf_token_file, "r", encoding="utf-8") as f:
                    logging.info("Using Hugging Face token from ~/.huggingface/token.")
                    return f.read().strip()

            hf_token = CoreUtils.get_hf_token_from_cli()
            if hf_token:
                logging.info("Using Hugging Face token from Hugging Face CLI.")
                return hf_token

            raise ValueError(
                "No Hugging Face token found. Please log in using the Hugging Face CLI."
            )
        except Exception as e:
            logging.exception("Error retrieving Hugging Face token: %s", e)
            raise

    @staticmethod
    def get_hf_token_from_cli():
        """Retrieve Hugging Face token using the CLI."""
        token_file = os.path.expanduser("~/.cache/huggingface/token")
        try:
            with open(token_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            logging.error(
                "Hugging Face token file not found. Please log in using `huggingface-cli login`."
            )
            return None
        except Exception as e:
            logging.exception("Error retrieving Hugging Face token from file: %s", e)
            return None

    @staticmethod
    def hf_login():
        """Log in using Hugging Face CLI."""
        try:
            result = subprocess.run(
                ["huggingface-cli", "login"], check=True, capture_output=True, text=True
            )
            logging.info("Hugging Face CLI login successful: %s", result.stdout)
        except subprocess.CalledProcessError as e:
            logging.error("Hugging Face CLI login failed: %s", e.stderr)
        except FileNotFoundError:
            logging.error(
                "Hugging Face CLI not found. Please install it and try again."
            )
        except Exception as e:
            logging.exception("Unexpected error during Hugging Face CLI login: %s", e)


__all__ = [
    "CoreUtils",
    "convert_to_bytes",
    "interactive_branch_selection",
    "get_hf_token",
    "get_hf_token_from_cli",
    "hf_login",
]

convert_to_bytes = CoreUtils.convert_to_bytes
interactive_branch_selection = CoreUtils.interactive_branch_selection
get_hf_token = CoreUtils.get_hf_token
get_hf_token_from_cli = CoreUtils.get_hf_token_from_cli
hf_login = CoreUtils.hf_login

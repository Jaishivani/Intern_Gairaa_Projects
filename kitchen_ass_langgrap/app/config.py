import os

from dotenv import load_dotenv


load_dotenv()


def get_required_setting(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}"
        )

    return value


def get_nvidia_model() -> str:
    return os.getenv(
        "NVIDIA_MODEL",
        "meta/llama-3.3-70b-instruct",
    ).strip()
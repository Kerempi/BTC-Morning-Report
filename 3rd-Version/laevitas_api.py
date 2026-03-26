import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

DEFAULT_BASE_URL = "https://api.laevitas.ch"
ENV_PATH = Path(__file__).with_name(".env")


def load_dotenv(path: Path = ENV_PATH) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
        if key:
            values[key] = value
    return values


def get_settings() -> Dict[str, str]:
    load_dotenv()
    api_key = os.environ.get("LAEVITAS_API_KEY", "").strip()
    base_url = os.environ.get("LAEVITAS_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    default_endpoint = os.environ.get("LAEVITAS_DEFAULT_ENDPOINT", "").strip()
    return {
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
        "default_endpoint": default_endpoint,
    }


def _normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.strip()
    if not endpoint:
        raise ValueError("Endpoint is required.")
    return endpoint if endpoint.startswith("/") else f"/{endpoint}"


def laevitas_get(endpoint: str, params: Optional[Dict[str, str]] = None, timeout: int = 30, retries: int = 2):
    settings = get_settings()
    if not settings["api_key"]:
        raise RuntimeError(
            "LAEVITAS_API_KEY is empty. Add it to the .env file in the project root."
        )

    url = f"{settings['base_url']}{_normalize_endpoint(endpoint)}"
    headers = {"apikey": settings["api_key"]}

    try:
        import requests

        last_error = None
        for attempt in range(retries + 1):
            try:
                session = requests.Session()
                session.trust_env = False
                response = session.get(url, headers=headers, params=params, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
                if attempt >= retries:
                    raise
                time.sleep(1.25 * (attempt + 1))
        raise last_error
    except ModuleNotFoundError:
        import urllib.parse
        import urllib.request

        last_error = None
        for attempt in range(retries + 1):
            try:
                query = urllib.parse.urlencode(params or {})
                final_url = url if not query else f"{url}?{query}"
                request = urllib.request.Request(final_url, headers=headers, method="GET")
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    body = response.read().decode("utf-8")
                return json.loads(body)
            except Exception as exc:
                last_error = exc
                if attempt >= retries:
                    raise
                time.sleep(1.25 * (attempt + 1))
        raise last_error

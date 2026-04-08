import argparse
import json
import sys

from laevitas_api import get_settings, laevitas_get


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a GET request against the Laevitas API.")
    parser.add_argument("endpoint", nargs="?", help="Endpoint path, e.g. /v2/...")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Query parameter in key=value form. Repeatable.",
    )
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
    args = parser.parse_args()

    settings = get_settings()
    endpoint = args.endpoint or settings["default_endpoint"]
    if not endpoint:
        print("No endpoint provided. Pass one on the command line or set LAEVITAS_DEFAULT_ENDPOINT in .env.", file=sys.stderr)
        return 2

    params = {}
    for item in args.param:
        if "=" not in item:
            print(f"Invalid --param value: {item}", file=sys.stderr)
            return 2
        key, value = item.split("=", 1)
        params[key] = value

    try:
        data = laevitas_get(endpoint, params=params or None, timeout=args.timeout)
    except Exception as exc:
        print(f"Laevitas request failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(data, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

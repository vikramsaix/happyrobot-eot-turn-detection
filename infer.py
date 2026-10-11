from __future__ import annotations

import argparse
import json
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the local turn detector API.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--text", required=True)
    parser.add_argument("--pause-ms", type=float, default=500)
    parser.add_argument("--stability-ms", type=float, default=250)
    parser.add_argument("--language", default="en")
    args = parser.parse_args()
    body = json.dumps({"text": args.text, "pause_ms": args.pause_ms,
                       "stability_ms": args.stability_ms, "language": args.language}).encode()
    request = urllib.request.Request(args.url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3) as response:
        print(json.dumps(json.loads(response.read()), indent=2))

if __name__ == "__main__":
    main()

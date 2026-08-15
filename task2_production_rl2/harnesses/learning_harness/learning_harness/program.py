# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.0"]
# ///
"""One-shot harness program; all model traffic goes through Verifiers interception."""

import argparse
import asyncio

from openai import AsyncOpenAI


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system-prompt", default="")
    return parser.parse_args()


async def main() -> None:
    args = arguments()
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": args.prompt})
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    await client.chat.completions.create(model=args.model, messages=messages)


if __name__ == "__main__":
    asyncio.run(main())

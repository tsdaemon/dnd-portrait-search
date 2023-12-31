import base64
import logging
from pathlib import Path

import backoff
from loguru import logger
from openai import AsyncOpenAI, BadRequestError, RateLimitError


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class OpenAIClient:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)

    @backoff.on_exception(backoff.expo, RateLimitError, max_tries=10, backoff_log_level=logging.INFO)
    async def make_image_query(self, query: str, image_path: Path) -> str:
        base64_image = encode_image(image_path)

        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": query},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                n=1,
                seed=43,
                max_tokens=2000,
            )

            return response.choices[0].message.content or ""

        except BadRequestError as e:
            logger.exception("OpenAI bad request", exc_info=True)
            if "You uploaded an unsupported image" in e.message:
                # We just ignore bad images for now
                return ""
            else:
                raise

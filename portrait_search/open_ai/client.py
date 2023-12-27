import base64
from pathlib import Path
import backoff
from openai import AsyncOpenAI, RateLimitError


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class OpenAIClient:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)

    @backoff.on_exception(backoff.expo, RateLimitError, max_tries=5)
    async def make_image_query(self, query: str, image_path: Path) -> str:
        base64_image = encode_image(image_path)

        response = await self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            n=1,
            seed=43,
            max_tokens=2000,
        )

        return response.choices[0].message.content or ""

"""
完成与豆包的交互
"""
import base64
import json
import mimetypes  # 用于自动判断图片类型

from volcenginesdkarkruntime import Ark

import settings
from src.utils import logger_config

logger = logger_config.get_logger(__name__)


def image_to_base64(image_path):
    """
    将图像文件转换为 base64 数据 URL
    """
    # Determine the MIME type of the image
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        # Fallback or raise an error if MIME type can't be guessed
        # For common types, you can hardcode or have a small mapping
        if image_path.lower().endswith(('.jpg', '.jpeg')):
            mime_type = 'image/jpeg'
        elif image_path.lower().endswith('.png'):
            mime_type = 'image/png'
        else:
            raise ValueError(f"Could not determine MIME type for {image_path}")

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{encoded_string}"


class DoubaoVisClient:
    """
    构建豆包聊天模型
    """

    def __init__(self):
        self.client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=settings.ARK_API_KEY,
        )

    def is_image_relevant(
        self,
        query: str,
        options: dict,
        image: dict,
    ) -> dict:
        """
        判断 image 是否与 query, options相关

        image 键包含 title, path, source_type, similarity
        """
        system_prompt = """
            Your task: Assess if a provided image and its title are helpful for selecting the correct answer to a given multiple-choice question.
            Input: Question, 4 options (A, B, C, D), image, image title.
            Focus on helpfulness for choosing an option, NOT on answering the question directly.

            Consider if the image/title:
            - Directly depicts elements from the question/options.
            - Provides useful context.
            - Is itself relevant.
            - Could be misleading.
            - Is completely unrelated.

            Output MUST be STRICTLY in this JSON format:
            Don't include DOUBLE QUOTES in the values.
            Don't include DOUBLE QUOTES in the values.
            Don't include DOUBLE QUOTES in the values.
            {
                "analysis": "string",
                "relevant": boolean,
                "score": float
            }

            Definitions:
            - `analysis`: Your reasoning explaining how the image/title relates (or doesn't) to the question/options, justifying your `relevant` and `score` values.
            - `relevant`: `true` if image/title helps choose the correct option or understand context; `false` if unhelpful, misleading, or unrelated.
            - `score`: Your confidence (0.0-1.0) in your `relevant` assessment.
        """
        user_prompt = f"""
            Query: {query}
            Options: {options}
            Image Title: {image["title"]}
        """
        encoded_image = image_to_base64(image["path"])

        response = self._generate_response(system_prompt, user_prompt,
                                           encoded_image)
        # 去除转义字符, 防止 json.loads 报错
        response = response.replace("\\", "")
        # logger.info("is_image_relevant response: %s", response)

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error("JSON decode error: %s", e)
            return {
                "analysis": "Invalid JSON response",
                "relevant": False,
                "score": 0.0
            }

    def _generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        encoded_image: str,
        **args,
    ) -> str:
        params = {
            "model": settings.ARK_VISION_ENDPOINT_ID,
            "temperature": 0.3,
            "max_tokens": 1000,
        }
        params.update(args)

        completion = self.client.chat.completions.create(
            messages=[{
                "role": "system",
                "content": system_prompt
            }, {
                "role":
                    "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": encoded_image
                        },
                    },
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                ],
            }],
            **params,
        )

        return completion.choices[0].message.content


def main():
    """
    主函数
    """
    pass


if __name__ == "__main__":
    main()

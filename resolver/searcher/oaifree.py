import json
import requests
from cxapi.schema import QuestionModel
from . import SearcherBase, SearcherResp
from logger import Logger


class OaifreeSearcherAPI(SearcherBase):
    """Oaifree  在线答题器"""

    config: dict

    def __init__(self, **config) -> None:
        super().__init__()
        self.config = config
        self.logger = Logger("OaifreeSearcherAPI")

    def invoke(self, question: QuestionModel) -> SearcherResp:
        # 将选项从 JSON 转换成人类易读的形式
        options_str = ""
        if question.options is not None:
            options_str = "\n\u9009\u9879\uff1a\n"
            if isinstance(question.options, dict):
                for k, v in question.options.items():
                    options_str += f"{k}. {v};"
            elif isinstance(question.options, list):
                for v in question.options:
                    options_str += f"{v};"

        # 生成请求内容
        prompt = self.config["prompt"].format(
            type=question.type.name,
            value=question.value,
            options=options_str,
        )

        system_prompt = self.config.get("system_prompt", "")
        if not system_prompt:
            raise ValueError("system_prompt is required in the configuration.")

        self.logger.info(
            f"\u4ece {self.config['prompt']} \u751f\u6210\u63d0\u95ee\uff1a{prompt}"
        )

        # 构造请求的 JSON 数据
        request_data = {
            "model": self.config.get("model", "gpt-4o-mini"),
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
        }

        # 发出请求
        try:
            response = requests.post(
                url=f"{self.config['base_url']}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config['api_key']}",
                    "Content-Type": "application/json",
                },
                json=request_data,
            )

            # 检查响应状态
            response.raise_for_status()
            response_json = response.json()

            # 替换 response_text 为 choices 中 message 的 content
            response_text = response_json["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as err:
            return SearcherResp(-500, str(err), self, question.value, None)

        # 单选题需要进一步预处理 AI 返回结果，以使 QuestionResolver 能正确命中
        if question.type.value == 0:
            response_text = response_text.strip()
            for k, v in question.options.items():
                if response_text.startswith(f"{k}.") or (v in response_text):
                    response_text = v
                    break

        # 多选题处理
        elif question.type.value == 1:
            selected_answers = [v for k, v in question.options.items() if v in response_text]
            response_text = "#".join(selected_answers)  # 将所有选中的答案以 '#' 分隔

        self.logger.info(f"\u8fd4\u56de\u7ed3\u679c\uff1a{response_text}")
        return SearcherResp(0, "", self, question.value, response_text)

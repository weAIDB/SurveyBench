# embedding_service
from openai import OpenAI
import numpy as np
from tqdm import tqdm


API_KEY = "sk-YROcKnB4CrsjqX8x3lPfzLUwd6mxejffVEqxgy1wySxoIE7p"
BAIDU_API_KEY = "sk-9333c8120e384860ba15973144a06201"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
QWEN_MODEL = "qwen-flash"
BASE_URL="https://api.aiaiapi.com/v1/"
DEFAULT_HEADERS={"x-foo": "true"}
DIMENSION = 1536


class EmbeddingService:
    def __init__(self, api_key=API_KEY, base_url=BASE_URL):
        """
        初始化远程 embedding 服务
        api_key: 服务端密钥
        base_url: 可选，如果是官方 OpenAI，不填
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def get_embedding(self, texts, batch_size=None, show_progress_bar=False):
        """
        支持单条或多条文本的 embedding
        texts: str 或 List[str]
        batch_size: 可选，设置为 None 表示一次性发送；设置数字则分批请求
        show_progress_bar: 如果需要 tqdm 进度条可设为 True
        返回：np.ndarray (shape: [num_texts, dim])
        """
        if isinstance(texts, str):
            texts = [texts]  # 统一成列表

        # 不分批的情况：一次请求所有文本
        if batch_size is None:
            resp = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts
            )
            return np.array([d.embedding for d in resp.data], dtype=np.float32)

        # 分批请求
        all_embeddings = []
        iterator = range(0, len(texts), batch_size)
        if show_progress_bar:
            iterator = tqdm(iterator, desc="Embedding batches")

        for start in iterator:
            batch = texts[start:start + batch_size]
            resp = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch
            )
            batch_emb = [d.embedding for d in resp.data]
            all_embeddings.extend(batch_emb)

        return np.array(all_embeddings, dtype=np.float32)


# chat_service
class ChatService:
    def __init__(self, api_key=API_KEY, base_url=BASE_URL, default_headers=DEFAULT_HEADERS):
        """
        初始化远程 Chat 服务
        api_key: 远程服务密钥
        base_url: 可选，第三方网关 URL
        default_headers: 可选，网关要求的自定义 header
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers
        )
        # # 如果需要自定义 header
        # if default_headers:
        #     self.client.default_headers = default_headers

    def send_message(self, messages):
        """
        messages: list of dict, [{"role": "user", "content": "..."}]
        返回：模型的文本回答
        """
        resp = self.client.chat.completions.create(
            model=LLM_MODEL,   # 可替换成你网关支持的版本
            messages=messages
        )
        return resp.choices[0].message.content

if __name__ == "__main__":
    # 测试调用
    text = "Hello, this is a test message."
    model = ChatService()
    message = [{"role": "user", "content": text}]
    response = model.send_message(message)
    print(response)
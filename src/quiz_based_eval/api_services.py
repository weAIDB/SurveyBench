# embedding_service
from openai import OpenAI
import numpy as np
from tqdm import tqdm


LLM_API_KEY = None
EMB_API_KEY = None
EMBEDDING_MODEL = None
LLM_MODEL = None
LLM_URL=None
EMB_URL=None
DIMENSION = None


class EmbeddingService:
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """
        初始化远程 embedding 服务
        api_key: 服务端密钥
        base_url: 可选，如果是官方 OpenAI，不填
        """
        api_key = api_key or EMB_API_KEY
        base_url = base_url or EMB_URL

        if not api_key or not base_url:
            raise ValueError(
                "请提供 api_key 和 base_url")
        print("API_KEY:", EMB_API_KEY)
        print("base_url:", base_url)
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
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """
        初始化远程 LLM 服务
        api_key: 远程服务密钥
        base_url: 网关 URL
        """
        api_key = api_key or LLM_API_KEY
        base_url = base_url or LLM_URL

        if not api_key or not base_url:
            raise ValueError("请提供 api_key 和 base_url：要么通过参数传入，要么在 main 中赋值 module.LLM_API_KEY/module.LLM_URL")
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def send_message(self, messages):
        """
        messages: list of dict, [{"role": "user", "content": "..."}]
        返回：模型的文本回答
        """
        print("MODEL:", LLM_MODEL)
        if LLM_MODEL.startswith('qwen'):
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,  # 可替换成你网关支持的版本
                messages=messages,
                extra_body={"enable_thinking": False},
            )
        else:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,  # 可替换成你网关支持的版本
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
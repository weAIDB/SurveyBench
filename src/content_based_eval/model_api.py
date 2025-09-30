from openai import OpenAI


def call_api(text, model, key, url):
    client = OpenAI(
        api_key=key,
        base_url=url,
    )

    if model.startswith('qwen'):
        completion = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': f'{text}'}],
            extra_body={"enable_thinking": False},
        )
    else:
        completion = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': f'{text}'}],
        )

    # response content
    response_text = completion.choices[0].message.content

    # token
    usage = getattr(completion, "usage", None)
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens


    return {
        "response": response_text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


if __name__ == "__main__":
    # 测试
    text = "Hello, this is a test message."
    model = "deepseek-chat"
    key = "sk-xxx"
    url = "https://api.deepseek.com/v1"
    result = call_api(text, model, key, url)
    print(result)

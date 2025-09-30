import re
import json
from .prompt import *
from .model_api import call_api
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def extract_text_from_md(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_outline_from_md(text):
    # 匹配 ## 或 ### 开头 + 数字编号 + 标题
    pattern = r'^(#{2,6})\s+(\d+(?:\.\d+)*)(.*?)$'
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    outline = []
    for level, number, title in matches:
        outline.append({
            "level": len(level),
            "number": number.strip(),
            "title": title.strip()
        })
    lines = []
    for item in outline:
        indent = '    ' * (item['level'] - 2)  # ## 不缩进，### 缩一次，以此类推
        lines.append(f"{indent}{item['number']} {item['title']}")
    return '\n'.join(lines)


def evaluate_outline(topic, model, method, api_key, api_url):

    text = extract_text_from_md(f'../data/{method}/{topic}.md')
    outline = extract_outline_from_md(text)

    human_survey = extract_text_from_md(f'../data/HumanSurvey/{topic}.md')
    human_outline = extract_outline_from_md(human_survey)
    # print(human_outline)

    coverage_prompt = OUTLINE_COVERAGE_EVAL_PROMPT_COMPARE.format(topic=topic, ai_outline=outline, human_outline=human_outline)
    relevance_prompt = OUTLINE_RELEVANCE_EVAL_PROMPT_COMPARE.format(topic=topic, ai_outline=outline, human_outline=human_outline)
    structure_prompt = OUTLINE_STRUCTURE_EVAL_PROMPT_COMPARE.format(topic=topic, ai_outline=outline, human_outline=human_outline)

    coverage_score_1 = call_api(coverage_prompt, model, api_key, api_url)
    relevance_score_1 = call_api(relevance_prompt, model, api_key, api_url)
    structure_score_1 = call_api(structure_prompt, model, api_key, api_url)

    coverage_score_2 = call_api(coverage_prompt, model, api_key, api_url)
    relevance_score_2 = call_api(relevance_prompt, model, api_key, api_url)
    structure_score_2 = call_api(structure_prompt, model, api_key, api_url)

    coverage_score_3 = call_api(coverage_prompt, model, api_key, api_url)
    relevance_score_3 = call_api(relevance_prompt, model, api_key, api_url)
    structure_score_3 = call_api(structure_prompt, model, api_key, api_url)

    return {
        "topic": topic,
        "coverage_score_1": coverage_score_1,
        "relevance_score_1": relevance_score_1,
        "structure_score_1": structure_score_1,
        "coverage_score_2": coverage_score_2,
        "relevance_score_2": relevance_score_2,
        "structure_score_2": structure_score_2,
        "coverage_score_3": coverage_score_3,
        "relevance_score_3": relevance_score_3,
        "structure_score_3": structure_score_3,
    }


if __name__ == "__main__":

    topics = [
        "Multimodal Large Language Models",
        "Evaluation of Large Language Models",
        "3D Object Detection in Autonomous Driving",
        "Vision Transformers",
        "Hallucination in Large Language Models",
        "Generative Diffusion Models",
        "3D Gaussian Splatting",
        "LLM-based Multi-Agent",
        "Graph Neural Networks",
        "Retrieval-Augmented Generation for Large Language Models",
        "Agentic Reinforcement Learning",
        "Alignment of Large Language Models",
        "Efficient Inference for Large Language Models",
        "Vision-Language-Action Models",
        "Explainability for Large Language Models",
        "Scientific Large Language Models",
        "Safety in Large Language Models",
        "Large Language Models for Time Series",
        "Large Language Models for Recommendation",
        "Reinforcement Learning for Large Language Models"
    ]

    model = "gpt-4o-mini"
    # methods = ["SurveyForge", "AutoSurvey", "LLMxMR-V2", "HumanSurvey"]
    methods = ["OpenAI_DeepResearch"]
    mode = 'compare'

    for method in methods:

        outline_scores = []
        with open("./HumanSurvey/outlines.json", "r", encoding="utf-8") as f:
            raw_human_outlines = json.load(f)
            human_outlines = {item["topic"]: item["outline"] for item in raw_human_outlines}

        with ThreadPoolExecutor(max_workers=20) as executor:
            future_to_topic = {executor.submit(evaluate_outline, topic, model, method, human_outlines[topic]): topic for topic in topics}
            for future in tqdm(as_completed(future_to_topic), total=len(topics), desc=f"{method}"):
                result = future.result()
                outline_scores.append(result)


        with open(f"./result/outline/{method}_outline_scores_{model}_{mode}.json", "w", encoding="utf-8") as f:
            json.dump(outline_scores, f, indent=4, ensure_ascii=False)

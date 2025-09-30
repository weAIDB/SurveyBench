import re
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from .model_api import call_api
from .prompt import *


def extract_text_from_md(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def extract_sections_from_md(text):
    # 匹配 ## 开头 + 数字编号 + 标题，记录其位置
    pattern = r'^(##)\s+(\d+(?:\.\d+)*)(.*?)$'
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))

    sections = []

    for i, match in enumerate(matches):
        level = len(match.group(1))
        number = match.group(2).strip()
        title = match.group(3).strip()
        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        content = text[start:end].strip()

        if len(content) > 2000:
            sections.append({
                'level': level,
                'number': number,
                'title': title,
                'content': content
            })

    return sections


def drop_ref(text):
    marker = "## References"
    idx = text.find(marker)
    if idx != -1:
        return text[:idx].strip()
    marker = "## REFERENCES"
    idx = text.find(marker)
    if idx != -1:
        return text[:idx].strip()
    return text.strip()


def normalize_score(text): 
    match = re.search(r"\d+", text)
    return match.group(0)


# w/o human survey as Ref.
# chapter-level evaluation
def evaluate_content_chapter(topic, model, method, api_key, api_url):
    text = extract_text_from_md(f'../data/{method}/{topic}.md')
    text = drop_ref(text)
    sections = extract_sections_from_md(text)

    section_scores = []

    for section in sections:
        section_title = section['number'] + ' ' + section['title']
        section_content = section['content']

        content_coherence_prompt = CONTENT_COHERENCE_EVAL_PROMPT_CHAPTER.format(topic=topic, section_title=section_title, section_content=section_content)
        content_depth_prompt = CONTENT_DEPTH_EVAL_PROMPT_CHAPTER.format(topic=topic, section_title=section_title, section_content=section_content)
        content_focus_prompt = CONTENT_FOCUS_EVAL_PROMPT_CHAPTER.format(topic=topic, section_title=section_title, section_content=section_content)
        content_coverage_prompt = CONTENT_COVERAGE_EVAL_PROMPT_CHAPTER.format(topic=topic, section_title=section_title, section_content=section_content)
        content_fluency_prompt = CONTENT_FLUENCY_EVAL_PROMPT_CHAPTER.format(topic=topic, section_title=section_title, section_content=section_content)

        prompts = {
            "coverage_score": content_coverage_prompt,
            "coherence_score": content_coherence_prompt,
            "depth_score": content_depth_prompt,
            "focus_score": content_focus_prompt,
            "fluency_score": content_fluency_prompt
        }

        scores = {key: call_api(prompt, model, api_key, api_url) for key, prompt in prompts.items()}

        section_scores.append({
            "topic": topic,
            # "level": section['level'],
            "section_number": section['number'],
            "section_title": section['title'],
            # "section_content": section['content'],
            **scores
        })

    return section_scores


# w/o human survey as Ref.
# document-level evaluation
def evaluate_content_document(topic, model, method, api_key, api_url):
    text = extract_text_from_md(f'../data/{method}/{topic}.md')
    text = drop_ref(text)

    survey_scores = []

    content_coherence_prompt = CONTENT_COHERENCE_EVAL_PROMPT_ENTIRE_SURVEY.format(topic=topic, content=text)
    content_depth_prompt = CONTENT_DEPTH_EVAL_PROMPT_ENTIRE_SURVEY.format(topic=topic, content=text)
    content_focus_prompt = CONTENT_FOCUS_EVAL_PROMPT_ENTIRE_SURVEY.format(topic=topic, content=text)
    content_coverage_prompt = CONTENT_COVERAGE_EVAL_PROMPT_ENTIRE_SURVEY.format(topic=topic, content=text)
    content_fluency_prompt = CONTENT_FLUENCY_EVAL_PROMPT_ENTIRE_SURVEY.format(topic=topic, content=text)

    prompts = {
        "coverage_score": content_coverage_prompt,
        "coherence_score": content_coherence_prompt,
        "depth_score": content_depth_prompt,
        "focus_score": content_focus_prompt,
        "fluency_score": content_fluency_prompt
    }

    scores = {key: call_api(prompt, model, api_key, api_url) for key, prompt in prompts.items()}

    survey_scores.append({
        "topic": topic,
        **scores
    })

    return survey_scores


# w/ human survey as Ref.
def evaluate_content_compare(topic, model, method, api_key, api_url):

    text = extract_text_from_md(f'../data/{method}/{topic}.md')
    ai_survey = drop_ref(text)
    ai_survey = ai_survey[:100000]

    human_survey = extract_text_from_md(f'../data/HumanSurvey/{topic}.md')
    human_survey = drop_ref(human_survey)
    human_survey = human_survey[:150000]

    survey_scores = []

    content_coherence_prompt = CONTENT_COHERENCE_EVAL_PROMPT_COMPARE.format(topic=topic, human_survey=human_survey, ai_survey=ai_survey)
    content_depth_prompt = CONTENT_DEPTH_EVAL_PROMPT_COMPARE.format(topic=topic, human_survey=human_survey, ai_survey=ai_survey)
    content_focus_prompt = CONTENT_FOCUS_EVAL_PROMPT_COMPARE.format(topic=topic, human_survey=human_survey, ai_survey=ai_survey)
    content_coverage_prompt = CONTENT_COVERAGE_EVAL_PROMPT_COMPARE.format(topic=topic, human_survey=human_survey, ai_survey=ai_survey)
    content_fluency_prompt = CONTENT_FLUENCY_EVAL_PROMPT_COMPARE.format(topic=topic, human_survey=human_survey, ai_survey=ai_survey)

    prompts = {
        "coverage_score": content_coverage_prompt,
        "coherence_score": content_coherence_prompt,
        "depth_score": content_depth_prompt,
        "focus_score": content_focus_prompt,
        "fluency_score": content_fluency_prompt
    }

    scores = {}
    for key, prompt in prompts.items():
        result = call_api(prompt, model, api_key, api_url)
        scores[key] = result

    survey_scores.append({
        "topic": topic,
        **scores
    })

    return survey_scores


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
        "Reinforcement Learning for Large Language Models",
    ]

    # model = "gpt-4o-mini"
    model = "qwen-flash"
    # methods = ["AutoSurvey", "SurveyForge", "LLMxMR-V2"]
    methods = ["OpenAI_DeepResearch"]
    setting = "with_human_survey"  # "without_human_survey_chapter" / "without_human_survey_document" / "with_human_survey"

    for method in methods:

        content_scores = []

        if setting == "without_human_survey_chapter":
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_topic = {executor.submit(evaluate_content_chapter, topic, model, method): topic for topic in topics}
                for future in tqdm(as_completed(future_to_topic), total=len(topics), desc=f"{method}"):
                    result = future.result()
                    content_scores.append(result)

        elif setting == "without_human_survey_document":
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_topic = {executor.submit(evaluate_content_document, topic, model, method): topic for topic in topics}
                for future in tqdm(as_completed(future_to_topic), total=len(topics), desc=f"{method}"):
                    result = future.result()
                    content_scores.append(result)

        elif setting == "with_human_survey":
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_topic = {executor.submit(evaluate_content_compare, topic, model, method): topic for topic in topics}
                for future in tqdm(as_completed(future_to_topic), total=len(topics), desc=f"{method}"):
                    result = future.result()
                    content_scores.append(result)

        with open(f"./result/content/{method}_content_scores_{setting}_{model}.json", "w", encoding="utf-8") as f:
            json.dump(content_scores, f, indent=4, ensure_ascii=False)

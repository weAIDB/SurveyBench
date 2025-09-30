import json
import pandas as pd

modes = ['with_human_survey', 'without_human_survey_chapter', 'without_human_survey_entire_survey']
modes = ['without_human_survey_chapter', 'without_human_survey_entire_survey']
modes = ['with_human_survey']
models = ['gpt-4o-mini', 'qwen-flash']

# 按照时间排序的 topic
old_topics = [
    "Graph Neural Networks",
    "Vision Transformers",
    "3D Object Detection in Autonomous Driving",
    "Generative Diffusion Models",
    "Large Language Models for Recommendation",
    "Multimodal Large Language Models",
    "Alignment of Large Language Models",
    "Evaluation of Large Language Models",
    "LLM-based Multi-Agent",
    "Hallucination in Large Language Models",
]

new_topics = [
    "Explainability for Large Language Models",
    "Retrieval-Augmented Generation for Large Language Models",
    "3D Gaussian Splatting",
    "Large Language Models for Time Series",
    "Efficient Inference for Large Language Models",
    "Safety in Large Language Models",
    "Vision-Language-Action Models",
    "Scientific Large Language Models",
    "Reinforcement Learning for Large Language Models",
    "Agentic Reinforcement Learning",
]

for mode in modes:
    print(f"\n=== Mode: {mode}, Model: AVERAGE of {models} ===")

    records = []

    for model in models:
        if mode == "with_human_survey":
            files = {
                "OpenAI-DR": f"./result/content/OpenAI_DeepResearch_content_scores_{mode}_{model}.json",
                "AutoSurvey": f"./result/content/AutoSurvey_content_scores_{mode}_{model}.json",
                "SurveyForge": f"./result/content/SurveyForge_content_scores_{mode}_{model}.json",
                "LLMxMR-V2": f"./result/content/LLMxMR-V2_content_scores_{mode}_{model}.json",
            }
        else:
            files = {
                "OpenAI-DR": f"./result/content/OpenAI_DeepResearch_content_scores_{mode}_{model}.json",
                "AutoSurvey": f"./result/content/AutoSurvey_content_scores_{mode}_{model}.json",
                "SurveyForge": f"./result/content/SurveyForge_content_scores_{mode}_{model}.json",
                "LLMxMR-V2": f"./result/content/LLMxMR-V2_content_scores_{mode}_{model}.json",
                "HumanSurvey": f"./result/content/HumanSurvey_content_scores_{mode}_{model}.json",
            }

        for method, path in files.items():
            with open(path, "r", encoding="utf-8") as f:
                all_data = json.load(f)  # 外层 list，每个元素也是 list（每个 topic 的 section 列表）

                for section_list in all_data:
                    for entry in section_list:
                        coverage = int(entry["coverage_score"]["response"])
                        depth = int(entry["depth_score"]["response"])
                        focus = int(entry["focus_score"]["response"])
                        coherence = int(entry["coherence_score"]["response"])
                        fluency = int(entry["fluency_score"]["response"])

                        records.append({
                            "Method": method,
                            "Topic": entry["topic"],
                            "Coverage": coverage,
                            "Depth": depth,
                            "Focus": focus,
                            "Coherence": coherence,
                            "Fluency": fluency,
                            "Average": (coverage + depth + focus + coherence + fluency) / 5
                        })

    # 转为 DataFrame
    df = pd.DataFrame(records)

    # 两个模型取均值
    df_avg = df.groupby(["Method", "Topic"]).mean(numeric_only=True).reset_index()

    # 再按 Method 平均（得到最终的两个模型的平均分）
    avg_scores = df_avg.groupby("Method")[["Coverage", "Depth", "Focus", "Coherence", "Fluency", "Average"]].mean().round(2)
    print("\n--- Overall Average by Method ---")
    print(avg_scores.to_string())

    # 按 topic 平均
    avg_scores = df_avg.groupby("Topic")[["Coverage", "Depth", "Focus", "Coherence", "Fluency", "Average"]].mean().round(2)
    print("\n--- Average by Topic ---")
    print(avg_scores.to_string())

    # 按 old/new 分组
    df_avg["Group"] = df_avg["Topic"].apply(lambda t: "Old" if t in old_topics else "New")
    avg_old_new = df_avg.groupby(["Method", "Group"])[["Coverage", "Depth", "Focus", "Coherence", "Fluency", "Average"]].mean().round(2)

    print("\n--- Old vs New Average by Method ---")
    print(avg_old_new.to_string())

import json
import pandas as pd

modes = ['compare', 'single']
modes = ['compare']
models = ['gpt-4o-mini', 'qwen-flash']

# 按照时间排序的 topic 列表 (Old vs New)
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

base_files_template = {
    "AutoSurvey": "./result/outline/AutoSurvey_outline_scores_{}_{}.json",
    "SurveyForge": "./result/outline/SurveyForge_outline_scores_{}_{}.json",
    "LLMxMR-V2": "./result/outline/LLMxMR-V2_outline_scores_{}_{}.json",
    "OpenAI-DR": "./result/outline/OpenAI_DeepResearch_outline_scores_{}_{}.json",
}

for mode in modes:
    # 如果是 single 模式，加上 HumanSurvey
    files_template = base_files_template.copy()
    if mode == "single":
        files_template["HumanSurvey"] = "./result/outline/HumanSurvey_outline_scores_{}_{}.json"

    records = []

    # 遍历所有模型，把结果收集到 records
    for model in models:
        for method, path_template in files_template.items():
            path = path_template.format(model, mode)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for entry in data:
                    coverage_score = (
                        int(entry["coverage_score_1"]["response"]) +
                        int(entry["coverage_score_2"]["response"]) +
                        int(entry["coverage_score_3"]["response"])
                    ) / 3
                    relevance_score = (
                        int(entry["relevance_score_1"]["response"]) +
                        int(entry["relevance_score_2"]["response"]) +
                        int(entry["relevance_score_3"]["response"])
                    ) / 3
                    structure_score = (
                        int(entry["structure_score_1"]["response"]) +
                        int(entry["structure_score_2"]["response"]) +
                        int(entry["structure_score_3"]["response"])
                    ) / 3
                    records.append({
                        "Method": method,
                        "Topic": entry["topic"],
                        "Coverage": coverage_score,
                        "Relevance": relevance_score,
                        "Structure": structure_score,
                        "Average": (coverage_score + relevance_score + structure_score) / 3
                    })

    # 转为 DataFrame
    df = pd.DataFrame(records)

    # === 两个模型的平均结果 ===
    df_avg = df.groupby(["Method", "Topic"]).mean(numeric_only=True).reset_index()

    # 再按 Method 求均值（最终输出）
    avg_scores = df_avg.groupby("Topic")[["Coverage", "Relevance", "Structure", "Average"]].mean().round(2)

    print(f"\n=== Mode: {mode} | Average Scores over Models ===")
    print(avg_scores.to_string())

    # 按 old/new 分组
    df_avg["Group"] = df_avg["Topic"].apply(lambda t: "Old" if t in old_topics else "New")
    avg_old_new = df_avg.groupby(["Method", "Group"])[["Coverage", "Relevance", "Structure", "Average"]].mean().round(2)

    print("\n--- Old vs New Average by Method ---")
    print(avg_old_new.to_string())




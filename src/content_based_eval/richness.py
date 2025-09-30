import os
import re
import pandas as pd
from .eval_content import drop_ref

def get_richness(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    text = drop_ref(text)

    # 图：匹配 ![...](...) 形式
    num_figures = len(re.findall(r"!\[.*?\]\(.*?\)", text))

    # HTML 表格
    num_html_tables = text.count("<table>")

    # Markdown 表格
    lines = text.splitlines()
    num_md_tables = 0
    in_table = False
    for line in lines:
        if "|" in line:
            if re.match(r"^\s*\|?.*---.*\|?.*$", line):  # 分隔行 ---
                if not in_table:
                    num_md_tables += 1
                    in_table = True
        else:
            in_table = False

    num_tables = num_html_tables + num_md_tables

    # 文章长度 (字符数)
    length = len(text)
    richness = (num_figures + num_tables) / length * 1e5 if length > 0 else 0

    return num_figures, num_tables, length, richness


def analyze_md_folder(folder, method):
    records = []
    for file in os.listdir(folder):
        if file.endswith(".md"):
            path = os.path.join(folder, file)
            figs, tabs, length, richness = get_richness(path)
            records.append({
                "method": method,
                "file": file,
                "figures": figs,
                "tables": tabs,
                "length": length,
                "richness": richness
            })
    return pd.DataFrame(records)


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

    methods = ["HumanSurvey", "LLMxMR-V2", "OpenAI_DeepResearch", "AutoSurvey", "SurveyForge"]

    all_results = []
    for method in methods:
        md_files_path = f'../../data/{method}'
        # print(f"========== {method} ==========")
        df = analyze_md_folder(md_files_path, method)
        # print(df)
        all_results.append(df)

    # 合并所有结果
    final_df = pd.concat(all_results, ignore_index=True)
    # print("========== 汇总结果 ==========")
    # print(final_df)

    # 按方法统计平均richness（先求和再平均）
    method_stats = []
    for method, group in final_df.groupby("method"):
        avg_figs = group["figures"].mean()
        avg_tabs = group["tables"].mean()
        avg_richness = group["richness"].mean()
        method_stats.append({
            "method": method,
            "avg_figures": avg_figs,
            "avg_tables": avg_tabs,
            "avg_richness": avg_richness
        })

    method_df = pd.DataFrame(method_stats)
    print("========== Avg Richness by Methods ==========")
    print(method_df)



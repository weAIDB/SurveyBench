import json
from pathlib import Path

# 问题分类定义
QUESTION_CATEGORIES = {
    1: {"difficulty": "Easy", "category": "Concept Definition", "subcategory": "Concept Definition"},
    2: {"difficulty": "Easy", "category": "Concept Definition", "subcategory": "Concept Definition"},
    3: {"difficulty": "Easy", "category": "Concept Definition", "subcategory": "Concept Definition"},
    4: {"difficulty": "Easy", "category": "Concept Definition", "subcategory": "Concept Definition"},
    5: {"difficulty": "Easy", "category": "Knowledge Organization and Classification",
        "subcategory": "Knowledge Organization"},
    6: {"difficulty": "Easy", "category": "Knowledge Organization and Classification",
        "subcategory": "Knowledge Organization"},
    7: {"difficulty": "Easy", "category": "Knowledge Organization and Classification",
        "subcategory": "Knowledge Organization"},
    8: {"difficulty": "Easy", "category": "Knowledge Organization and Classification",
         "subcategory": "Knowledge Organization"},
    9: {"difficulty": "Easy", "category": "Historical Understanding", "subcategory": "Historical Understanding"},
    10: {"difficulty": "Easy", "category": "Historical Understanding", "subcategory": "Historical Understanding"},
    11: {"difficulty": "Medium", "category": "Algorithmic Principles", "subcategory": "Algorithmic Principles"},
    12: {"difficulty": "Medium", "category": "Algorithmic Principles", "subcategory": "Algorithmic Principles"},
    13: {"difficulty": "Medium", "category": "Practical Guidance", "subcategory": "Practical Guidance"},
    14: {"difficulty": "Medium", "category": "Practical Guidance", "subcategory": "Practical Guidance"},
    15: {"difficulty": "Medium", "category": "Practical Guidance", "subcategory": "Practical Guidance"},
    16: {"difficulty": "Medium", "category": "Performance Analysis", "subcategory": "Performance Analysis"},
    17: {"difficulty": "Medium", "category": "Performance Analysis", "subcategory": "Performance Analysis"},
    18: {"difficulty": "Medium", "category": "Performance Analysis", "subcategory": "Performance Analysis"},
    19: {"difficulty": "Medium", "category": "Performance Analysis", "subcategory": "Performance Analysis"},
    20: {"difficulty": "Hard", "category": "Predictions", "subcategory": "Predictions"},
    21: {"difficulty": "Hard", "category": "Predictions", "subcategory": "Predictions"},
    22: {"difficulty": "Hard", "category": "Predictions", "subcategory": "Predictions"}
}


def get_question_category(question_id):
    """
    根据问题ID获取问题分类信息

    Args:
        question_id (int): 问题ID

    Returns:
        dict: 包含难度级别和类别信息的字典
    """
    return QUESTION_CATEGORIES.get(question_id, {
        "difficulty": "Unknown",
        "category": "Unknown",
        "subcategory": "Unknown"
    })


def analyze_json_data(json_file_path: str | Path):
    """
    分析JSON数据，统计better_answer相关信息

    Args:
        json_file_path (str): JSON文件路径

    Returns:
        dict: 包含统计结果的字典
    """

    # 读取JSON文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print(f"错误：找不到文件 {json_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"错误：JSON文件格式不正确")
        return None

    # 初始化统计变量
    better_answer_2_count = 0
    better_answer_1_reasons = []
    category_stats = {}

    # 处理results数组
    if 'results' in data:
        results = data['results']

        for result in results:
            question_id = result.get('question_id')
            better_answer = result.get('better_answer')
            comparison_reason = result.get('comparison_reason', '')

            # 获取问题分类信息
            category_info = get_question_category(question_id)

            # 统计better_answer为2的数量
            if better_answer == 2:
                better_answer_2_count += 1

            # 收集better_answer为1时的comparison_reason（包含分类信息）
            elif better_answer == 1:
                better_answer_1_reasons.append({
                    'question_id': question_id,
                    'comparison_reason': comparison_reason,
                    'difficulty': category_info['difficulty'],
                    'category': category_info['category'],
                    'subcategory': category_info['subcategory']
                })

            # 按类别统计
            category_key = f"{category_info['difficulty']}_{category_info['category']}"
            if category_key not in category_stats:
                category_stats[category_key] = {
                    'difficulty': category_info['difficulty'],
                    'category': category_info['category'],
                    'total': 0,
                    'better_answer_1': 0,
                    'better_answer_2': 0,
                    'other': 0
                }

            category_stats[category_key]['total'] += 1
            if better_answer == 1:
                category_stats[category_key]['better_answer_1'] += 1
            elif better_answer == 2:
                category_stats[category_key]['better_answer_2'] += 1
            else:
                category_stats[category_key]['other'] += 1

    return {
        'better_answer_2_count': better_answer_2_count,
        'better_answer_1_reasons': better_answer_1_reasons,
        'category_stats': category_stats,
        'total_questions': len(data.get('results', []))
    }


def analyze_json_string(json_string):
    """
    直接分析JSON字符串

    Args:
        json_string (str): JSON字符串

    Returns:
        dict: 包含统计结果的字典
    """

    try:
        data = json.loads(json_string)
    except json.JSONDecodeError:
        print("错误：JSON字符串格式不正确")
        return None

    # 初始化统计变量
    better_answer_2_count = 0
    better_answer_1_reasons = []
    category_stats = {}

    # 处理results数组
    if 'results' in data:
        results = data['results']

        for result in results:
            question_id = result.get('question_id')
            better_answer = result.get('better_answer')
            comparison_reason = result.get('comparison_reason', '')

            # 获取问题分类信息
            category_info = get_question_category(question_id)

            # 统计better_answer为2的数量
            if better_answer == 2:
                better_answer_2_count += 1

            # 收集better_answer为1时的comparison_reason（包含分类信息）
            elif better_answer == 1:
                better_answer_1_reasons.append({
                    'question_id': question_id,
                    'comparison_reason': comparison_reason,
                    'difficulty': category_info['difficulty'],
                    'category': category_info['category'],
                    'subcategory': category_info['subcategory']
                })

            # 按类别统计
            category_key = f"{category_info['difficulty']}_{category_info['category']}"
            if category_key not in category_stats:
                category_stats[category_key] = {
                    'difficulty': category_info['difficulty'],
                    'category': category_info['category'],
                    'total': 0,
                    'better_answer_1': 0,
                    'better_answer_2': 0,
                    'other': 0
                }

            category_stats[category_key]['total'] += 1
            if better_answer == 1:
                category_stats[category_key]['better_answer_1'] += 1
            elif better_answer == 2:
                category_stats[category_key]['better_answer_2'] += 1
            else:
                category_stats[category_key]['other'] += 1

    return {
        'better_answer_2_count': better_answer_2_count,
        'better_answer_1_reasons': better_answer_1_reasons,
        'category_stats': category_stats,
        'total_questions': len(data.get('results', []))
    }


def save_results_to_json(results, filename="analysis_results.json"):
    """
    将分析结果保存到 JSON 文件。

    Args:
        results (dict): 分析结果字典
        filename (str): 保存结果的文件名，默认为 "analysis_results.json"
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"分析结果已保存到 {filename}")
    except Exception as e:
        print(f"保存分析结果时发生错误: {e}")

def print_analysis_results(results, save_to_file=False, filename="analysis_results.json"):
    """
    打印分析结果，并可选择保存到文件

    Args:
        results (dict): 分析结果字典
        save_to_file (bool): 是否保存结果到 JSON 文件，默认为 False
        filename (str): 保存结果的文件名，默认为 "analysis_results.json"
    """
    if results is None:
        return

    print("=" * 60)
    print("JSON数据分析结果")
    print("=" * 60)

    print(f"总问题数量: {results['total_questions']}")
    print(f"better_answer为待测试survey的数量: {results['better_answer_2_count']}")
    print(f"better_answer为人工survey的数量: {len(results['better_answer_1_reasons'])}")

    # 按类别统计结果
    print("\n" + "=" * 60)
    print("按类别统计结果:（1为人工survey，2为待测试survey）")
    print("=" * 60)

    # 按难度级别分组显示
    difficulties = ["Easy", "Medium", "Hard"]
    for difficulty in difficulties:
        print(f"\n【{difficulty}级别】")
        print("-" * 40)

        difficulty_found = False
        for category_key, stats in results['category_stats'].items():
            if stats['difficulty'] == difficulty:
                difficulty_found = True
                print(f"  {stats['category']}:")
                print(f"    总题数: {stats['total']}")
                print(
                    f"    人工survey回答更好: {stats['better_answer_1']} ({stats['better_answer_1'] / stats['total'] * 100:.1f}%)")
                print(
                    f"    待测试survey回答更好: {stats['better_answer_2']} ({stats['better_answer_2'] / stats['total'] * 100:.1f}%)")
                if stats['other'] > 0:
                    print(f"    其他: {stats['other']}")

        if not difficulty_found:
            print(f"  暂无{difficulty}级别的题目")

    print("\n" + "=" * 60)
    print("better_answer为人工survey时的详细分析:")
    print("=" * 60)

    # 按类别分组显示better_answer为1的原因
    for difficulty in difficulties:
        difficulty_items = [item for item in results['better_answer_1_reasons']
                            if item['difficulty'] == difficulty]

        if difficulty_items:
            print(f"\n【{difficulty}级别 - 人工survey更好的情况】")
            print("-" * 50)

            # 按类别进一步分组
            categories = {}
            for item in difficulty_items:
                cat = item['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(item)

            for category, items in categories.items():
                print(f"\n  ◆ {category} ({len(items)}题):")
                for i, item in enumerate(items, 1):
                    print(f"    {i}. 问题ID: {item['question_id']}")
                    print(f"       原因: {item['comparison_reason']}")
                    print()

    # 如果需要保存到文件
    if save_to_file:
        save_results_to_json(results, filename)


def get_compare_final_results(json_path: str | Path, output_path: str | Path):
    # 假设你已经有分析好的结果
    results = analyze_json_data(json_path)
    print_analysis_results(results, save_to_file=True, filename=output_path)
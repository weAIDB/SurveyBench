import json
import os
from typing import Dict, List, Tuple
from pathlib import Path


def analyze_json_scores(filename: str | Path, output_path: str | Path) -> Dict:
    """
    读取文件夹内的JSON文件，统计所有score

    Args:
        folder_path (str): 文件夹路径

    Returns:
        Dict: 包含统计结果的字典
    """
    filename = Path(filename)

    # 初始化统计变量
    total_files = 0
    processed_files = 0
    all_scores = []
    file_scores = {}
    error_files = []

    # 遍历文件夹中的所有JSON文件
    for json_file in [filename]:
        total_files += 1

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 提取results中的所有score
            if 'results' in data and isinstance(data['results'], list):
                file_score_list = []
                for result in data['results']:
                    if 'score' in result and isinstance(result['score'], (int, float)):
                        score = float(result['score'])
                        all_scores.append(score)
                        file_score_list.append(score)

                # 记录每个文件的得分统计
                if file_score_list:
                    file_scores[json_file.name] = {
                        'scores': file_score_list,
                        'count': len(file_score_list),
                        'sum': sum(file_score_list),
                        'average': sum(file_score_list) / len(file_score_list),
                        'min': min(file_score_list),
                        'max': max(file_score_list)
                    }
                    processed_files += 1
                else:
                    error_files.append(f"{json_file.name}: 未找到有效的score数据")
            else:
                error_files.append(f"{json_file.name}: 缺少results字段或格式不正确")

        except json.JSONDecodeError as e:
            error_files.append(f"{json_file.name}: JSON解析错误 - {str(e)}")
        except Exception as e:
            error_files.append(f"{json_file.name}: 读取错误 - {str(e)}")

    # 计算总体统计
    overall_stats = {}
    if all_scores:
        overall_stats = {
            'total_scores': len(all_scores),
            'sum': sum(all_scores),
            'average': sum(all_scores) / len(all_scores),
            'min': min(all_scores),
            'max': max(all_scores),
            'median': sorted(all_scores)[len(all_scores) // 2] if len(all_scores) > 0 else 0
        }

        # 计算得分分布
        score_distribution = {}
        for score in all_scores:
            score_range = f"{int(score)}-{int(score) + 1}"
            score_distribution[score_range] = score_distribution.get(score_range, 0) + 1

    # 返回完整的统计结果
    result = {
        'summary': {
            'folder_path': str(filename.parent),
            'total_json_files': total_files,
            'successfully_processed': processed_files,
            'failed_files': len(error_files)
        },
        'overall_statistics': overall_stats,
        'file_statistics': file_scores,
        'errors': error_files
    }

    if all_scores:
        result['score_distribution'] = score_distribution

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    return result


def print_score_summary(stats: Dict) -> None:
    """
    打印得分统计摘要

    Args:
        stats (Dict): analyze_json_scores函数返回的统计结果
    """
    print("=" * 60)
    print("JSON文件Score统计报告")
    print("=" * 60)

    # 打印基本信息
    summary = stats['summary']
    print(f"文件夹路径: {summary['folder_path']}")
    print(f"总JSON文件数: {summary['total_json_files']}")
    print(f"成功处理文件数: {summary['successfully_processed']}")
    print(f"处理失败文件数: {summary['failed_files']}")
    print()

    # 打印总体统计
    if stats['overall_statistics']:
        overall = stats['overall_statistics']
        print("总体统计:")
        print(f"  总得分数量: {overall['total_scores']}")
        print(f"  得分总和: {overall['sum']:.2f}")
        print(f"  平均得分: {overall['average']:.2f}")
        print(f"  最低得分: {overall['min']:.2f}")
        print(f"  最高得分: {overall['max']:.2f}")
        print(f"  中位数得分: {overall['median']:.2f}")
        print()

        # 打印得分分布
        if 'score_distribution' in stats:
            print("得分分布:")
            for score_range, count in sorted(stats['score_distribution'].items()):
                print(f"  {score_range}: {count}个")
            print()

    # 打印每个文件的统计
    if stats['file_statistics']:
        print("各文件统计:")
        for filename, file_stats in stats['file_statistics'].items():
            print(f"  {filename}:")
            print(f"    得分数量: {file_stats['count']}")
            print(f"    平均得分: {file_stats['average']:.2f}")
            print(f"    得分范围: {file_stats['min']:.2f} - {file_stats['max']:.2f}")
        print()

    # 打印错误信息
    if stats['errors']:
        print("处理错误:")
        for error in stats['errors']:
            print(f"  {error}")


# 使用示例
def get_final_specific_results(json_path, output_path):
    try:
        # 分析JSON文件中的score
        results = analyze_json_scores(json_path, output_path)

        # 打印统计摘要
        print_score_summary(results)

    except Exception as e:
        print(f"错误: {e}")
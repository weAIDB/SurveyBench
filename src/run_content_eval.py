from content_based_eval.eval_content import evaluate_content_document, evaluate_content_chapter, evaluate_content_compare
from content_based_eval.eval_outline import evaluate_outline
from content_based_eval.richness import get_richness
import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def save_result(result, args):

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'content':
        filename = os.path.join(
            args.output_dir,
            f"{args.mode}_{args.setting}_{args.method}.json"
        )
    else:
        filename = os.path.join(
            args.output_dir,
            f"{args.mode}_{args.method}.json"
        )

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": args.mode,
        "setting": getattr(args, "setting", None),
        "model": args.model,
        "method": args.method,
        "result": result
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {filename}")



def run_single(args, topic):
    """Run evaluation for a single topic (file)"""
    model, api_key, api_url = args.model, args.api_key, args.api_url
    method = args.method
    mode = args.mode
    setting = getattr(args, "setting", "with_ref")

    if mode == "content":
        if setting == "with_ref":
            res = evaluate_content_compare(topic, model, method, api_key, api_url)
            return {
                "coverage": int(res[0]["coverage_score"]["response"]),
                "coherence": int(res[0]["coherence_score"]["response"]),
                "depth": int(res[0]["depth_score"]["response"]),
                "focus": int(res[0]["focus_score"]["response"]),
                "fluency": int(res[0]["fluency_score"]["response"]),
            }
        elif setting == "without_ref_chapter":
            res = evaluate_content_chapter(topic, model, method, api_key, api_url)
            n = len(res)
            return {
                "coverage": sum(int(x["coverage_score"]["response"]) for x in res) / n,
                "coherence": sum(int(x["coherence_score"]["response"]) for x in res) / n,
                "depth": sum(int(x["depth_score"]["response"]) for x in res) / n,
                "focus": sum(int(x["focus_score"]["response"]) for x in res) / n,
                "fluency": sum(int(x["fluency_score"]["response"]) for x in res) / n,
            }
        elif setting == "without_ref_document":
            res = evaluate_content_document(topic, model, method, api_key, api_url)
            return {
                "coverage": int(res[0]["coverage_score"]["response"]),
                "coherence": int(res[0]["coherence_score"]["response"]),
                "depth": int(res[0]["depth_score"]["response"]),
                "focus": int(res[0]["focus_score"]["response"]),
                "fluency": int(res[0]["fluency_score"]["response"]),
            }

    elif mode == "outline":
        res = evaluate_outline(topic, model, method, api_key, api_url)
        def parse(val):
            return int(val.strip().replace("Score:", "").replace("score:", "").replace("Score", ""))
        return {
            "coverage": sum(parse(res[f"coverage_score_{i}"]["response"]) for i in range(1, 4)) / 3,
            "relevance": sum(parse(res[f"relevance_score_{i}"]["response"]) for i in range(1, 4)) / 3,
            "structure": sum(parse(res[f"structure_score_{i}"]["response"]) for i in range(1, 4)) / 3,
        }

    elif mode == "richness":
        res = get_richness(f'../data/{method}/{topic}.md')
        return {"figures": res[0], "tables": res[1], "richness": res[3]}

    return {}


def main(args):
    data_dir = Path(f"../data/{args.method}")
    topics = [f.stem for f in data_dir.glob("*.md")]

    print(f'Found {len(topics)} .md files in {data_dir}')

    if not topics:
        print(f"No .md files found in {data_dir}")
        return

    if args.mode == "overall":
        all_content, all_outline, all_richness = [], [], []
        for topic in topics:
            all_content.append(run_single(argparse.Namespace(**{**vars(args), "mode": "content", "setting": "with_ref"}), topic))
            all_outline.append(run_single(argparse.Namespace(**{**vars(args), "mode": "outline"}), topic))
            all_richness.append(run_single(argparse.Namespace(**{**vars(args), "mode": "richness"}), topic))

        def avg_dict(list_of_dicts):
            return {k: sum(d[k] for d in list_of_dicts) / len(list_of_dicts) for k in list_of_dicts[0]}

        overall = {
            "content_avg": avg_dict(all_content),
            "outline_avg": avg_dict(all_outline),
            "richness_avg": avg_dict(all_richness)
        }
        save_result(overall, args)
        print("========== Overall Evaluation ==========")
        print(overall)

    else:
        all_scores = []
        for topic in topics:
            all_scores.append(run_single(args, topic))
        avg_scores = {k: sum(s[k] for s in all_scores) / len(all_scores) for k in all_scores[0]}
        save_result(avg_scores, args)
        print(f"========== {args.mode.capitalize()} Evaluation ==========")
        print(avg_scores)



def get_parser():
    parser = argparse.ArgumentParser(description="SurveyBench Evaluation")

    parser.add_argument('--mode', required=True, type=str,
                        choices=["overall", "content", "outline", "richness"],
                        help="Evaluation mode")
    parser.add_argument('--setting', default='with_ref', type=str,
                        help="Settings for content mode")
    parser.add_argument('--method', default='AutoSurvey', type=str,
                        help="Method to evaluate")
    parser.add_argument('--model', default='gpt-4o-mini', type=str,
                        help="Model to use")
    parser.add_argument('--api_key', default='', type=str,
                        help="API key")
    parser.add_argument('--api_url', default='', type=str,
                        help="API URL")
    parser.add_argument('--output_dir', default='./result/content', type=str, 
                        help="Output directory for results")

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)

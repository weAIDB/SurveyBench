from content_based_eval.eval_content import evaluate_content_document, evaluate_content_chapter, evaluate_content_compare
from content_based_eval.eval_outline import evaluate_outline
from content_based_eval.richness import get_richness
import argparse
import json
import os
from datetime import datetime


def save_result(result, args):
    os.makedirs("result/content", exist_ok=True)

    if args.mode == 'content':
        filename = f"result/content/{args.mode}_{args.setting}_{args.method}_{args.topic.replace(' ', '_')}.json"
    else:
        filename = f"result/content/{args.mode}_{args.method}_{args.topic.replace(' ', '_')}.json"

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": args.mode,
        "setting": getattr(args, "setting", None),
        "model": args.model,
        "method": args.method,
        "topic": args.topic,
        "result": result
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    # print(f"Results saved to {filename}")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='content', type=str, help='Evaluation mode: overall / content / outline / richness')
    parser.add_argument('--setting', default='with_ref', type=str, help='Content quality evaluation setting: with_ref / without_ref_chapter / without_ref_document')
    
    parser.add_argument('--model', default='gpt-4o-mini', type=str, help='Model to use for evaluation')
    parser.add_argument('--api_key', default='', type=str, help='API key for the model')
    parser.add_argument('--api_url', default='', type=str, help='API URL for the model')

    parser.add_argument('--method', default='AutoSurvey', type=str, help='Method to evaluate')
    parser.add_argument('--topic', default='Multimodal Large Language Models', type=str, help='Topic to evaluate')

    args = parser.parse_args()
    return args

def main(args):
    model, api_key, api_url = args.model, args.api_key, args.api_url
    method = args.method
    topic = args.topic
    mode = args.mode

    if mode == 'overall':
        print("========== Overall Content-Based Evaluation ==========")

        # 1. Content Evaluation (with_ref)
        res_content = evaluate_content_compare(topic, model, method, api_key, api_url)
        save_result(res_content, argparse.Namespace(**{**vars(args), "mode": "content", "setting": "with_ref"}))

        print("---- Content Evaluation (w/ Ref.) ----")
        print(f"Method: {method}")
        print(f"Topic: {topic}")
        print(f"  Coverage : {res_content[0]['coverage_score']['response']}")
        print(f"  Coherence: {res_content[0]['coherence_score']['response']}")
        print(f"  Depth    : {res_content[0]['depth_score']['response']}")
        print(f"  Focus    : {res_content[0]['focus_score']['response']}")
        print(f"  Fluency  : {res_content[0]['fluency_score']['response']}")

        # 2. Outline Evaluation
        res_outline = evaluate_outline(topic, model, method, api_key, api_url)
        save_result(res_outline, argparse.Namespace(**{**vars(args), "mode": "outline"}))

        coverage_scores = [res_outline[f"coverage_score_{i}"]["response"] for i in range(1, 3+1)]
        relevance_scores = [res_outline[f"relevance_score_{i}"]["response"] for i in range(1, 3+1)]
        structure_scores = [res_outline[f"structure_score_{i}"]["response"] for i in range(1, 3+1)]

        avg_coverage = sum(int(s.strip().replace("Score:", "").replace("score:", "").replace("Score", "")) for s in coverage_scores) / len(coverage_scores)
        avg_relevance = sum(int(s.strip().replace("Score:", "").replace("score:", "").replace("Score", "")) for s in relevance_scores) / len(relevance_scores)
        avg_structure = sum(int(s.strip().replace("Score:", "").replace("score:", "").replace("Score", "")) for s in structure_scores) / len(structure_scores)

        print("---- Outline Evaluation ----")
        print(f"Method: {method}")
        print(f"Topic: {topic}")
        print(f"  Coverage : {avg_coverage:.2f}")
        print(f"  Relevance: {avg_relevance:.2f}")
        print(f"  Structure: {avg_structure:.2f}")

        # 3. Richness Evaluation
        res_richness = get_richness(f'../data/{method}/{topic}.md')
        save_result(res_richness, argparse.Namespace(**{**vars(args), "mode": "richness"}))

        print("---- Richness Evaluation ----")
        print(f"Method: {method}")
        print(f"Topic: {topic}")
        print("  Figure Num.: {}, Table Num.: {}, Richness: {:.4f}".format(res_richness[0], res_richness[1], res_richness[3]))
    

    elif mode == 'content':
        setting = args.setting
        if setting == "with_ref":
            res = evaluate_content_compare(topic, model, method, api_key, api_url)
            save_result(res, args)
            print("========== Content Evaluation Result (w/ Ref.) ==========")
            print(f"Method: {method}")
            print(f"Topic: {topic}")
            print(f"  Coverage : {res[0]['coverage_score']['response']}")
            print(f"  Coherence: {res[0]['coherence_score']['response']}")
            print(f"  Depth    : {res[0]['depth_score']['response']}")
            print(f"  Focus    : {res[0]['focus_score']['response']}")
            print(f"  Fluency  : {res[0]['fluency_score']['response']}")

        elif setting == "without_ref_chapter":
            res = evaluate_content_chapter(topic, model, method, api_key, api_url)
            save_result(res, args)
            print("========== Content Evaluation Result (w/o Ref. Chapter-level) ==========")

            n = len(res)
            avg_coverage = sum(int(item["coverage_score"]["response"]) for item in res) / n
            avg_coherence = sum(int(item["coherence_score"]["response"]) for item in res) / n
            avg_depth = sum(int(item["depth_score"]["response"]) for item in res) / n
            avg_focus = sum(int(item["focus_score"]["response"]) for item in res) / n
            avg_fluency = sum(int(item["fluency_score"]["response"]) for item in res) / n

            print(f"Method: {method}")
            print(f"Topic: {topic}")
            print(f"  Coverage : {avg_coverage:.2f}")
            print(f"  Coherence: {avg_coherence:.2f}")
            print(f"  Depth    : {avg_depth:.2f}")
            print(f"  Focus    : {avg_focus:.2f}")
            print(f"  Fluency  : {avg_fluency:.2f}")

        elif setting == "without_ref_document":
            res = evaluate_content_document(topic, model, method, api_key, api_url)
            save_result(res, args)
            print("========== Content Evaluation Result (w/o Ref. Document-level) ==========")

            print(f"Method: {method}")
            print(f"Topic: {topic}")
            print(f"  Coverage : {res[0]['coverage_score']['response']}")
            print(f"  Coherence: {res[0]['coherence_score']['response']}")
            print(f"  Depth    : {res[0]['depth_score']['response']}")
            print(f"  Focus    : {res[0]['focus_score']['response']}")
            print(f"  Fluency  : {res[0]['fluency_score']['response']}")

    elif mode == 'outline':
        res = evaluate_outline(topic, model, method, api_key, api_url)
        save_result(res, args)
        print("========== Outline Evaluation Result ==========")

        coverage_scores = [res[f"coverage_score_{i}"]["response"] for i in range(1, 4)]
        relevance_scores = [res[f"relevance_score_{i}"]["response"] for i in range(1, 4)]
        structure_scores = [res[f"structure_score_{i}"]["response"] for i in range(1, 4)]

        avg_coverage = sum(int(s) for s in coverage_scores) / len(coverage_scores)
        avg_relevance = sum(int(s) for s in relevance_scores) / len(relevance_scores)
        avg_structure = sum(int(s) for s in structure_scores) / len(structure_scores)

        print(f"Method: {method}")
        print(f"Topic: {topic}")
        print(f"  Coverage : {avg_coverage:.2f}")
        print(f"  Relevance: {avg_relevance:.2f}")
        print(f"  Structure: {avg_structure:.2f}")

    elif mode == 'richness':
        res = get_richness(f'../data/{method}/{topic}.md')
        save_result(res, args)
        print("========== Richness Evaluation Result ==========")
        print(f"Method: {method}")
        print(f"Topic: {topic}")
        print("  Figure Num.: {}, Table Num.: {}, Richness: {:.4f}".format(res[0], res[1], res[3]))

    


if __name__ == "__main__":
    args = get_parser()
    main(args)



# python run_content_eval.py --mode overall --model gpt-4o-mini --method LLMxMR-V2 --topic 'Multimodal Large Language Models' --api_key sk-YPCgrExDNPUoUCmvEfB2568c034c4bCeBa414fF43f1eEeDd --api_url https://api.vveai.com/v1

# python run_content_eval.py --mode content --setting with_ref --model gpt-4o-mini --method AutoSurvey --topic 'Multimodal Large Language Models' --api_key sk-YPCgrExDNPUoUCmvEfB2568c034c4bCeBa414fF43f1eEeDd --api_url https://api.vveai.com/v1

# python run_content_eval.py --mode content --setting without_ref_document --model gpt-4o-mini --method AutoSurvey --topic 'Multimodal Large Language Models' --api_key sk-YPCgrExDNPUoUCmvEfB2568c034c4bCeBa414fF43f1eEeDd --api_url https://api.vveai.com/v1

# python run_content_eval.py --mode outline --model gpt-4o-mini --method AutoSurvey --topic 'Multimodal Large Language Models' --api_key sk-YPCgrExDNPUoUCmvEfB2568c034c4bCeBa414fF43f1eEeDd --api_url https://api.vveai.com/v1

# python run_content_eval.py --mode richness --method AutoSurvey --topic 'Multimodal Large Language Models'
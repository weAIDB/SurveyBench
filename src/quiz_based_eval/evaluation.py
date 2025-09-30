import re
from pathlib import Path
import json
import os
from datetime import datetime
from typing import Dict, Any, List
from api_services import *


class EvaluationService:
    """
    A comprehensive evaluation service for comparing answers using LLM evaluation.

    Supports two evaluation modes:
    1. Ground truth based evaluation with reference support checking
    2. Comparative evaluation between two answer-reference pairs
    """

    def __init__(self, model: ChatService, score_range: tuple = (0.0, 10.0)):
        """
        Parameters
        ----------
        model : ChatService
            LLM interface with a send_message method.
        score_range : tuple
            Expected numeric range for scores (default 0-10).
        """
        self.model = model
        self.score_range = score_range

    def check_reference_support(
            self,
            answer: str,
            references: List[Dict[str, Any]],
            use_external_knowledge: bool = True
    ) -> bool:
        """
        Check if an answer is fully supported by given references.

        Parameters
        ----------
        answer : str
            The answer to verify
        references : List[Dict[str, Any]]
            List of reference documents, each with 'text' field
        use_external_knowledge : bool
            If False, only use provided references (no external knowledge)

        Returns
        -------
        bool
            True if fully supported, False otherwise
        """
        if not references:
            return False

        combined_ref_text = "\n".join(
            f"[{ref.get('doc_number', ref.get('title', 'N/A'))}] {ref.get('text', '')}"
            for ref in references
        )

        knowledge_instruction = (
            "ONLY use the provided references below. Do not use any external knowledge."
            if not use_external_knowledge else
            "Use the provided references as your primary source, supplemented by your knowledge if needed."
        )

        prompt = (
            "You are a fact-checking assistant.\n"
            f"{knowledge_instruction}\n"
            "Determine whether the ANSWER below can be fully justified by the PROVIDED_REFERENCES.\n"
            "If the answer contains any claim or detail that cannot be directly inferred "
            "from these references, respond only with 'False'.\n"
            "Otherwise respond only with 'True'. No explanation, just one word.\n\n"
            f"ANSWER:\n{answer}\n\n"
            f"PROVIDED_REFERENCES:\n{combined_ref_text}\n"
        )

        message = [{"role": "user", "content": prompt}]
        raw = self.model.send_message(message).strip().lower()
        return raw.startswith("true")

    def score_with_ground_truth(
            self,
            question: str,
            answer: str,
            ground_truth: str,
            references: List[Dict[str, Any]] = None
    ) -> float:
        """
        Score an answer against ground truth, with optional reference support checking.

        Parameters
        ----------
        question : str
            The original question
        answer : str
            The answer to evaluate
        ground_truth : str
            The ground truth answer
        references : List[Dict[str, Any]], optional
            Reference documents to check support

        Returns
        -------
        float
            Score within self.score_range, or 0.0 if not supported by references
        """
        # First check reference support if references provided
        if references:
            if not self.check_reference_support(answer, references, use_external_knowledge=True):
                return 0.0

        # Score against ground truth
        prompt = (
            "You are an impartial evaluator.\n"
            "Compare the USER_ANSWER with the GROUND_TRUTH for the given QUESTION.\n"
            "Consider accuracy, completeness, and relevance.\n"
            f"Rate the USER_ANSWER strictly as a number between {self.score_range[0]} and {self.score_range[1]} (inclusive).\n"
            "You may score based on your own knowledge and do not have to strictly adhere to the ground truth."
            "Output only the number and nothing else.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"GROUND_TRUTH:\n{ground_truth}\n\n"
            f"USER_ANSWER:\n{answer}\n"
        )

        message = [{"role": "user", "content": prompt}]
        raw = self.model.send_message(message).strip()
        match = re.search(r"[-+]?\d*\.?\d+", raw)
        if not match:
            raise ValueError(f"Could not parse numeric score from: {raw}")

        score = float(match.group())
        low, high = self.score_range
        return max(low, min(high, score))

    def comparative_evaluation(
            self,
            question: str,
            answer1: str,
            references1: List[Dict[str, Any]],
            answer2: str,
            references2: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare two answer-reference pairs and determine which is better.

        Parameters
        ----------
        question : str
            The original question
        answer1 : str
            First answer
        references1 : List[Dict[str, Any]]
            References for first answer
        answer2 : str
            Second answer
        references2 : List[Dict[str, Any]]
            References for second answer

        Returns
        -------
        Dict[str, Any]
            {
                "answer1_supported": bool,
                "answer2_supported": bool,
                "answer1_score": float,
                "answer2_score": float,
                "better_answer": int,  # 1 or 2, or 0 if tie
                "comparison_reason": str
            }
        """
        # Check reference support for both answers (no external knowledge)
        answer1_supported = self.check_reference_support(answer1, references1, use_external_knowledge=False)
        answer2_supported = self.check_reference_support(answer2, references2, use_external_knowledge=False)

        # If not supported, score is 0
        answer1_score = 0.0 if not answer1_supported else None
        answer2_score = 0.0 if not answer2_supported else None

        # Score supported answers and compare
        if answer1_supported or answer2_supported:
            # Generate individual scores for supported answers
            if answer1_supported:
                answer1_score = self._score_single_answer(question, answer1, references1)
            if answer2_supported:
                answer2_score = self._score_single_answer(question, answer2, references2)

            # Comparative evaluation
            comparison_result = self._compare_answers(question, answer1, answer2,
                                                      answer1_supported, answer2_supported)
            better_answer = comparison_result["better"]
            comparison_reason = comparison_result["reason"]
        else:
            # Both unsupported
            better_answer = 0
            comparison_reason = "Both answers are not supported by their respective references."

        return {
            "answer1_supported": answer1_supported,
            "answer2_supported": answer2_supported,
            "answer1_score": answer1_score,
            "answer2_score": answer2_score,
            "better_answer": better_answer,
            "comparison_reason": comparison_reason
        }

    def _score_single_answer(
            self,
            question: str,
            answer: str,
            references: List[Dict[str, Any]]
    ) -> float:
        """Score a single answer based on question and its references."""
        combined_ref_text = "\n".join(
            f"[{ref.get('doc_number', ref.get('title', 'N/A'))}] {ref.get('text', '')}"
            for ref in references
        )

        prompt = (
            "You are an expert evaluator.\n"
            "Score how well the ANSWER addresses the QUESTION based on the provided REFERENCES.\n"
            "Consider accuracy, completeness, clarity, and relevance.\n"
            f"Rate the answer strictly as a number between {self.score_range[0]} and {self.score_range[1]} (inclusive).\n"
            "Output only the number and nothing else.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"REFERENCES:\n{combined_ref_text}\n\n"
            f"ANSWER:\n{answer}\n"
        )

        message = [{"role": "user", "content": prompt}]
        raw = self.model.send_message(message).strip()
        match = re.search(r"[-+]?\d*\.?\d+", raw)
        if not match:
            raise ValueError(f"Could not parse numeric score from: {raw}")

        score = float(match.group())
        low, high = self.score_range
        return max(low, min(high, score))

    def _compare_answers(
            self,
            question: str,
            answer1: str,
            answer2: str,
            answer1_supported: bool,
            answer2_supported: bool
    ) -> Dict[str, Any]:
        """Compare two answers and determine which is better."""
        if not answer1_supported and not answer2_supported:
            return {"better": 0, "reason": "Both answers lack reference support"}
        elif not answer1_supported:
            return {"better": 2, "reason": "Answer 1 lacks reference support"}
        elif not answer2_supported:
            return {"better": 1, "reason": "Answer 2 lacks reference support"}

        # Both supported, do comparative evaluation
        prompt = (
            "You are an impartial judge comparing two answers to the same question.\n"
            "Both answers are supported by their respective references.\n"
            "Compare them on accuracy, completeness, clarity, and helpfulness.\n"
            "Determine which answer is better overall.\n"
            "Respond with only '1' if Answer 1 is better, '2' if Answer 2 is better, or '0' if they are equally good.\n"
            "Then on a new line, provide a brief reason (max 50 words).\n\n"
            f"QUESTION:\n{question}\n\n"
            f"ANSWER 1:\n{answer1}\n\n"
            f"ANSWER 2:\n{answer2}\n"
        )

        message = [{"role": "user", "content": prompt}]
        raw = self.model.send_message(message).strip()
        lines = raw.split('\n', 1)

        try:
            better = int(lines[0].strip())
            reason = lines[1].strip() if len(lines) > 1 else "No reason provided"
        except (ValueError, IndexError):
            better = 0
            reason = "Could not parse comparison result"

        return {"better": better, "reason": reason}

    def evaluate_from_json_files(
            self,
            human_answers_path: str | Path,
            machine_answers_path: str | Path,
            evaluation_mode: str = "comparative"
    ) -> List[Dict[str, Any]]:
        """
        Load and evaluate answers from two JSON files.

        Parameters
        ----------
        human_answers_path : str
            Path to human answers JSON file
        machine_answers_path : str
            Path to machine answers JSON file
        evaluation_mode : str
            "comparative" or "ground_truth" (if ground truth available)

        Returns
        -------
        List[Dict[str, Any]]
            Evaluation results for each question
        """
        with open(human_answers_path, "r", encoding="utf-8") as f:
            human_data = json.load(f)

        with open(machine_answers_path, "r", encoding="utf-8") as f:
            machine_data = json.load(f)

        human_results = human_data["results"] if evaluation_mode == "comparative" else human_data
        machine_results = machine_data["results"]

        if len(human_results) != len(machine_results):
            raise ValueError("Mismatched number of questions between files")

        evaluation_results = []

        for human_item, machine_item in zip(human_results, machine_results):
            if human_item["question"] != machine_item["question"]:
                raise ValueError(f"Question mismatch at ID {human_item.get('question_id')}")

            question = human_item["question"]

            if evaluation_mode == "comparative":
                result = self.comparative_evaluation(
                    question=question,
                    answer1=human_item["answer"],
                    references1=human_item["reference"],
                    answer2=machine_item["answer"],
                    references2=machine_item["reference"]
                )
                result.update({
                    "question_id": human_item.get("question_id"),
                    "question": question,
                    "human_answer": human_item["answer"],
                    "machine_answer": machine_item["answer"]
                })
            else:
                machine_score = self.score_with_ground_truth(
                    question=question,
                    answer=machine_item["answer"],
                    ground_truth=human_item["answer"],
                    references=machine_item["reference"]
                )
                result = {
                    "question_id": human_item.get("question_id"),
                    "question": question,
                    "machine_answer": machine_item["answer"],
                    "ground_truth": human_item["answer"],
                    "score": machine_score
                }

            evaluation_results.append(result)

        return evaluation_results


def save_results_to_json(results: List[Dict[str, Any]], output_path: str | Path, metadata: Dict[str, Any] = None):
    """Save evaluation results to JSON file with metadata."""
    output_data = {
        "metadata": metadata or {},
        "results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Results saved to: {output_path}")


def print_summary_stats(results: List[Dict[str, Any]], evaluation_mode: str):
    """Print summary statistics of the evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    total_questions = len(results)
    print(f"Total questions evaluated: {total_questions}")

    if evaluation_mode == "comparative":
        answer1_supported = sum(1 for r in results if r.get("answer1_supported", False))
        answer2_supported = sum(1 for r in results if r.get("answer2_supported", False))

        print(
            f"Human answers supported by references: {answer1_supported}/{total_questions} ({answer1_supported / total_questions * 100:.1f}%)")
        print(
            f"Machine answers supported by references: {answer2_supported}/{total_questions} ({answer2_supported / total_questions * 100:.1f}%)")

        # Count better answers
        human_wins = sum(1 for r in results if r.get("better_answer") == 1)
        machine_wins = sum(1 for r in results if r.get("better_answer") == 2)
        ties = sum(1 for r in results if r.get("better_answer") == 0)

        print(f"\nComparison Results:")
        print(f"Human answers better: {human_wins}/{total_questions} ({human_wins / total_questions * 100:.1f}%)")
        print(f"Machine answers better: {machine_wins}/{total_questions} ({machine_wins / total_questions * 100:.1f}%)")
        print(f"Ties/Both unsupported: {ties}/{total_questions} ({ties / total_questions * 100:.1f}%)")

        # Average scores for supported answers
        human_scores = [r["answer1_score"] for r in results if
                        r.get("answer1_supported") and r["answer1_score"] is not None]
        machine_scores = [r["answer2_score"] for r in results if
                          r.get("answer2_supported") and r["answer2_score"] is not None]

        if human_scores:
            print(f"\nAverage human answer score: {sum(human_scores) / len(human_scores):.2f} (n={len(human_scores)})")
        if machine_scores:
            print(
                f"Average machine answer score: {sum(machine_scores) / len(machine_scores):.2f} (n={len(machine_scores)})")

    elif evaluation_mode == "ground_truth":
        scores = [r["score"] for r in results if r["score"] is not None]
        zero_scores = sum(1 for r in results if r["score"] == 0.0)

        print(
            f"Answers with 0 score (unsupported): {zero_scores}/{total_questions} ({zero_scores / total_questions * 100:.1f}%)")

        if scores:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            print(f"Average score: {avg_score:.2f}")
            print(f"Score range: {min_score:.2f} - {max_score:.2f}")

    print("=" * 60)


def main():

    human_answers_dir = "../data/HumanSurvey/template_answers"
    machine_answers_dir = "../data/LLM×MR-V2/template_answers"
    output_dir = "./LLM×MR-V2_test_results_qwen"

    human_answers_dir = Path(human_answers_dir)
    machine_answers_dir = Path(machine_answers_dir)
    output_dir = Path(output_dir)

    mode = "comparative"

    score_range = [0.0, 10.0]

    # 创建输出目录（如果不存在）
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取文件列表（只保留文件）
    human_files = [f for f in os.listdir(human_answers_dir) if os.path.isfile(human_answers_dir / f)]
    machine_files = [f for f in os.listdir(machine_answers_dir) if os.path.isfile(machine_answers_dir / f)]

    # 取交集，确保两边都有对应文件
    # common_files = sorted(list(set(human_files) & set(machine_files)))
    common_files = ["Large Language Models for Recommendation_answer_for_template.json"]

    print("Common files number = {}".format(len(common_files)))

    # 按文件名匹配取出文件对
    for filename in common_files:
        human_path = human_answers_dir / filename
        machine_path = machine_answers_dir / filename

        print("Starting evaluation...")
        print(f"Human answers: {human_path}")
        print(f"Machine answers: {machine_path}")
        # ["comparative", "ground_truth"]
        print(f"Evaluation mode: {mode}")
        print(f"Score range: {score_range[0]} - {score_range[1]}")

        model = ChatService()

        evaluator = EvaluationService(model=model, score_range=tuple(score_range))

        try:
            # 运行评估
            results = evaluator.evaluate_from_json_files(
                human_answers_path=human_path,
                machine_answers_path=machine_path,
                evaluation_mode=mode
            )

            # 准备 metadata
            metadata = {
                "evaluation_timestamp": datetime.now().isoformat(),
                "evaluation_mode": mode,
                "score_range": score_range,
                "input_files": {
                    "human_answers": str(human_path),
                    "machine_answers": str(machine_path)
                },
                "total_questions": len(results),
                "evaluation_settings": {
                    "model_type": str(type(model).__name__),
                    "score_range": score_range
                }
            }

            # 每对文件单独命名保存
            output_file = output_dir / f"{filename}_results.json"
            save_results_to_json(results, output_file, metadata)

            # 打印 summary
            print_summary_stats(results, mode)
            print(f"Evaluation for {filename} completed successfully!\n")

        except Exception as e:
            print(f"Error during evaluation of {filename}: {e}")
            continue


def ground_truth_answer(tool_name: str):

    machine_answers_dir = "../data/" + tool_name + "/text_answers"
    human_dir = "../data/" + "text_derived_questions"
    output_dir = "./" + tool_name + "_driven_results_qwen"
    machine_answers_dir = Path(machine_answers_dir)

    output_dir = Path(output_dir)

    mode = "ground_truth"

    score_range = [0.0, 10.0]

    # 创建输出目录（如果不存在）
    output_dir.mkdir(parents=True, exist_ok=True)

    machine_files = [f for f in os.listdir(machine_answers_dir) if os.path.isfile(machine_answers_dir / f)]

    print("files number = {}".format(len(machine_files)))

    # 按文件名匹配取出文件对
    for filename in machine_files:
        machine_path = machine_answers_dir / filename
        human_path = human_dir + "/qa_pairs_" + filename[:-21] + ".json"

        print("Starting evaluation...")
        print(f"Machine answers: {machine_path}")
        print(f"Human truth: {human_path}")
        # ["comparative", "ground_truth"]
        print(f"Evaluation mode: {mode}")
        print(f"Score range: {score_range[0]} - {score_range[1]}")

        # save_results_to_json(parse_json_results(machine_path), "machine_mid_save.json", {"score_range": score_range})
        # save_results_to_json(parse_json_results(human_path), "human_mid_save.json", {"score_range": score_range})

        # 初始化模型和 evaluator （保持原有逻辑）
        model = ChatService()

        evaluator = EvaluationService(model=model, score_range=tuple(score_range))

        try:
            # 运行评估
            results = evaluator.evaluate_from_json_files(
                human_answers_path=human_path,
                machine_answers_path=machine_path,
                evaluation_mode=mode
            )

            # 准备 metadata
            metadata = {
                "evaluation_timestamp": datetime.now().isoformat(),
                "evaluation_mode": mode,
                "score_range": score_range,
                "input_files": {
                    "machine_answers": str(machine_path)
                },
                "total_questions": len(results),
                "evaluation_settings": {
                    "model_type": str(type(model).__name__),
                    "score_range": score_range
                }
            }

            # 每对文件单独命名保存
            clean_name = filename[:-3]
            output_file = output_dir / f"{clean_name}_results.json"
            save_results_to_json(results, output_file, metadata)

            # 打印 summary
            print_summary_stats(results, mode)
            print(f"Evaluation for {filename} completed successfully!\n")

        except Exception as e:
            print(f"Error during evaluation of {filename}: {e}")
            continue


if __name__ == "__main__":
    # You can uncomment this to run the example instead of command-line args
    # run_example_evaluation()

    # ../data/HumanSurvey/template_answers/3D Gaussian Splatting_answer_for_template.json
    # ../data/AutoSurvey/template_answers/3D Gaussian Splatting_answer_for_template.json
    main()

    # tool_list = [
    #     "DeepResearch"
    # ]
    #
    # for tool_name in tool_list:
    #     ground_truth_answer(tool_name)

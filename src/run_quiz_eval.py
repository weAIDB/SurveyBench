import argparse
import shutil
import logging
import re
import json
import os
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import quiz_based_eval.api_services as model_service

from quiz_based_eval.answer import SurveyContextBuilder, load_json_questions
# from quiz_based_eval.api_services import *
from quiz_based_eval.evaluation import EvaluationService, save_results_to_json, print_summary_stats
from quiz_based_eval.generate_questions import drop_ref, MarkdownParser, SurveyQAGenerator, parse_yaml_questions
from quiz_based_eval.compare_result_statistics import get_compare_final_results
from quiz_based_eval.score_specific_statistics import get_final_specific_results

logger = logging.getLogger(__name__)
logger.info("程序启动，日志等级=%s", "logging.INFO")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="日志等级（默认 INFO）"
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="可选：日志输出文件路径，不填则只输出到控制台"
    )
    return parser.parse_args()

def setup_logger(level: str, file: str | None):
    """
    根据参数设置全局 logger。
    任何模块调用 logging.getLogger(__name__) 都会继承这个配置。
    """
    handlers = []
    console = logging.StreamHandler()
    handlers.append(console)

    if file:
        file_handler = logging.FileHandler(file, mode="a", encoding="utf-8")
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers
    )


def compare_dirs(survey_dir: Path, human_dir: Path, output_dir: Path, temp_dir: Path, llm_model, emb_model, dimension):
    print(f"To test: {survey_dir}")
    print(f"Human: {human_dir}")
    output_dir = Path("../results") / output_dir
    print(f"Output: {output_dir}")
    temp_dir = Path("../results") / temp_dir
    print(f"Temp dir: {temp_dir}")
    if not survey_dir.exists() or not human_dir.exists():
        logger.error("The folder to be detected does not exist")

    output_dir.mkdir(parents=True, exist_ok=True) if not output_dir.exists() else None

    generate_surveys = os.listdir(survey_dir)
    human_surveys = os.listdir(human_dir)
    total_filenames = [filename[:-3] for filename in generate_surveys if filename in human_surveys]

    if len(total_filenames) != len(generate_surveys):
        logger.error("You need to ensure that the file names in the two folders are the same.")

    for filename in total_filenames:
        test_single_survey(
            survey_dir / Path(f"{filename}.md"),
            human_dir / Path(f"{filename}.md"),
            output_dir,
            temp_dir,
            llm_model,
            emb_model,
            dimension
        )


def test_single_survey(survey: Path, human: Path, output: Path, temp_dir: Path, llm_model, emb_model, dimension):
    with open(survey, 'r', encoding='utf-8') as f:
        survey_text = f.read()
    with open(human, 'r', encoding='utf-8') as f:
        human_text = f.read()
    survey_text = drop_ref(survey_text)
    human_text = drop_ref(human_text)

    clean_name = str(survey.name)[:-3]

    # 1,Generate general query
    general_queries = parse_yaml_questions("quiz_based_eval/questions.yaml", clean_name)["questions"]

    # 2,Generate specific query
    textparser = MarkdownParser()
    entries = textparser.parse_and_build_index(human_text)
    center_texts = [entry['main_sentence'] for entry in entries]
    full_texts = [entry['full_text'] for entry in entries]
    qa_generator = SurveyQAGenerator(center_texts, full_texts, llm_model)
    qa_pairs = qa_generator.generate_qa_pairs(n_questions=25, questions_per_segment=1)
    storage_dir = temp_dir / "qa_pairs"
    if not os.path.exists(storage_dir):
        os.makedirs(storage_dir)
    qa_generator.save_qa_pairs(qa_pairs, storage_dir / f"qa_pairs_{clean_name}.json", format='json')

    # 3,answer general questions for both human survey and survey to test
    executor = SurveyContextBuilder(emb_model, llm_model, dimension)
    # 3.1, test human survey
    human_temp_dir = temp_dir / "human_general_answers"
    human_general_answer_file = human_temp_dir / f"{clean_name}.json"
    title_text_dict, chunk_titles = executor.process_markdown_for_faiss(human_text)
    executor.batch_query_and_save(general_queries, title_text_dict, chunk_titles, human_general_answer_file)
    # 3.2, test generate survey
    survey_temp_dir = temp_dir / "generate_general_answers"
    survey_general_answer_file = survey_temp_dir / f"{clean_name}.json"
    title_text_dict, chunk_titles = executor.process_markdown_for_faiss(survey_text)
    executor.batch_query_and_save(general_queries, title_text_dict, chunk_titles, survey_general_answer_file)

    # 4,answer topic-specific queries only for survey to test
    # 4.1 load specific queries
    specific_queries = load_json_questions(storage_dir / f"qa_pairs_{clean_name}.json")
    # 4.2 answer and save
    specific_temp_dir = temp_dir / "generate_specific_answers"
    survey_specific_file = specific_temp_dir / f"{clean_name}.json"
    title_text_dict, chunk_titles = executor.process_markdown_for_faiss(survey_text)
    executor.batch_query_and_save(specific_queries, title_text_dict, chunk_titles, survey_specific_file)

    compare_evaluation(survey_general_answer_file, human_general_answer_file, clean_name, temp_dir)
    ground_truth_evaluation(survey_specific_file, storage_dir / f"qa_pairs_{clean_name}.json", clean_name, temp_dir)

    compare_results_path = temp_dir / f"general_evaluation_results/{clean_name}_general_results.json"
    output_compare_path = output / f"{clean_name}_compare_results.json"
    get_compare_final_results(compare_results_path, output_compare_path)

    specific_results_path = temp_dir / f"specific_evaluation_results/{clean_name}_specific_results.json"
    output_specific_path = output / f"{clean_name}_specific_results.json"
    get_final_specific_results(specific_results_path, output_specific_path)

def compare_evaluation(survey_answer: Path, human_answer: Path, clean_name: str, temp_dir: Path):
    results_dir = temp_dir / "general_evaluation_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    mode = "comparative"
    score_range = [0.0, 10.0]
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Starting evaluation...")
    print(f"Human answers: {human_answer}")
    print(f"Machine answers: {survey_answer}")
    print(f"Evaluation mode: {mode}")
    print(f"Score range: {score_range[0]} - {score_range[1]}")

    model = model_service.ChatService()

    evaluator = EvaluationService(model=model, score_range=tuple(score_range))

    try:
        results = evaluator.evaluate_from_json_files(
            human_answers_path=human_answer,
            machine_answers_path=survey_answer,
            evaluation_mode=mode
        )

        metadata = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "evaluation_mode": mode,
            "score_range": score_range,
            "input_files": {
                "human_answers": str(human_answer),
                "machine_answers": str(survey_answer)
            },
            "total_questions": len(results),
            "evaluation_settings": {
                "model_type": str(type(model).__name__),
                "score_range": score_range
            }
        }

        output_file = results_dir / f"{clean_name}_general_results.json"
        save_results_to_json(results, output_file, metadata)

        print_summary_stats(results, mode)
        print(f"Evaluation for {clean_name} completed successfully!\n")

    except Exception as e:
        print(f"Error during evaluation of {clean_name}: {e}")


def ground_truth_evaluation(survey_answer: Path, human_truth: Path, clean_name: str, temp_dir: Path):
    results_dir = temp_dir / "specific_evaluation_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    mode = "ground_truth"

    score_range = [0.0, 10.0]

    print("Starting evaluation...")
    print(f"Machine answer: {survey_answer}")
    print(f"Human ground truth: {human_truth}")
    print(f"Evaluation mode: {mode}")
    print(f"Score range: {score_range[0]} - {score_range[1]}")

    model = model_service.ChatService()

    evaluator = EvaluationService(model=model, score_range=tuple(score_range))

    try:
        results = evaluator.evaluate_from_json_files(
            human_answers_path=human_truth,
            machine_answers_path=survey_answer,
            evaluation_mode=mode
        )

        metadata = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "evaluation_mode": mode,
            "score_range": score_range,
            "input_files": {
                "machine_answers": str(survey_answer),
            },
            "total_questions": len(results),
            "evaluation_settings": {
                "model_type": str(type(model).__name__),
                "score_range": score_range
            }
        }

        output_file = results_dir / f"{clean_name}_specific_results.json"
        save_results_to_json(results, output_file, metadata)

        # 打印 summary
        print_summary_stats(results, mode)
        print(f"Evaluation for {clean_name} completed successfully!\n")

    except Exception as e:
        print(f"Error during evaluation of {clean_name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare generated survey(s) with human survey(s)."
    )

    # ---- 文件夹模式 ----
    parser.add_argument("--survey_dir", type=Path, required=True,
                            help="待检测 survey 文件夹路径")
    parser.add_argument("--human_dir", type=Path, required=True,
                            help="人工 survey 文件夹路径")
    parser.add_argument("--output_dir", type=Path, required=True,
                            help="结果输出文件夹路径")
    parser.add_argument("--llm", default='gpt-4o-mini', type=str,
                        help="LLM to use")
    parser.add_argument("--llm_api_key", default='', type=str,
                        help="LLM API key")
    parser.add_argument("--llm_api_url", default='', type=str,
                        help="LLM API URL")
    parser.add_argument("--emb_model", default='text-embedding-3-small', type=str,
                        help="Embedding model to use")
    parser.add_argument("--emb_dimension", default='', type=str,
                        help="Embedding dimension to use")
    parser.add_argument("--emb_api_key", default='', type=str,
                        help="Embedding API key")
    parser.add_argument("--emb_api_url", default='', type=str,
                        help="Embedding API URL")

    args = parser.parse_args()
    setup_logger("logging.INFO", "../results/quiz_logs.txt")

    model_service.LLM_API_KEY = args.llm_api_key
    model_service.LLM_URL = args.llm_api_url
    model_service.EMB_API_KEY = args.emb_api_key
    model_service.EMB_URL = args.emb_api_url
    model_service.DIMENSION = int(args.emb_dimension)

    model_service.EMBEDDING_MODEL = args.emb_model
    model_service.LLM_MODEL = args.llm

    llm_model = model_service.ChatService()
    emb_model = model_service.EmbeddingService()
    dimension = model_service.DIMENSION

    temp_files_dir = Path("../temp_output")
    if temp_files_dir.exists():
        shutil.rmtree(temp_files_dir)
    temp_files_dir.mkdir(parents=True, exist_ok=True)

    compare_dirs(args.survey_dir, args.human_dir, args.output_dir, temp_files_dir, llm_model, emb_model, dimension)

if __name__ == "__main__":
    main()
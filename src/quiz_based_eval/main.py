import argparse
import shutil

from answer import *
from evaluation import *
from generate_questions import *


llm_model = ChatService()
emb_model = EmbeddingService()
dimension = DIMENSION


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


def compare_file(survey: Path, human: Path, output: Path, temp_dir: Path):
    print(f"To test: {survey}")
    print(f"Human: {human}")
    print(f"Output: {output}")
    print(f"Temp dir: {temp_dir}")
    test_single_survey(survey, human, output, temp_dir)


def compare_dirs(survey_dir: Path, human_dir: Path, output_dir: Path, temp_dir: Path):
    print(f"To test: {survey_dir}")
    print(f"Human: {human_dir}")
    print(f"Output: {output_dir}")
    print(f"Temp dir: {temp_dir}")
    if not survey_dir.exists() or not human_dir.exists():
        logger.error("The folder to be detected does not exist")

    output_dir.mkdir(parents=True, exist_ok=True) if not output_dir.exists() else None

    generate_surveys = os.listdir(survey_dir)
    human_surveys = os.listdir(human_dir)
    total_filenames = [filename for filename in generate_surveys if filename in human_surveys]

    if len(total_filenames) != len(generate_surveys):
        logger.error("You need to ensure that the file names in the two folders are the same.")

    for filename in total_filenames:
        test_single_survey(
            Path(survey_dir + f"/{filename}.md"),
            Path(human_dir + f"/{filename}.md"),
            Path(output_dir + f"/{filename}_test_results.json"),
            temp_dir
        )


def test_single_survey(survey: Path, human: Path, output: Path, temp_dir: Path):
    with open(survey, 'r', encoding='utf-8') as f:
        survey_text = f.read()
    with open(human, 'r', encoding='utf-8') as f:
        human_text = f.read()
    survey_text = drop_ref(survey_text)
    human_text = drop_ref(human_text)

    clean_name = str(survey.name)[:-3]

    # 1,Generate general query
    general_queries = parse_yaml_questions("questions.yaml", clean_name)["questions"]

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


def compare_evaluation(survey_answer: Path, human_answer: Path, clean_name: str, temp_dir: Path):
    results_dir = temp_dir / "evaluation_results"
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

    model = ChatService()

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
    results_dir = temp_dir / "evaluation_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    mode = "ground_truth"

    score_range = [0.0, 10.0]

    print("Starting evaluation...")
    print(f"Machine answer: {survey_answer}")
    print(f"Human ground truth: {human_truth}")
    print(f"Evaluation mode: {mode}")
    print(f"Score range: {score_range[0]} - {score_range[1]}")

    model = ChatService()

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
    subparsers = parser.add_subparsers(dest="mode", required=True,
                                       help="选择运行模式：file 或 dir")

    # ---- 单文件模式 ----
    parser_file = subparsers.add_parser("file", help="单文件比较")
    parser_file.add_argument("--survey", type=Path, required=True,
                             help="待检测 survey 文件路径")
    parser_file.add_argument("--human", type=Path, required=True,
                             help="人工 survey 文件路径")
    parser_file.add_argument("--output", type=Path, required=True,
                             help="结果输出文件路径")
    parser_file.add_argument("--logfile", type=Path, required=True,
                             help="输出日志文件路径")

    # ---- 文件夹模式 ----
    parser_dir = subparsers.add_parser("dir", help="文件夹比较")
    parser_dir.add_argument("--survey_dir", type=Path, required=True,
                            help="待检测 survey 文件夹路径")
    parser_dir.add_argument("--human_dir", type=Path, required=True,
                            help="人工 survey 文件夹路径")
    parser_dir.add_argument("--output_dir", type=Path, required=True,
                            help="结果输出文件夹路径")
    parser_dir.add_argument("--logfile", type=Path, required=True,
                             help="输出日志文件路径")

    args = parser.parse_args()
    setup_logger("logging.INFO", args.logfile)

    temp_files_dir = Path("../temp_output")
    if temp_files_dir.exists():
        shutil.rmtree(temp_files_dir)
    temp_files_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.info("程序启动，日志等级=%s", "logging.INFO")

    if args.mode == "file":
        compare_file(args.survey, args.human, args.output, temp_files_dir)
    elif args.mode == "dir":
        compare_dirs(args.survey_dir, args.human_dir, args.output, temp_files_dir)

if __name__ == "__main__":
    main()
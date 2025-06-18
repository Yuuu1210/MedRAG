"""
评估模型在 nsclc 上的 performance
"""
import json
import os

from src.llm_clients import doubao_client, qwen_client
from src.utils import logger_config

logger = logger_config.get_logger(__name__)

# llm = bytedance.DoubaoChat()
llm = qwen_client.QwenChat()


def evaluate_text_questions_performance():
    """
    评估 text_questions.json 中的问题
    """
    question_file = "./data/nsclc/all_text_questions.json"
    answer_sheet_file = "./data/nsclc/doubao_text_question_answer_sheet.json"

    # 尝试加载结果文件
    try:
        with open(answer_sheet_file, "r", encoding="utf-8") as f:
            answer_sheet = json.load(f)
            # 使用 set 加速查找
            processed_question_ids = {item["id"] for item in answer_sheet}
            correct_count = int(answer_sheet[-1]["correct_count"])
            total_questions = int(answer_sheet[-1]["total_questions"])
    except (FileNotFoundError, IndexError, json.JSONDecodeError):
        answer_sheet = []
        processed_question_ids = set()
        correct_count = 0
        total_questions = 0

    with open(question_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for question in data:
        question_id = question["id"]
        question_text = question["question"]
        options = question["options"]
        correct_answer = question["answer"]

        if question_id in processed_question_ids:
            logger.info("question %d processed, skip", question_id)
            continue  # 跳过已处理的问题

        logger.info("processing question %d", question_id)
        response = llm.answer_text_questions(question_text, options)
        llm_answer = response.get("answer")

        if llm_answer == correct_answer:
            correct_count += 1
        total_questions += 1

        # 将结果保存到 answer_sheet 列表中
        answer_sheet.append({
            "id": question_id,
            "question": question_text,
            "options": options,
            "answer": correct_answer,
            "llm_response": response,
            "correct_count": correct_count,
            "total_questions": total_questions
        })

        # 实时将 answer_sheet 保存到 JSON 文件
        with open(answer_sheet_file, "w", encoding="utf-8") as f:
            json.dump(answer_sheet, f, indent=4, ensure_ascii=False)

        if total_questions > 0:  # 避免除以 0 错误
            logger.info("current accuracy: %d/%d", correct_count,
                        total_questions)
        else:
            logger.info("current accuracy: 0.0")


def evaluate_img_questions_performance():
    """
    评估 img_questions.json 中的问题
    """
    question_file = "./data/nsclc/all_img_questions.json"
    answer_sheet_file = "./data/nsclc/doubao_img_question_answer_sheet.json"

    # 检查结果文件是否存在，如果存在则加载
    if os.path.exists(answer_sheet_file):
        with open(answer_sheet_file, "r", encoding="utf-8") as f:
            try:
                results_data = json.load(f)
            except json.JSONDecodeError:
                # 如果 JSON 文件为空或损坏，则初始化为空列表
                results_data = []
            processed_question_ids = {
                item["question_id"] for item in results_data
            }  # 使用 set 加速查找
    else:
        results_data = []
        processed_question_ids = set()

    with open(question_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    correct_count = 0
    total_questions = 0
    for question in data:
        question_id = question["question_id"]
        question_text = question["question"]
        img_path = question["img_path"]
        options = question["options"]
        correct_answer = question["correct_answer"]

        if question_id in processed_question_ids:
            logger.info("question %d processed, skip", question_id)
            continue  # 跳过已处理的问题

        logger.info("processing question %d", question_id)
        response = llm.answer_img_questions(question_text, img_path, options)

        llm_answer = response.get("answer")
        is_correct = False
        if llm_answer == correct_answer:
            is_correct = True
            correct_count += 1
        total_questions += 1

        # 将结果保存到 results_data 列表中
        results_data.append({
            "question_id": question_id,
            "question": question_text,
            "img_path": img_path,
            "options": options,
            "correct_answer": correct_answer,
            "llm_response": response,
            "llm_answer": llm_answer,
            "is_correct": is_correct,
            "correct_count": correct_count,
            "total_questions": total_questions
        })

        # 实时将 results_data 保存到 JSON 文件
        with open(answer_sheet_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=4, ensure_ascii=False)

        if total_questions > 0:  # 避免除以 0 错误
            logger.info("current accuracy: %d/%d", correct_count,
                        total_questions)
        else:
            logger.info("current accuracy: 0.0")

    if total_questions > 0:  # 避免除以 0 错误
        logger.info("\nfinal accuracy: %d/%d", correct_count, total_questions)
    else:
        logger.info("\nfinal accuracy: 0.0")


def main():
    """
    主程序入口
    """
    # evaluate_text_questions_performance()
    # evaluate_img_questions_performance()


if __name__ == "__main__":
    main()

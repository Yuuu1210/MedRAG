"""
评估模型在 healthfc 上的 performance
"""
import json
import os

from src.llm_clients import doubao_client, qwen_client
from src.utils import logger_config

logger = logger_config.get_logger(__name__)

llm = doubao_client.DoubaoChat()
# llm = ali.QwenChat()


def evaluate_healthfc():
    """
    评估 healthfc 数据集上的问题
    """
    query_file = "./data/dataset/healthfc/doubao_query.json"
    results_file = "./data/dataset/healthfc/qwen_direct_results.json"
    # 用于提示每个选项的含义
    label_map = {0: "Supported", 1: "Not enough information", 2: "Refuted"}

    with open(query_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            result = json.load(f)
        cor_count = result[-1]["correct_count"]
    else:
        result = []
        cor_count = 0

    # 限制 query_id 在 [mid_id, max_id] 之间
    min_id = 1
    max_id = 750
    for item in data:
        query_id = item["id"]
        if query_id < min_id:
            continue
        if query_id > max_id:
            break

        logger.info("-" * 50)
        logger.info("processing query %d", query_id)
        query = item["query"]
        answer = int(item["answer"])

        response = llm.fc_answer_query(query)
        if response.get("answer") == answer:
            cor_count += 1
        logger.info("correct_count:%d", cor_count)
        logger.info("total_questions:%d", query_id)

        result.append({
            "id": query_id,
            "query": query,
            "answer": answer,
            "response": response,
            "correct_count": cor_count,
        })

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)


def evaluate_healthfc_inverted_index():
    """
    使用 inverted_index 方法评估 healthfc 数据集
    """
    query_file = "./data/dataset/healthfc/doubao_query.json"
    results_file = "./data/dataset/healthfc/doubao_inverted_index_results.json"
    final_score_dir = "./data/knowledge_base/healthfc/inverted_index_final_score/"

    with open(query_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            result = json.load(f)
        cor_count = result[-1]["correct_count"]
    else:
        result = []
        cor_count = 0

    # 限制 query_id 在 [mid_id, max_id] 之间
    min_id = 1
    max_id = 750
    for item in data:
        query_id = item["id"]
        if query_id < min_id:
            continue
        if query_id > max_id:
            break

        logger.info("-" * 50)
        logger.info("processing query %d", query_id)
        query = item["query"]
        answer = int(item["answer"])

        summary_file = os.path.join(final_score_dir, f"{query_id}_summary.json")
        if not os.path.exists(summary_file):
            context = []
        else:
            with open(summary_file, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
            context = [item.get("summary") for item in summary_data]

        response = llm.fc_answer_query_rag(query, context)
        if response.get("answer") == answer:
            cor_count += 1
        logger.info("correct_count:%d", cor_count)
        logger.info("total_questions:%d", query_id)

        result.append({
            "id": query_id,
            "query": query,
            "answer": answer,
            "context": context,
            "response": response,
            "correct_count": cor_count,
        })

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)


def main():
    """
    主程序入口
    """
    # evaluate_healthfc()
    evaluate_healthfc_inverted_index()


if __name__ == "__main__":
    main()

"""
评估模型在 medmcqa 上的 performance
"""
import json
import os

from src.llm_clients import doubao_client, qwen_client
from src.utils import logger_config

# llm = bytedance.DoubaoChat()
llm = qwen_client.QwenChat()

logger = logger_config.get_logger(__name__)


def evaluate_medmcqa():
    """
    评估 MedMCQA 数据集上的问题
    """
    query_file = "./experiments/medmcqa/jsons/doubao_query.json"
    results_file = "./experiments/medmcqa/results/doubao/doubao_results_3.json"

    with open(query_file, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            result = json.load(f)
        cor_count = result[-1]["cor_count"]
    else:
        result = []
        cor_count = 0

    # 限制 query_id 在 [mid_id, max_id] 之间
    min_id = 1
    max_id = 1000
    processed_ids = set()
    for item in query_data:
        query_id = item["id"]
        if query_id < min_id or query_id in processed_ids:
            query_id += 1
            continue
        if query_id > max_id:
            break

        logger.info("-" * 50)
        logger.info("processing query %d", query_id)
        query = item["query"]
        options = item["options"]
        answer = item["answer"]

        response = llm.qa_answer_query(query, options)
        if response.get("answer") == answer:
            cor_count += 1
        logger.info("accuracy: %.2f", cor_count / query_id * 100)

        result.append({
            "id": query_id,
            "query": query,
            "options": options,
            "answer": answer,
            "response": response,
            "cor_count": cor_count,
        })

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        processed_ids.add(query_id)
        query_id += 1


def evaluate_medmcqa_rag():
    """
    使用 rag 方法评估 medmcqa 数据集
    """
    query_file = "./experiments/medmcqa/jsons/doubao_query.json"
    results_file = "./experiments/medmcqa/results/qwen/qwen_textbook_faiss_index_topk_7_results_3.json"
    summary_dir = "./experiments/medmcqa/scores/textbook/en_faiss_index/topk_7/"

    with open(query_file, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        cor_count = results[-1]["cor_count"]
    else:
        results = []
        cor_count = 0

    # 限制 query_id 在 [mid_id, max_id] 之间
    min_id = 1
    max_id = 1000
    processed_ids = set()
    for item in query_data:
        query_id = item["id"]
        if query_id < min_id or query_id in processed_ids:
            continue
        if query_id > max_id:
            break

        logger.info("-" * 50)
        logger.info("processing query %d", query_id)
        query = item["query"]
        options = item["options"]
        answer = item["answer"]

        summary_file = os.path.join(summary_dir, f"{query_id}_summary.json")
        if not os.path.exists(summary_file):
            contexts = []
        else:
            with open(summary_file, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
            contexts = [item.get("summary") for item in summary_data]

        response = llm.qa_answer_query_rag(query, options, contexts)
        if response.get("answer") == answer:
            cor_count += 1
        logger.info("accuracy: %.2f", cor_count / query_id * 100)

        results.append({
            "id": query_id,
            "query": query,
            "options": options,
            "answer": answer,
            "contexts": contexts,
            "response": response,
            "cor_count": cor_count,
        })

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        processed_ids.add(query_id)


def evaluate_medmcqa_rag_wo_summary():
    """
    使用 rag 方法评估 medmcqa 数据集, 不使用 summary
    """
    query_file = "./experiments/medmcqa/jsons/doubao_query.json"
    results_file = "./experiments/medmcqa/results/qwen/qwen_ddg_textbook_alpha_07_topk_3_wo_summary_results_3.json"
    node_and_neighbors_dir = "./experiments/medmcqa/scores/ddg_textbook/alpha_07_topk_3/"

    with open(query_file, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        cor_count = results[-1]["cor_count"]
    else:
        results = []
        cor_count = 0

    # 限制 query_id 在 [mid_id, max_id] 之间
    min_id = 1
    max_id = 1000
    processed_ids = set()
    for item in query_data:
        query_id = item["id"]
        if query_id < min_id or query_id in processed_ids:
            continue
        if query_id > max_id:
            break

        logger.info("-" * 50)
        logger.info("processing query %d", query_id)
        query = item["query"]
        options = item["options"]
        answer = item["answer"]

        node_and_neighbors_file = os.path.join(
            node_and_neighbors_dir, f"{query_id}_node_and_neighbors.json")
        if not os.path.exists(node_and_neighbors_file):
            contexts = []
        else:
            with open(node_and_neighbors_file, "r", encoding="utf-8") as f:
                node_and_neighbors_data = json.load(f)

            contexts = []
            for item in node_and_neighbors_data:
                node = item["node"]
                neighbors = item["neighbors"]

                contexts.append(node["chunk"])
                for neighbor in neighbors:
                    contexts.append(neighbor["chunk"])

        response = llm.qa_answer_query_rag(query, options, contexts)
        if response.get("answer") == answer:
            cor_count += 1
        logger.info("accuracy: %.2f", cor_count / query_id * 100)

        results.append({
            "id": query_id,
            "query": query,
            "options": options,
            "answer": answer,
            "contexts": contexts,
            "response": response,
            "cor_count": cor_count,
        })

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        processed_ids.add(query_id)


def main():
    """
    主程序入口
    """
    # evaluate_medmcqa()
    evaluate_medmcqa_rag()
    # evaluate_medmcqa_rag_wo_summary()


if __name__ == "__main__":
    main()

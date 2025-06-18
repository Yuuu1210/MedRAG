"""
重写 user query
"""
import csv
import json
import os

import requests

from src.llm_clients import doubao_client
from src.utils import logger_config

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/91.0.4472.124 Safari/537.36",
})

llm = doubao_client.DoubaoClient()

logger = logger_config.get_logger(__name__)


def qa_build_pubmed_ddg_queries():
    """
    (qa dataset) 重写 user query 为 pubmed query 和 duckduckgo query
    """
    normalized_entity_file = "./experiments/medmcqa/jsons/doubao_query_entities_normalized_threshold_085.json"
    query_file = "./experiments/medmcqa/jsons/doubao_query751_1000.json"

    with open(normalized_entity_file, "r", encoding="utf-8") as f:
        normalized_entity_data = json.load(f)

    if os.path.exists(query_file):
        with open(query_file, "r", encoding="utf-8") as f:
            result = json.load(f)
    else:
        result = []

    min_id = 751
    max_id = 1000
    for item in normalized_entity_data:
        query_id = item["id"]
        if query_id < min_id:
            continue
        if query_id > max_id:
            break

        logger.info("-" * 50)
        query = item["query"]
        options = item["options"]
        cop = item["cop"]

        # 对每个 option 重写 query
        pubmed_queries = []
        ddg_queries = []
        for option_letter, option_text in options.items():
            logger.info("processing query %d option %s", query_id,
                        option_letter)
            pubmed_query = llm.qa_rewrite_pubmed_query(
                query,
                option_text,
                item["query_entity_to_mesh"],
                item[f"op{option_letter.lower()}_entity_to_mesh"],
            )["pubmed_query"]
            ddg_query = llm.qa_rewrite_ddg_query(
                query,
                option_text,
                list(item["query_entity_to_mesh"].keys()),
                list(item[f"op{option_letter.lower()}_entity_to_mesh"].keys()),
            )["ddg_query"]
            pubmed_queries.append(pubmed_query)
            ddg_queries.append(ddg_query)

        result.append({
            "id": query_id,
            "query": query,
            "options": options,
            "cop": cop,
            "pubmed_queries": pubmed_queries,
            "ddg_queries": ddg_queries
        })

        with open(query_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)


def fc_build_pubmed_ddg_queries():
    """
    重写 user query 为 pubmed query 和 duckduckgo query
    """
    dataset_file = "./data/dataset/healthfc/healthFC_annotated.csv"
    query_file = "./data/dataset/healthfc/doubao_query.json"
    # 用于提示每个选项的含义
    label_map = {0: "Supported", 1: "Not enough information", 2: "Refuted"}

    if os.path.exists(query_file):
        with open(query_file, "r", encoding="utf-8") as f:
            result = json.load(f)
    else:
        result = []

    query_id = 1
    min_id = 11
    max_id = 1000
    with open(dataset_file, "r", newline="", encoding="utf-8") as f:
        dataset_data = csv.DictReader(f)

        for row in dataset_data:
            if query_id < min_id:
                query_id += 1
                continue
            if query_id > max_id:
                break

            logger.info("-" * 50)
            logger.info("processing question %d", query_id)
            question = row["en_claim"]
            answer = row["label"]

            # pubmed_query 由 llm 给出 keywords, 后续 AND 连接
            pubmed_query = llm.fc_rewrite_pubmed_query(question)["keywords"]
            # ddg_query 由 llm 直接给出
            ddg_query = llm.fc_rewrite_ddg_query(question)["query"]

            result.append({
                "id": query_id,
                "question": question,
                "answer": answer,
                "pubmed_query": pubmed_query,
                "ddg_query": ddg_query
            })

            with open(query_file, "w", encoding="utf-8") as file:
                json.dump(result, f, indent=4, ensure_ascii=False)

            query_id += 1


def main():
    """
    主函数
    """
    qa_build_pubmed_ddg_queries()
    # fc_build_pubmed_ddg_queries()


if __name__ == "__main__":
    main()

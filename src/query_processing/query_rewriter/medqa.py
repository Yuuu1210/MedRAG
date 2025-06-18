"""
重写 medqa 数据集 query
"""
import json
import os

from src.llm_clients import doubao_client
from src.utils import logger_config

llm = doubao_client.DoubaoChat()

logger = logger_config.get_logger(__name__)


def build_ddg_query():
    """
    (qa dataset) 重写 user query 为 ddg_query
    不使用 normalized_entity
    """
    dataset_file = "./data/dataset/medqa/questions/US/test.jsonl"
    query_file = "./experiments/medqa/jsons/doubao_query.json"

    if os.path.exists(query_file):
        with open(query_file, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []

    min_id = 392
    max_id = 1000

    with open(dataset_file, "r", encoding="utf-8") as f:
        for query_id, line in enumerate(f, 1):
            if query_id < min_id:
                continue
            if query_id > max_id:
                break

            logger.info("processing %d", query_id)
            # 获取 query 信息
            data = json.loads(line)
            query = data["question"]
            options = data["options"]
            answer = data["answer_idx"]

            # 对每个 option 重写 query
            ddg_queries = []
            for option_letter, option_text in options.items():
                ddg_query = llm.qa_rewrite_ddg_query(query,
                                                     option_text)["query"]
                ddg_queries.append(ddg_query)

            results.append({
                "id": query_id,
                "query": query,
                "options": options,
                "answer": answer,
                "ddg_query": ddg_queries
            })

            with open(query_file, "w", encoding="utf-8") as query_f:
                json.dump(results, query_f, indent=4, ensure_ascii=False)


def main():
    """
    主程序入口
    """
    build_ddg_query()


if __name__ == "__main__":
    main()

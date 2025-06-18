"""
为数据集中的 query 选择 en textbooks
"""
import json
import os

from src.llm_clients import doubao_client
from src.utils import logger_config

logger = logger_config.get_logger(__name__)

llm = doubao_client.DoubaoClient()


def qa_select_en_textbooks():
    """
    (qa dataset) 选择 en 教科书
    """
    dataset_file = "./data/dataset/medmcqa/data/dev.jsonl"
    textbook_file = "./experiments/medmcqa/jsons/doubao_en_textbooks.json"
    cop_map = {1: "A", 2: "B", 3: "C", 4: "D"}

    with open(dataset_file, "r", encoding="utf-8") as f:
        data = f.readlines()

    if os.path.exists(textbook_file):
        with open(textbook_file, "r", encoding="utf-8") as f:
            result = json.load(f)
    else:
        result = []

    # 限制 query_id 在 [mid_id, max_id] 之间
    min_id = 1
    max_id = 1000
    query_id = 1
    for line in data:
        if query_id < min_id:
            query_id += 1
            continue
        if query_id > max_id:
            break

        logger.info("-" * 50)
        logger.info("processing query %d", query_id)
        item = json.loads(line)
        query = item["question"]
        options = {
            "A": item["opa"],
            "B": item["opb"],
            "C": item["opc"],
            "D": item["opd"]
        }
        answer = cop_map[item["cop"]]

        response = llm.qa_select_en_textbooks(query, options)
        result.append({
            "id": query_id,
            "query": query,
            "options": options,
            "answer": answer,
            "response": response,
            "textbooks": response["textbooks"]
        })

        with open(textbook_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        query_id += 1


def fc_select_en_textbooks():
    """
    (fact checking dataset) 选择 en 教科书
    """
    dataset_file = "./data/dataset/healthfc/doubao_query.json"
    textbook_file = "./data/dataset/healthfc/doubao_en_textbooks.json"

    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if os.path.exists(textbook_file):
        with open(textbook_file, "r", encoding="utf-8") as f:
            result = json.load(f)
    else:
        result = []

    # 限制 query_id 在 [mid_id, max_id] 之间
    min_id = 2
    max_id = 750
    for item in data:
        query_id = item["id"]
        if query_id < min_id:
            query_id += 1
            continue
        if query_id > max_id:
            break

        logger.info("-" * 50)
        logger.info("processing query %d", query_id)
        query = item["query"]
        answer = item["answer"]

        response = llm.fc_select_en_textbooks(query)
        result.append({
            "id": query_id,
            "query": query,
            "answer": answer,
            "response": response,
            "textbooks": response["textbooks"]
        })

        with open(textbook_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        query_id += 1


def main():
    """
    主程序入口
    """
    qa_select_en_textbooks()
    # fc_select_en_textbooks()


if __name__ == "__main__":
    main()

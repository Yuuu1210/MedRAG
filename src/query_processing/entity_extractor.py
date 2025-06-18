"""
提取 query 中的 biomedical entity
"""
import json

from src.llm_clients import doubao_client

llm = doubao_client.DoubaoClient()


def medmcqa_extract_query_entities():
    """
    处理 medmcqa 数据集中的 query, 提取出 query 中的 biomedical entity
    """
    dataset_file = "./data/dataset/medmcqa/data/dev.jsonl"
    entities_file = "./experiments/medmcqa/jsons/doubao_query_entities.json"
    cop_to_answer = {1: "A", 2: "B", 3: "C", 4: "D"}
    min_id = 1
    max_id = 1000

    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f]

    results = []
    for query_id, item in enumerate(dataset, 1):
        if query_id < min_id:
            continue
        if query_id > max_id:
            break

        print(f"processing query {query_id}")
        query = item["question"]
        options = {
            "A": item["opa"],
            "B": item["opb"],
            "C": item["opc"],
            "D": item["opd"]
        }
        cop = cop_to_answer[item["cop"]]

        # 提取 query 中的 biomedical entity
        query_entities = llm.extract_biomedical_entities(query)["entities"]
        opa_entites = llm.extract_biomedical_entities(options["A"])["entities"]
        opb_entites = llm.extract_biomedical_entities(options["B"])["entities"]
        opc_entites = llm.extract_biomedical_entities(options["C"])["entities"]
        opd_entites = llm.extract_biomedical_entities(options["D"])["entities"]

        results.append({
            "id": query_id,
            "query": query,
            "options": options,
            "cop": cop,
            "query_entities": query_entities,
            "opa_entities": opa_entites,
            "opb_entities": opb_entites,
            "opc_entities": opc_entites,
            "opd_entities": opd_entites
        })

        with open(entities_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


def main():
    """
    主函数
    """
    medmcqa_extract_query_entities()


if __name__ == "__main__":
    main()

"""
对文本生成总结, 对 image 筛选去掉无关 image
"""
import json
import os

from src.llm_clients import doubao_client, doubao_vis_client

llm = doubao_client.DoubaoClient()
llm_vis = doubao_vis_client.DoubaoVisClient()


def summarize_text(item: dict) -> list[dict]:
    """
    item 中 dict 键包含 id, query, options

    Returns:
        list[dict]: dict 键包含 type, summary
    """
    # 每次 summary 总结的 chunk 数量
    summary_batch_size = 4

    query_id = item["id"]
    query = item["query"]
    options = item["options"]

    text_file = f"./experiments/medmcqa/retrieval/{query_id}_text.json"
    if not os.path.exists(text_file):
        print(f"Text file {text_file} does not exist.")
        return []

    with open(text_file, "r", encoding="utf-8") as f:
        text_data = json.load(f)

    # 移除一些 key 以适配 summarize_chunks 的输入格式
    keys_to_remove = ["source_type", "score", "rerank_score"]
    new_data = [{
        k: v for k, v in item.items() if k not in keys_to_remove
    } for item in text_data]

    results = []
    data_len = len(new_data)
    for i in range(0, data_len, summary_batch_size):
        print(f"processing index {i} - {i + summary_batch_size}")

        response = llm.summarize_chunks(
            query,
            options,
            new_data[i:i + summary_batch_size],
        )

        if float(response.get("score")) >= 0.5:
            results.append({
                "type": "text",
                "summary": response.get("summary"),
            })

    return results


def summarize_image(item: dict) -> list[dict]:
    """
    item 中 dict 键包含 id, query, options

    Returns:
        list[dict]: dict 键包含 type, title, path
    """
    query_id = item["id"]
    query = item["query"]
    options = item["options"]

    image_file = f"./experiments/medmcqa/retrieval/{query_id}_image.json"
    if not os.path.exists(image_file):
        print(f"Image file {image_file} does not exist.")
        return []

    with open(image_file, "r", encoding="utf-8") as f:
        image_data = json.load(f)

    print("processing images...")
    results = []
    for image in image_data:
        response = llm_vis.is_image_relevant(query, options, image)

        if float(response.get("score", 0)) >= 0.5 and response.get("relevant"):
            results.append({
                "type": "image",
                "title": image["title"],
                "path": image["path"],
            })

    return results


def medmcqa_summarize():
    """
    对 MedMCQA 数据集的 text, image 进行总结
    """
    query_file = "./experiments/medmcqa/jsons/doubao_query.json"
    min_id = 1
    max_id = 1

    with open(query_file, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    for item in query_data:
        query_id = item["id"]
        if query_id < min_id:
            continue
        if query_id > max_id:
            break

        print("processing query id:", query_id)
        results = summarize_text(item)
        results.extend(summarize_image(item))

        summary_file = f"./experiments/medmcqa/summary/{query_id}_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


def main():
    """
    主函数
    """
    medmcqa_summarize()


if __name__ == "__main__":
    main()

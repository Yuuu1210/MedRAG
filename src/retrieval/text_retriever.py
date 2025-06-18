"""
使用 bm25 和 dpr 混合搜索获取相关文本
"""
import json
import os
import re

from FlagEmbedding import FlagReranker

import settings
from src.retrieval import bm25, dpr

# Setting use_fp16 to True speeds up computation with a slight performance degradation
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)


def find_content_list(root_dir: str) -> list[str]:
    """
    (Helper) 从 root_dir 中找到 content_list.json

    Args:
        root_dir: 目录
    
    Returns:
        list[str]: content_list.json 的路径列表
    """
    content_list_paths = []

    for dir_path, _, file_names in os.walk(root_dir):
        for file_name in file_names:
            if file_name.endswith("content_list.json"):
                file_path = os.path.join(dir_path, file_name)
                content_list_paths.append(file_path)

    return content_list_paths


def is_pubmed_content_list(content_list: str):
    """
    (Helper) 根据 content_list.json 文件名判断是否来自 pubmed
    """
    file_name = os.path.basename(content_list)
    pattern = r"^PMID\d{8}\_content_list.json$"
    match = re.fullmatch(pattern, file_name, re.IGNORECASE)
    return bool(match)


def get_pubmed_ddg_chunks(docs_dir: str) -> list[dict]:
    """
    (Helper) 获取 pubmed, ddg 来源的 chunk

    Args:
        docs_dir (str): 文档目录路径
    
    Returns:
        list[dict]
    """
    if not os.path.exists(docs_dir):
        return ""

    content_lists = find_content_list(docs_dir)

    results = []
    for content_list in content_lists:
        if is_pubmed_content_list(content_list):
            source_type = "pubmed"
        else:
            source_type = "ddg"

        with open(content_list, "r", encoding="utf-8") as file:
            content = json.load(file)

        texts = ""
        for item in content:
            if item["type"] != "text":
                continue
            texts += item["text"]

        chunks = dpr.split_into_chunks(texts)

        for chunk in chunks:
            results.append({
                "chunk": chunk,
                "source": content_list,
                "source_type": source_type
            })

    return results


def get_textbook_chunks(query_id: int):
    """
    获取 textbook 来源的 chunk
    """
    textbook_file = "./experiments/medmcqa/jsons/doubao_en_textbooks.json"
    results = []

    with open(textbook_file, "r", encoding="utf-8") as file:
        textbook_data = json.load(file)

    for item in textbook_data:
        if item["id"] == query_id:
            textbook_categories = item["textbooks"]
            break

    if not textbook_categories:
        print(f"query {query_id} don't have relevant textbook.")
        return []

    textbook_paths = []
    for item in settings.EN_TEXTBOOKS:
        if item["category"] in textbook_categories:
            textbook_paths.extend(item["file_path"])

    for path in textbook_paths:
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()

        chunks = dpr.split_into_chunks(content)

        for chunk in chunks:
            results.append({
                "chunk": chunk,
                "source": path,
                "source_type": "textbook"
            })

    return results


def rerank_corpus(query: str, corpus: list[dict], k: int = 32):
    """
    (Helper) 使用 reranker 计算 [query, chunk] 的 score 并重排

    Args:
        query (str): 查询语句。
        corpus (list[dict]): 待重排的文档列表。
                             每个字典至少需要包含一个 'chunk' 键，其值为文档文本。
                             例如: [{'chunk': 'text1', 'source': 'pubmed'}, {'chunk': 'text2', 'source': 'textbook'}]
        reranker_model (FlagReranker): 已初始化的 FlagReranker 模型实例。
        k (int, optional): 返回的重排后文档数量。默认为 32。
        normalize_score (bool, optional): 是否将分数归一化到 0-1 范围。默认为 False。

    Returns:
        list[dict]: 重排后的文档列表，每个字典会新增一个 'rerank_score' 键。
    """
    if not corpus:
        return []

    # 1. 准备 reranker 的输入格式：[[query, chunk1_text], [query, chunk2_text], ...]
    sentence_pairs = []
    for doc_info in corpus:
        # 确保每个字典中都有 'chunk' 键，并且其值是字符串
        if 'chunk' in doc_info and isinstance(doc_info['chunk'], str):
            sentence_pairs.append([query, doc_info['chunk']])
        else:
            # 可以选择跳过格式不正确的条目或抛出错误
            print(f"警告: 文档信息 {doc_info} 缺少 'chunk' 键或其值不是字符串，已跳过。")
            continue

    if not sentence_pairs:  # 如果所有条目都格式不正确
        return []

    # 2. 使用 reranker 计算分数
    # reranker.compute_score() 返回一个分数列表，与 sentence_pairs 中的顺序对应
    scores = reranker.compute_score(sentence_pairs, normalize=True)

    # 3. 将分数与原始文档信息结合
    # 我们需要确保分数与正确的文档对应起来。
    # 由于我们是按顺序构建 sentence_pairs 的，所以 scores 的顺序也对应 corpus 中有效条目的顺序。

    results_with_scores = []
    score_idx = 0  # 用于追踪 scores 列表的索引
    for doc_info in corpus:
        # 再次检查，只为那些成功创建了 pair 的文档添加分数
        if 'chunk' in doc_info and isinstance(doc_info['chunk'], str):
            # 创建一个新的字典副本，以避免修改原始 corpus 中的字典（如果它是从其他地方传递的）
            # 或者直接修改 doc_info，取决于你的需求
            updated_doc_info = doc_info.copy()  # 创建副本
            updated_doc_info['rerank_score'] = scores[score_idx]
            results_with_scores.append(updated_doc_info)
            score_idx += 1
            if score_idx >= len(scores):  # 以防万一，虽然理论上不应该发生
                break

    # 4. 根据 rerank_score 降序排序
    results_with_scores.sort(key=lambda x: x['rerank_score'], reverse=True)

    # 5. 返回前 k 个结果
    return results_with_scores[:k]


def medmcqa_hybrid_search():
    """
    bm25 和 dpr 混合搜索
    """
    query_file = "./experiments/medmcqa/jsons/doubao_query.json"
    min_id = 1
    max_id = 1

    with open(query_file, "r", encoding="utf-8") as file:
        query_data = json.load(file)

    for item in query_data:
        query_id = item["id"]
        query = item["query"]
        options = item["options"]

        query_and_options = query
        for _, option_text in options.items():
            query_and_options += " " + option_text

        if query_id < min_id:
            continue
        if query_id > max_id:
            break

        print(f"Processing query {query_id}")
        docs_dir = f"./experiments/medmcqa/docs/{query_id}/"

        print("chunking...")
        corpus = []
        corpus.extend(get_pubmed_ddg_chunks(docs_dir))
        corpus.extend(get_textbook_chunks(query_id))

        # results 为 list[dict], dict 包含键 chunk, source, source_type, score
        print("retrieving...")
        bm25_results = bm25.bm25_retrieve(query_and_options, corpus, k=32)
        dpr_results = dpr.dpr_retrieve(query_and_options, corpus, k=32)

        # 合并 bm25 和 dpr 的结果并去重
        combined_results = {
            doc['chunk']: doc for doc in bm25_results + dpr_results
        }.values()
        combined_results = list(combined_results)

        print("reranking...")
        reranked_results = rerank_corpus(query_and_options,
                                         combined_results,
                                         k=32)

        print("saving results...")
        text_file = f"./experiments/medmcqa/retrieval/{query_id}_text.json"
        with open(text_file, "w", encoding="utf-8") as file:
            json.dump(reranked_results, file, ensure_ascii=False, indent=4)


def main():
    """
    主函数
    """
    medmcqa_hybrid_search()


if __name__ == "__main__":
    main()

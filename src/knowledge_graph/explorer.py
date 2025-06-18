"""
从 Knowledge graph 中获取 query 相关信息
"""
import json
import os

from src.llm_clients import doubao_client
# from src.document import qa_calc_score
from src.utils import logger_config

# import faiss
# import numpy as np
# import torch
# from sentence_transformers import SentenceTransformer

# 组合 sim_score 和 ner_score 的权重
ALPHA = 0.3
TOP_K = 3

llm = doubao_client.DoubaoChat()

logger = logger_config.get_logger(__name__)


def get_rel_score_data(base_score_file: str):
    """
    (Helper) 获取 base_score_file 对应的 rel_score_data
    """
    with open(base_score_file, "r", encoding="utf-8") as f:
        base_score_data = json.load(f)

    results = []
    for item in base_score_data:
        item["rel_score"] = ALPHA * item["sim_score"] + (
            1 - ALPHA) * item["ner_score"]
        results.append(item)

    return results


def get_query_and_options(query_id: int):
    """
    (Helper) 根据 query_id 获取 query 和 options
    """
    query_file = "./experiments/medmcqa/jsons/doubao_query.json"
    with open(query_file, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    for item in query_data:
        if item["id"] == query_id:
            return item["query"], item["options"]


def get_chunks_and_source(node_and_neighbors_dir: str, query_id: int):
    """
    (Helper) 根据 query_id 获取 chunks 和 source
    """
    node_and_neighbors_file = os.path.join(
        node_and_neighbors_dir, f"{query_id}_node_and_neighbors.json")
    chunks_and_source = []

    with open(node_and_neighbors_file, "r", encoding="utf-8") as f:
        node_and_neighbors_data = json.load(f)

    for item in node_and_neighbors_data:
        node = item["node"]
        neighbors = item["neighbors"]

        # 处理 node
        base_name = os.path.basename(node["source"])
        if node["source_type"] == "ddg":
            source = base_name.replace("_content_list.json", ".pdf")
        else:
            source = base_name.replace(".json", ".txt")

        chunks_and_source.append({
            "chunk": node["chunk"],
            "source": source,
        })

        # 处理 neighbors
        for neighbor in neighbors:
            base_name = os.path.basename(neighbor["source"])
            if neighbor["source_type"] == "ddg":
                source = base_name.replace("_content_list.json", ".pdf")
            else:
                source = base_name.replace(".json", ".txt")

            chunks_and_source.append({
                "chunk": neighbor["chunk"],
                "source": source
            })

    return chunks_and_source


def merge_faiss_index_and_metadata(textbooks: list[str]):
    """
    (Helper) 合并不同 category 的 faiss index, metadata
    """
    faiss_index_dir = "./data/textbook/en_faiss_index/"

    merged_vectors = []
    merged_metadata = []
    for category in textbooks:
        faiss_index_file = os.path.join(faiss_index_dir, f"{category}.index")
        metadata_file = os.path.join(faiss_index_dir,
                                     f"{category}_metadata.json")

        # 合并 vectors
        cur_index = faiss.read_index(faiss_index_file)
        cur_vectors = cur_index.reconstruct_n(0, cur_index.ntotal)
        merged_vectors.append(cur_vectors)

        # 合并 metadata
        with open(metadata_file, "r", encoding="utf-8") as f:
            cur_metadata = json.load(f)
        merged_metadata.extend(cur_metadata)

    # 后续处理 vectors
    final_vectors = np.vstack(merged_vectors)
    merged_index = faiss.IndexFlatIP(final_vectors.shape[1])
    merged_index.add(final_vectors)

    return merged_index, merged_metadata


def get_top_k_with_neighbors_faiss_index():
    """
    仅 textbook 来源, faiss_index 方法的 top_k nodes_and_neighbors
    用于比较 inverted_index
    """
    textbook_file = "./experiments/medmcqa/jsons/doubao_en_textbooks.json"
    node_and_neighbors_dir = f"./experiments/medmcqa/scores/textbook/en_faiss_index/topk_{TOP_K}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2",
                                device=device)

    min_id = 1
    max_id = 1000

    with open(textbook_file, "r", encoding="utf-8") as f:
        textbook_data = json.load(f)

    for item in textbook_data:
        query_id = item["id"]
        query = item["query"]
        options = item["options"]
        textbooks = item["textbooks"]

        if query_id < min_id:
            continue
        if query_id > max_id:
            break
        if textbooks is None or len(textbooks) == 0:
            logger.info("textbooks is None or empty")
            continue

        logger.info("-" * 50)
        logger.info("processing query %d", query_id)
        # 获取 query_and_options, 用于 faiss search
        query_and_options = query
        for option_letter, option_text in options.items():
            query_and_options += f" {option_letter}: {option_text}"

        # 合并不同 category 的 faiss index
        index, metadata = merge_faiss_index_and_metadata(textbooks)

        # 将 query_and_options 转换为向量
        query_emb = model.encode(query_and_options, convert_to_numpy=True)
        if query_emb.dtype != 'float32':
            query_emb = query_emb.astype('float32')
        # 归一化
        query_emb = query_emb.reshape(1, -1)
        faiss.normalize_L2(query_emb)

        # 获取 top_k 的 nodes
        D1, I1 = index.search(query_emb, TOP_K)

        # 整理结果
        results = []
        for i, node_idx in enumerate(I1[0]):
            # 获取当前节点信息
            node = metadata[node_idx].copy()
            node["score"] = float(D1[0][i])

            # 以 node["chunk"] 为 query 获取邻居
            node_emb = model.encode(node["chunk"], convert_to_numpy=True)
            if node_emb.dtype != 'float32':
                node_emb = node_emb.astype('float32')
            # 归一化
            node_emb = node_emb.reshape(1, -1)
            faiss.normalize_L2(node_emb)

            neighbors = []
            # 取 top_k + 1 是因为第一个是自身
            D2, I2 = index.search(node_emb, TOP_K + 1)

            for j, neighbor_idx in enumerate(I2[0]):
                if neighbor_idx == node_idx:
                    continue

                # 获取邻居信息
                neighbor = metadata[neighbor_idx].copy()
                neighbor["score"] = float(D2[0][j])
                neighbors.append(neighbor)

            results.append({"node": node, "neighbors": neighbors})

        # 保存结果
        node_and_neighbors_file = os.path.join(
            node_and_neighbors_dir, f"{query_id}_node_and_neighbors.json")
        with open(node_and_neighbors_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


def get_top_k_with_neighbors_inverted_index():
    """
    仅 textbook 来源, faiss_index 方法的 top_k nodes_and_neighbors
    用于比较 faiss index 
    """
    base_score_dir = "./experiments/medmcqa/scores/ddg_textbook/base_score/"
    node_and_neighbors_dir = f"./experiments/medmcqa/scores/textbook/en_inverted_index/alpha_{int(ALPHA * 10):02d}_topk_{TOP_K}/"
    os.makedirs(node_and_neighbors_dir, exist_ok=True)

    min_id = 1
    max_id = 1000

    for query_id in range(min_id, max_id + 1):
        base_score_file = os.path.join(base_score_dir,
                                       f"{query_id}_base_score.json")
        if not os.path.exists(base_score_file):
            logger.info("%s not exists", base_score_file)
            continue
        with open(base_score_file, "r", encoding="utf-8") as f:
            base_score_data = json.load(f)

        # 从 ddg_textbook 来源中过滤 ddg 来源
        textbook_score_data = [
            item for item in base_score_data
            if item["source_type"] == "textbook"
        ]

        if len(textbook_score_data) == 0:
            logger.info("textbook_score_data is empty")
            continue

        logger.info("processing %d", query_id)
        rel_score_data = get_rel_score_data(base_score_file)

        # 获得相似度矩阵
        chunks = [item["chunk"] for item in rel_score_data]
        sim_matrix = qa_calc_score.calc_chunks_sim_score(chunks)

        # 根据相似度矩阵获取 node_and_neighbors
        node_and_neighbors_file = os.path.join(
            node_and_neighbors_dir, f"{query_id}_node_and_neighbors.json")

        # 按 rel_score 降序排序并取 top_k
        sorted_indices = sorted(range(len(rel_score_data)),
                                key=lambda i: -rel_score_data[i]["rel_score"])
        top_k_indices = sorted_indices[:TOP_K]

        result = []
        for idx in top_k_indices:
            # 当前节点信息（直接包含全部字段）
            node = {
                "chunk": rel_score_data[idx]["chunk"],
                "source_type": rel_score_data[idx]["source_type"],
                "source": rel_score_data[idx]["source"],
                "sim_score": rel_score_data[idx]["sim_score"],
                "ner_score": rel_score_data[idx]["ner_score"],
                "rel_score": rel_score_data[idx]["rel_score"],
            }

            # 获取所有非自身节点的索引
            all_indices = np.arange(len(rel_score_data))
            mask = (all_indices != idx)
            candidate_indices = all_indices[mask]

            # 按相似度降序排序
            sorted_neighbors = sorted(candidate_indices,
                                      key=lambda i: -sim_matrix[idx][i])

            # 取前 top_k 个邻居
            neighbor_indices = sorted_neighbors[:TOP_K]

            # 邻居信息
            neighbors = [
                {
                    "chunk": rel_score_data[i]["chunk"],
                    "source_type": rel_score_data[i]["source_type"],
                    "source": rel_score_data[i]["source"],
                    "sim_score": rel_score_data[i]["sim_score"],
                    "ner_score": rel_score_data[i]["ner_score"],
                    "rel_score": rel_score_data[i]["rel_score"],
                    "inter_chunk_sim":
                        float(sim_matrix[idx][i])  # 添加相似度信息便于调试
                } for i in neighbor_indices
            ]

            result.append({"node": node, "neighbors": neighbors})

        with open(node_and_neighbors_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)


def get_top_k_with_neighbors():
    """
    对每个问题，获取 top k 个 rel_score 最大的节点及其 top_k 的邻居
    """
    base_score_dir = "./experiments/medmcqa/scores/ddg_textbook/base_score/"
    node_and_neighbors_dir = f"./experiments/medmcqa/scores/ddg_textbook/alpha_{int(ALPHA * 10):02d}_topk_{TOP_K}"
    os.makedirs(node_and_neighbors_dir, exist_ok=True)
    min_id = 1
    max_id = 1000

    for query_id in range(min_id, max_id + 1):
        base_score_file = os.path.join(base_score_dir,
                                       f"{query_id}_base_score.json")

        if not os.path.exists(base_score_file):
            logger.info("%s not exists", base_score_file)
            continue

        logger.info("processing %s", base_score_file)
        rel_score_data = get_rel_score_data(base_score_file)

        # 获得相似度矩阵
        chunks = [item["chunk"] for item in rel_score_data]
        sim_matrix = qa_calc_score.calc_chunks_sim_score(chunks)

        # 根据相似度矩阵获取 node_and_neighbors
        node_and_neighbors_file = os.path.join(
            node_and_neighbors_dir, f"{query_id}_node_and_neighbors.json")

        # 按 rel_score 降序排序并取 top_k
        sorted_indices = sorted(range(len(rel_score_data)),
                                key=lambda i: -rel_score_data[i]["rel_score"])
        top_k_indices = sorted_indices[:TOP_K]

        result = []
        for idx in top_k_indices:
            # 当前节点信息（直接包含全部字段）
            node = {
                "chunk": rel_score_data[idx]["chunk"],
                "source_type": rel_score_data[idx]["source_type"],
                "source": rel_score_data[idx]["source"],
                "sim_score": rel_score_data[idx]["sim_score"],
                "ner_score": rel_score_data[idx]["ner_score"],
                "rel_score": rel_score_data[idx]["rel_score"],
            }

            # 获取所有非自身节点的索引
            all_indices = np.arange(len(rel_score_data))
            mask = (all_indices != idx)
            candidate_indices = all_indices[mask]

            # 按相似度降序排序
            sorted_neighbors = sorted(candidate_indices,
                                      key=lambda i: -sim_matrix[idx][i])

            # 取前 top_k 个邻居
            neighbor_indices = sorted_neighbors[:TOP_K]

            # 邻居信息
            neighbors = [
                {
                    "chunk": rel_score_data[i]["chunk"],
                    "source_type": rel_score_data[i]["source_type"],
                    "source": rel_score_data[i]["source"],
                    "sim_score": rel_score_data[i]["sim_score"],
                    "ner_score": rel_score_data[i]["ner_score"],
                    "rel_score": rel_score_data[i]["rel_score"],
                    "inter_chunk_sim":
                        float(sim_matrix[idx][i])  # 添加相似度信息便于调试
                } for i in neighbor_indices
            ]

            result.append({"node": node, "neighbors": neighbors})

        with open(node_and_neighbors_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)


def summarize_query_and_chunks():
    """
    调用 llm 总结 query 相关 chunks 的信息
    """
    # node_and_neighbors_dir = f"./experiments/medmcqa/scores/ddg/alpha_{int(ALPHA * 10):02d}_topk_{TOP_K}"
    node_and_neighbors_dir = f"./experiments/medmcqa/scores/textbook/en_inverted_index/alpha_{int(ALPHA * 10):02d}_topk_{TOP_K}/"
    response_score_threshold = 0.5
    # 每次 summary 总结的 chunk 数量
    summary_batch_size = 5

    # 限制 query_id 在 [mid_id, max_id] 之间
    min_id = 1
    max_id = 1000
    for query_id in range(min_id, max_id + 1):
        logger.info("-" * 50)

        node_and_neighbors_file = os.path.join(
            node_and_neighbors_dir, f"{query_id}_node_and_neighbors.json")
        if not os.path.exists(node_and_neighbors_file):
            logger.info("%s not exists", node_and_neighbors_file)
            continue

        logger.info("processing query %d", query_id)
        # 获取 query 信息
        query, options = get_query_and_options(query_id)
        # 获取所有 chunks 及其 source
        chunks_and_source = get_chunks_and_source(node_and_neighbors_dir,
                                                  query_id)

        # 准备结果文件
        result = []
        summary_file = os.path.join(node_and_neighbors_dir,
                                    f"{query_id}_summary.json")

        chunks_len = len(chunks_and_source)
        for i in range(0, chunks_len, summary_batch_size):
            logger.info("processing index %d - %d", i, i + summary_batch_size)

            response = llm.summarize_chunks(
                query,
                options,
                chunks_and_source[i:i + summary_batch_size],
            )
            if float(response.get("score")) < response_score_threshold:
                response["summary"] = ""

            result.append({
                "query": query,
                "options": options,
                "context": chunks_and_source[i:i + summary_batch_size],
                "response": response,
                "summary": response.get("summary")
            })

        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)


def main():
    """
    主程序入口
    """
    # get_top_k_with_neighbors()
    # get_top_k_with_neighbors_faiss_index()
    # get_top_k_with_neighbors_inverted_index()
    summarize_query_and_chunks()


if __name__ == "__main__":
    main()

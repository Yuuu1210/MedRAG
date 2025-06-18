"""
使用 dpr 检索相关文档
"""
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer

# 该模型的上下文输入最长为 514 个 token
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("FremyCompany/BioLORD-2023")
tokenizer = AutoTokenizer.from_pretrained("FremyCompany/BioLORD-2023")
model.to(device)


def split_into_chunks(
    text: str,
    chunk_size: int = 256,
    overlap: int = 50,
) -> list[str]:
    """
    Splits a long text into overlapping chunks based on token count.

    Args:
        text (str): The input text to be split.
        chunk_size (int): The desired number of tokens in each chunk.
        overlap (int): The number of tokens to overlap between consecutive chunks.

    Returns:
        list[str]: A list of text chunks.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap.")

    # 将文本编码为 token IDs
    # add_special_tokens=False 因为我们是在处理文本段落，而不是完整的输入序列
    # 通常 CLS 和 SEP 会在构建 DPR 输入时再添加
    token_ids = tokenizer.encode(text,
                                 add_special_tokens=False,
                                 truncation=False)
    print(f"原始文本转换后的总 token 数: {len(token_ids)}")

    if not token_ids:
        return []

    chunks = []
    start_index = 0
    total_tokens = len(token_ids)

    while start_index < total_tokens:
        end_index = min(start_index + chunk_size, total_tokens)

        # 获取当前 chunk 的 token IDs
        current_chunk_token_ids = token_ids[start_index:end_index]

        # 将 token IDs 解码回文本
        # skip_special_tokens=True 避免解码出 tokenizer 可能在内部使用的特殊 token
        chunk_text = tokenizer.decode(current_chunk_token_ids,
                                      skip_special_tokens=True,
                                      clean_up_tokenization_spaces=True)

        # 确保解码后的文本不为空 (有时解码空的 token 列表可能返回空字符串)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())

        # 如果这是最后一个 chunk，或者没有足够的 token 形成下一个 chunk
        if end_index == total_tokens:
            break

        # 更新下一个 chunk 的起始位置
        # 减去 overlap 来创建重叠
        start_index += (chunk_size - overlap)

        # 防止 start_index 因为 overlap 过大而超过 end_index 导致死循环或无效切分
        # 这种情况理论上不应该发生，如果 chunk_size > overlap
        if start_index >= end_index and end_index < total_tokens:  # 额外检查，确保我们还有剩余内容
            # 如果发生了，可能是overlap设置导致步进太小，可以考虑直接跳到end_index
            print(
                f"Warning: Advancing start_index to end_index ({end_index}) to avoid potential stall."
            )
            start_index = end_index

    return chunks


def calc_query_text_similarity(query: str, text: str) -> float:
    """
    计算 query 与 text 的相似度

    similarity 在 [0,1] 范围内
    """
    if not text or text.isspace():
        return 0.0

    query_embedding = model.encode(query, convert_to_tensor=True)
    text_embedding = model.encode(text, convert_to_tensor=True)
    query_embedding = query_embedding.to(device)
    text_embedding = text_embedding.to(device)

    if query_embedding.ndim == 1:
        query_embedding = query_embedding.unsqueeze(0)
    if text_embedding.ndim == 1:
        text_embedding = text_embedding.unsqueeze(0)

    # cosine_scores 在 [-1, 1] 范围内
    cosine_scores = util.pytorch_cos_sim(query_embedding, text_embedding)

    # 将 cosine_scores 转换为 [0, 1] 范围
    normalized_score = (cosine_scores.item() + 1) / 2

    return normalized_score


def dpr_retrieve(query: str, corpus: list[dict], k: int = 32) -> list[dict]:
    """
    Retrieves the top-k most relevant chunks from the corpus for a given query
    using a dense retrieval model (SentenceTransformer).

    Args:
        query (str): The search query.
        corpus (list[dict]): A list of dictionaries. dict 包含 chunk, source, souce_type 三个键
        k (int): The number of top relevant chunks to return.

    Returns:
        list[dict]: A list of the top-k most relevant items from the original corpus,
                    sorted by relevance (highest score first). Each returned dictionary
                    will have an added 'score' key indicating the cosine similarity.
                    Returns an empty list if the corpus is empty or no chunks are found.
    """
    if not corpus:
        print("Warning: Corpus is empty.")
        return []

    # 1. Extract chunk texts from the corpus
    corpus_chunks = []
    original_indices = []  # To map back to original full dicts
    for i, item in enumerate(corpus):
        if 'chunk' in item and isinstance(item['chunk'],
                                          str) and item['chunk'].strip():
            corpus_chunks.append(item['chunk'])
            original_indices.append(i)
        else:
            print(
                f"Warning: Item at index {i} does not have a valid chunk or it's empty. Skipping."
            )

    if not corpus_chunks:
        print("Warning: No valid chunks found in the corpus after filtering.")
        return []

    # 2. Encode corpus chunks and the query
    # For very large corpora, consider pre-computing and indexing corpus_embeddings (e.g., with FAISS)
    # For this function, we encode on-the-fly.
    print(f"Encoding {len(corpus_chunks)} corpus chunks...")
    corpus_embeddings = model.encode(corpus_chunks,
                                     convert_to_tensor=True,
                                     show_progress_bar=True)

    print("Encoding query...")
    query_embedding = model.encode(query, convert_to_tensor=True)

    # 3. Perform semantic search using cosine similarity
    # util.semantic_search returns a list of lists for top_k results for each query.
    # Since we have one query, we take the first element hits[0].
    # Each result in the inner list is a dict: {'corpus_id': int, 'score': float}
    # The 'corpus_id' here refers to the index in `corpus_chunks`.
    print(f"Performing semantic search for top {k} results...")
    # Ensure k is not larger than the number of available chunks
    actual_k = min(k, len(corpus_chunks))
    hits = util.semantic_search(query_embedding,
                                corpus_embeddings,
                                top_k=actual_k)

    # We have one query, so we are interested in hits[0]
    top_hits = hits[0]

    # 4. Prepare results by mapping back to original corpus items and adding scores
    results = []
    for hit in top_hits:
        corpus_chunk_index = hit['corpus_id']
        original_corpus_index = original_indices[corpus_chunk_index]

        # Create a copy to avoid modifying the original corpus item
        result_item = corpus[original_corpus_index].copy()
        result_item['score'] = hit['score']  # Add the relevance score
        results.append(result_item)

    return results


def main():
    """
    主函数
    """
    pass


if __name__ == "__main__":
    main()

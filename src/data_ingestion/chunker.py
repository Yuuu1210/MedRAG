"""
将文本划分为 chunk
"""
from transformers import AutoTokenizer

# 使用 NER 模型来分块
NER_MODEL = "d4data/biomedical-ner-all"

# 在函数外部加载tokenizer，避免每次调用函数都重新加载，提高效率
tokenizer = AutoTokenizer.from_pretrained(NER_MODEL)


def split_into_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    """
    将文本按照指定的 token 数量分割成带有重叠的块。

    Args:
        text (str): 需要分割的原始文本。
        chunk_size (int): 每个块中期望的token数量
        overlap (int): 相邻块之间重叠的token数量。

    Returns:
        list[str]: 分割后的文本块列表。
    """
    if not text or not text.strip():
        return []

    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk_size.")

    # 1. 将整个文本编码为 token ID
    # 我们不在这里添加特殊token ([CLS], [SEP])，因为我们只关心内容token的数量。
    # 这些特殊token应在将chunk送入模型前再添加。
    all_token_ids = tokenizer.encode(text, add_special_tokens=False)

    if not all_token_ids:  # 如果文本只包含tokenizer会过滤掉的字符
        return []

    text_chunks = []
    stride = chunk_size - overlap  # 每次窗口滑动的步长

    # 如果总token数小于chunk_size，则整体作为一个chunk
    if len(all_token_ids) <= chunk_size:
        # 解码时，tokenizer会自动处理token拼接成可读文本的问题
        # skip_special_tokens=True 确保即使内部有特殊用途的token ID也被正确解码
        decoded_chunk = tokenizer.decode(all_token_ids,
                                         skip_special_tokens=True)
        if decoded_chunk.strip():  # 确保解码后的块不是空的
            text_chunks.append(decoded_chunk)
        return text_chunks

    # 2. 滑动窗口进行切分
    for i in range(0, len(all_token_ids), stride):
        # 获取当前块的token ID
        # 注意：最后一个块可能不足chunk_size
        chunk_token_ids = all_token_ids[i:i + chunk_size]

        if not chunk_token_ids:  # 如果切片结果为空，则停止
            break

        # 3. 将token ID块解码回文本
        # skip_special_tokens=True 对于解码纯内容块是合适的
        # clean_up_tokenization_spaces=True/False 可以根据需要调整，通常True更好
        decoded_chunk = tokenizer.decode(
            chunk_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True  # 默认是True，通常效果好
        )

        # 避免添加空的或只包含空格的块
        if decoded_chunk and decoded_chunk.strip():
            text_chunks.append(decoded_chunk)

        # 如果当前窗口的末尾已经到达或超过了token列表的末尾，
        # 并且这个块已经包含了所有剩余的tokens，则可以提前结束循环
        if i + chunk_size >= len(all_token_ids):
            break

    return text_chunks


def main():
    """
    主函数
    """
    pass


if __name__ == "__main__":
    main()

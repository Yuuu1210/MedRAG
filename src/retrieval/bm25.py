"""
使用 BM25 稀疏检索获取相关文档
"""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

# Download necessary NLTK data (run once)
# nltk.download('punkt', download_dir="./data/nltk/")
# nltk.download('stopwords', download_dir="./data/nltk/")
nltk.data.path.insert(0, "./data/nltk/")


def preprocess_text(text: str) -> list[str]:
    """
    Cleans, tokenizes, removes stop words, and stems text.
    """
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation (basic version)
    text = re.sub(r'[^\w\s]', '', text)  # Removes punctuation
    # 3. Tokenize
    tokens = word_tokenize(text)
    # 4. Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # 5. Stemming (optional, but often helpful for BM25)
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    return stemmed_tokens  # Or return filtered_tokens if not stemming


def bm25_retrieve(query: str, corpus: list[dict], k: int = 32):
    """
    在 corpus 中使用 BM25 检索与 query 相关的文档

    Args:
        query (str): The search query.
        corpus (lsit[dict]): dict 包含 chunk, source, souce_type 三个键
        k (int): The number of top results to return.

    Returns:
        list[dict]: A list of the top-k relevant items from the corpus, sorted by relevance.
                Each dictionary will have an additional 'score' key.
                Returns an empty list if the corpus is empty or the query is empty after basic processing.
    """
    if not corpus:
        print("Warning: Corpus is empty.")
        return []
    if not query or not query.strip():
        print("Warning: Query is empty.")
        return []

    print("start bm25 retrieval...")
    # 1. Preprocess and tokenize corpus chunks
    # Store original items to map back later if some chunks become empty after preprocessing
    processed_corpus_data = []
    for i, item in enumerate(corpus):
        tokens = preprocess_text(item['chunk'])
        if tokens:  # Only include if there are tokens after preprocessing
            processed_corpus_data.append({
                'original_index': i,
                'tokens': tokens,
                'original_item': item
            })

    if not processed_corpus_data:
        print("Warning: All corpus chunks are empty after preprocessing.")
        return []

    tokenized_corpus_for_bm25 = [
        data['tokens'] for data in processed_corpus_data
    ]

    try:
        bm25 = BM25Okapi(tokenized_corpus_for_bm25)
    except ValueError as e:  # Handles cases like empty tokenized_corpus_for_bm25
        print(f"Error initializing BM25: {e}.")
        return []

    # 2. Preprocess and tokenize query
    tokenized_query = preprocess_text(query)
    if not tokenized_query:
        print("Warning: Query became empty after preprocessing.")
        return []

    # Optional: Check if any query terms exist in the BM25 model's IDF scores
    query_terms_in_model_vocab = [
        term for term in tokenized_query if term in bm25.idf
    ]
    if not query_terms_in_model_vocab:
        print(
            f"Warning: None of the processed query terms {tokenized_query} were found in the corpus vocabulary."
        )
        return []

    # 3. Get scores (scores will correspond to tokenized_corpus_for_bm25)
    doc_scores = bm25.get_scores(tokenized_query)

    # 4. Combine scores with original corpus items
    results_with_scores = []
    for i, score in enumerate(doc_scores):
        # Get the original item corresponding to this score
        original_item_data = processed_corpus_data[i]
        result_item = original_item_data['original_item'].copy()
        result_item['score'] = score
        results_with_scores.append(result_item)

    # 5. Sort and return top-k
    # Filter out very low or negative scores if desired
    relevant_results = [
        res for res in results_with_scores if res['score'] > 0.0001
    ]
    sorted_results = sorted(relevant_results,
                            key=lambda x: x['score'],
                            reverse=True)

    print("bm25 retrieval completed.")
    return sorted_results[:k]


def main():
    """
    主函数
    """
    pass


if __name__ == "__main__":
    main()

"""
计算文本之间的相似度
"""
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

SIMILARITY_MODEL = "FremyCompany/BioLORD-2023"

model = SentenceTransformer(SIMILARITY_MODEL)


def calc_similarity(text1: str, text2: str) -> float:
    """
    Calculates the semantic similarity between two texts using the BioLORD-2023 model.

    Args:
        text1 (str): The first text.
        text2 (str): The second text.

    Returns:
        float: The cosine similarity score between the two texts (ranges from -1 to 1,
               but typically 0 to 1 for sentence embeddings). Returns 0.0 if model is not available.
    """
    try:
        # Encode the two texts into embeddings
        # The model.encode() method can take a list of sentences
        embeddings = model.encode([text1, text2])

        # embeddings will be a NumPy array of shape (2, N_dimensions)
        # e.g., embeddings[0] is the embedding for text1
        #       embeddings[1] is the embedding for text2

        # Calculate cosine similarity
        # cosine_similarity expects 2D arrays.
        # We want the similarity between the first embedding and the second.
        similarity_score = cosine_similarity(
            embeddings[0].reshape(1, -1),  # Reshape to (1, N_dimensions)
            embeddings[1].reshape(1, -1),
        )[0][0]  # Get the single similarity value from the 1x1 matrix

        return float(similarity_score)

    except Exception as e:
        print(f"Error during similarity calculation: {e}")
        return 0.0


def main():
    """
    主函数
    """
    pass


if __name__ == "__main__":
    main()

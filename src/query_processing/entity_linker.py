"""
使用 FAISS 库通过 cosine similarity 将 entity 链接到 mesh term
"""
import json
import os
import re
import time

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from src.utils import logger_config

logger = logger_config.get_logger(__name__)


def compute_mesh_embeddings():
    """
    (preprocess) 使用 SapBERT 计算 MeSH 术语的嵌入

    INPUT_CSV_PATH: 输入 CSV 文件路径。
        csv 文件有 3 列, 分别是 mesh_ui, term_string, is_preferred
        is_preferred 为 True, 表示首选术语, 为 False, 表示入口术语
    
    mesh_term_embeddings.npy 是一个二维数组, 行数为 Mesh 术语的数量, 列数为 768 (embedding 的维数)
    mesh_term_uis.npy 是一个一维数组, 包含 Mesh 术语的 UI (唯一标识符), 大小为 Mesh 术语的数量
    """
    # --- Configuration ---
    MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token"
    INPUT_CSV_PATH = './data/mesh/combined_terms.csv'
    OUTPUT_DIR = './data/mesh/embeddings/'
    EMBEDDINGS_FILENAME = 'mesh_term_embeddings.npy'
    MESH_UIS_FILENAME = 'mesh_term_uis.npy'
    # Batch size for inference - adjust based on your GPU memory
    BATCH_SIZE = 128  # Try 64 or 32 if you run out of memory
    # Max sequence length for tokenizer - from the Hugging Face example
    # WARNING: max_length=25 might truncate longer MeSH terms.
    # Consider increasing this (e.g., 64, 128) or removing it if truncation is an issue,
    # but be mindful of increased computation time/memory.
    MAX_LENGTH = 25

    # --- GPU Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # --- Create Output Directory ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    embeddings_output_path = os.path.join(OUTPUT_DIR, EMBEDDINGS_FILENAME)
    mesh_uis_output_path = os.path.join(OUTPUT_DIR, MESH_UIS_FILENAME)

    # --- Load Model and Tokenizer ---
    print(f"Loading tokenizer: {MODEL_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"Loading model: {MODEL_NAME} (this may take a while...)")
        # Load the model and move it to the designated device (GPU/CPU)
        model = AutoModel.from_pretrained(MODEL_NAME).to(device)
        model.eval()  # Set the model to evaluation mode (disables dropout etc.)
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        exit()

    # --- Load MeSH Terms Data ---
    print(f"Loading MeSH terms from: {INPUT_CSV_PATH}")
    try:
        df_combined = pd.read_csv(INPUT_CSV_PATH)
        # Ensure required columns exist
        if 'term_string' not in df_combined.columns or 'mesh_ui' not in df_combined.columns:
            print(
                f"Error: CSV file must contain 'term_string' and 'mesh_ui' columns."
            )
            exit()
        # Handle potential NaN values in term_string (replace with empty string or drop)
        df_combined['term_string'].fillna('', inplace=True)
        all_names = df_combined['term_string'].tolist()
        all_mesh_uis = df_combined['mesh_ui'].tolist()
        print(f"Loaded {len(all_names)} MeSH terms.")
        # Sanity check: ensure lists are of the same length
        if len(all_names) != len(all_mesh_uis):
            print(
                f"Error: Mismatch between number of terms ({len(all_names)}) and MeSH UIs ({len(all_mesh_uis)})."
            )
            exit()
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at '{INPUT_CSV_PATH}'")
        exit()

    # --- Compute Embeddings ---
    # 计算 embedding 的详细过程：
    # 有一个 str, 我们会拆分成 token
    # 每一个 token, 会被对应到模型的词汇表中的一个 id
    #   现在, 一个 str 对应一个 list[int], 这个 list 的长度是 max_length
    #   若不足 max_length, 则填充, 若超过 max_length, 则截断
    #   由 padding="max_length" 和 truncation=True 参数控制
    #   batch_encode_plus 会为每个 str 获取一个 input_ids 还有一个 attention_mask
    #   attention_mask 是一个 list[int], 1 表示这个 token 是有效的, 0 表示这个 token 是 padding
    #   attention_mask 告诉模型哪些 token 需要关注
    # 根据 token id, 经过模型的第一层（词嵌入层）会得到初始 embedding
    # 经过 attention 机制, 会得到包括上下文含义的 embedding
    # 通过 output[0], 我们获取 token 在最后一层的隐藏状态, 即 token 的最终 embedding
    # output[0].mean(1) 表示对所有 token 的 embedding 求平均, 得到 str 的 embedding
    print(f"Starting embedding computation with batch size {BATCH_SIZE}...")
    start_time = time.time()
    all_embs_list = []

    # Use torch.no_grad() for inference to reduce memory usage and speed up
    with torch.no_grad():
        for i in tqdm(range(0, len(all_names), BATCH_SIZE),
                      desc="Generating Embeddings"):
            # Get the batch of names
            batch_names = all_names[i:i + BATCH_SIZE]

            # Tokenize the batch
            toks = tokenizer.batch_encode_plus(
                batch_names,
                padding="max_length",  # Pad to max_length
                max_length=MAX_LENGTH,
                truncation=True,  # Truncate longer sequences
                return_tensors="pt")  # Return PyTorch tensors

            # Move tokens to the GPU (if available)
            toks_device = {k: v.to(device) for k, v in toks.items()}

            # Get model outputs
            # model(**toks_device)[0] gets the last hidden state
            # .mean(1) performs mean pooling over the sequence length dimension
            outputs = model(**toks_device)
            embeddings = outputs[0].mean(
                1)  # SapBERT-mean-token uses mean pooling

            # Move embeddings back to CPU and convert to NumPy array
            batch_embs = embeddings.cpu().detach().numpy()
            all_embs_list.append(batch_embs)

    # Concatenate all batch embeddings into a single NumPy array
    all_embs = np.concatenate(all_embs_list, axis=0)

    end_time = time.time()
    print(
        f"Embedding computation finished in {end_time - start_time:.2f} seconds."
    )
    print(f"Shape of generated embeddings: {all_embs.shape}"
         )  # Should be (num_terms, 768) for base models

    # --- Save Embeddings and MeSH UIs ---
    print(f"Saving embeddings to: {embeddings_output_path}")
    np.save(embeddings_output_path, all_embs)

    print(f"Saving corresponding MeSH UIs to: {mesh_uis_output_path}")
    # Convert list of UIs to NumPy array before saving for consistency
    np.save(mesh_uis_output_path, np.array(all_mesh_uis))

    print("\n--- Process Complete ---")
    print(f"Embeddings saved with shape: {all_embs.shape}")
    print(f"MeSH UIs saved with length: {len(all_mesh_uis)}")
    print(
        f"Next step: Use '{embeddings_output_path}' and '{mesh_uis_output_path}' to build a FAISS index."
    )


def build_faiss_index():
    """
    (preprocess) 使用 FAISS 库为 MeSH 术语嵌入向量构建一个索引
    """
    # --- Configuration ---
    EMBEDDINGS_PATH = "./data/mesh/embeddings/mesh_term_embeddings.npy"
    MESH_UIS_PATH = "./data/mesh/embeddings/mesh_term_uis.npy"
    INDEX_OUTPUT_DIR = "./data/mesh/faiss_index/"
    INDEX_FILENAME = "mesh_terms.index"

    # --- Create Output Directory ---
    os.makedirs(INDEX_OUTPUT_DIR, exist_ok=True)
    index_output_path = os.path.join(INDEX_OUTPUT_DIR, INDEX_FILENAME)

    # --- Load Embeddings and UIs ---
    print(f"Loading embeddings from: {EMBEDDINGS_PATH}")
    try:
        embeddings = np.load(EMBEDDINGS_PATH)
        print(
            f"Embeddings loaded. Shape: {embeddings.shape}, Dtype: {embeddings.dtype}"
        )

        # Ensure embeddings are float32, as FAISS typically expects this
        if embeddings.dtype != np.float32:
            print("Converting embeddings to float32...")
            embeddings = embeddings.astype(np.float32)

    except FileNotFoundError:
        print(f"Error: Embeddings file not found at '{EMBEDDINGS_PATH}'")
        exit()

    # Optional: Load UIs just to verify length consistency
    try:
        mesh_uis = np.load(MESH_UIS_PATH)
        print(f"MeSH UIs loaded. Length: {len(mesh_uis)}")
        if len(mesh_uis) != embeddings.shape[0]:
            print(
                f"Error: Number of MeSH UIs ({len(mesh_uis)}) does not match number of embeddings ({embeddings.shape[0]})!"
            )
            exit()
    except FileNotFoundError:
        print(
            f"Warning: MeSH UIs file not found at '{MESH_UIS_PATH}'. Cannot verify length consistency."
        )

    # --- Normalize Embeddings ---
    # L2 normalization is crucial for using IndexFlatIP for cosine similarity search.
    # Each vector's norm (length) will become 1.
    print("Normalizing embeddings (L2 norm)...")
    # Create a copy to ensure the original array isn't modified if normalize_L2 works in-place
    embeddings_normalized = embeddings.copy()
    faiss.normalize_L2(embeddings_normalized)
    print("Embeddings normalized.")

    # --- Build FAISS Index ---
    print("Building FAISS index (IndexFlatIP)...")
    start_time = time.time()

    # Get the dimensionality of the vectors
    d = embeddings_normalized.shape[1]

    # Create the index using Inner Product (IP) metric
    index = faiss.IndexFlatIP(d)

    # Add the normalized vectors to the index
    index.add(embeddings_normalized)

    end_time = time.time()
    print(f"Index built in {end_time - start_time:.2f} seconds.")
    print(f"Index type: IndexFlatIP")
    print(f"Index contains {index.ntotal} vectors.")
    print(f"Index dimensionality: {index.d}")
    print(f"Is index trained? {index.is_trained}"
         )  # IndexFlat does not require training

    # Verify the number of vectors added
    if index.ntotal != embeddings.shape[0]:
        print(
            f"Error: Number of vectors in index ({index.ntotal}) does not match input ({embeddings.shape[0]})!"
        )
    else:
        print("Vector count in index matches input.")

    # --- Save the Index ---
    print(f"Saving index to: {index_output_path}")
    try:
        faiss.write_index(index, index_output_path)
        print("FAISS index saved successfully.")
    except Exception as e:
        print(f"Error saving FAISS index: {e}")

    print("\n--- Process Complete ---")
    print(f"FAISS index file created at: {index_output_path}")
    print("This index can now be loaded for similarity search.")


def is_potential_abbreviation(word: str) -> bool:
    """
    判断一个 word 是不是缩写
    """
    # (Use the function defined in the previous answer, acting on the 'word')
    if len(word) < 2:
        return False
    if not re.fullmatch(r'[a-zA-Z0-9]+(?:[-.][a-zA-Z0-9]+)*', word):
        return False
    if not any(c.isupper() for c in word):
        return False

    if sum(1 for c in word if c.isupper()) >= 2:
        return True  # Pattern A
    has_upper = any(c.isupper() for c in word)
    has_digit = any(c.isdigit() for c in word)
    if has_upper and has_digit:
        return True  # Pattern B
    if word.isupper():  # Pattern C
        roman_numerals = {
            "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII",
            "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"
        }
        if word not in roman_numerals:
            return True
    return False


def initialize_linker_resources():
    """
    Loads all necessary resources for entity linking.
    """
    # 计算 embedding 所需参数
    model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token"
    index_path = './data/mesh/faiss_index/mesh_terms.index'
    mesh_uis_path = './data/mesh/embeddings/mesh_term_uis.npy'
    mesh_terms_csv_path = './data/mesh/combined_terms.csv'

    print("Initializing linker resources...")
    resources = {}

    # Setup Device
    if torch.cuda.is_available():
        resources['device'] = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        resources['device'] = torch.device("cpu")
        print("Using CPU")

    # Load Tokenizer and Model
    print(f"Loading tokenizer: {model_name}")
    try:
        resources['tokenizer'] = AutoTokenizer.from_pretrained(model_name)
        print(f"Loading model: {model_name}")
        resources['model'] = AutoModel.from_pretrained(model_name).to(
            resources['device'])
        resources['model'].eval()
        print("Model and tokenizer loaded.")
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        raise

    # Load FAISS Index
    print(f"Loading FAISS index from: {index_path}")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index file not found: {index_path}")
    try:
        resources['index'] = faiss.read_index(index_path)
        print(
            f"FAISS index loaded. Contains {resources['index'].ntotal} vectors."
        )
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        raise

    # Load MeSH UIs
    print(f"Loading MeSH UIs from: {mesh_uis_path}")
    if not os.path.exists(mesh_uis_path):
        raise FileNotFoundError(f"MeSH UIs file not found: {mesh_uis_path}")
    try:
        resources['mesh_uis'] = np.load(mesh_uis_path, allow_pickle=True)
        print(f"MeSH UIs loaded. Length: {len(resources['mesh_uis'])}")
    except Exception as e:
        print(f"Error loading MeSH UIs: {e}")
        raise

    # Load MeSH Term Strings
    print(f"Loading MeSH term strings from: {mesh_terms_csv_path}")
    if not os.path.exists(mesh_terms_csv_path):
        raise FileNotFoundError(
            f"MeSH terms CSV file not found: {mesh_terms_csv_path}")
    try:
        df_terms = pd.read_csv(mesh_terms_csv_path)
        if 'term_string' not in df_terms.columns:
            raise ValueError("CSV file must contain a 'term_string' column.")
        df_terms['term_string'].fillna(
            '', inplace=True)  # Match embedding creation NaN handling
        resources['mesh_terms'] = df_terms['term_string'].tolist()
        print(
            f"MeSH term strings loaded. Length: {len(resources['mesh_terms'])}")
    except Exception as e:
        print(f"Error loading MeSH term strings from CSV: {e}")
        raise

    # --- Validation ---
    if not (resources['index'].ntotal == len(resources['mesh_uis']) == len(
            resources['mesh_terms'])):
        print(
            "Error: Mismatch between index size, MeSH UI count, and MeSH term count!"
        )
        # ... (add print statements for counts as before if desired)
        raise ValueError("Index and mapping data sizes do not match.")

    print("Linker resources initialized successfully.")
    return resources


def get_embeddings(
    texts: list,
    model,
    tokenizer,
    device,
    batch_size: int,
) -> np.ndarray:
    """
    Computes embeddings for a list of texts using the provided model and tokenizer.
    """
    max_length = 25  # Should match embedding creation

    all_embs_list = []
    model.eval()  # Ensure model is in eval mode
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size),
                      desc="Generating query embeddings",
                      leave=False):
            batch_texts = texts[i:i + batch_size]
            toks = tokenizer.batch_encode_plus(batch_texts,
                                               padding="max_length",
                                               max_length=max_length,
                                               truncation=True,
                                               return_tensors="pt")
            toks_device = {k: v.to(device) for k, v in toks.items()}
            outputs = model(**toks_device)
            embeddings = outputs[0].mean(1)  # Mean pooling
            all_embs_list.append(embeddings.cpu().numpy())
    return np.concatenate(all_embs_list, axis=0)


def link_entities(
    entity_texts: list[str],
    resources: dict,
    top_k: int = 5,
) -> list[list[dict]]:
    """
    Links a list of entity text strings to MeSH terms using pre-loaded resources.

    Args:
        entity_texts: A list of entity strings to link.
        resources: A dictionary containing 'model', 'tokenizer', 'index',
                   'mesh_uis', 'mesh_terms', and 'device'.
        top_k: The number of top similar MeSH terms to return.
        batch_size: Batch size for computing entity embeddings.

    Returns:
        A list of lists, structured as in the class-based version.
    """
    batch_size = 64  # Default batch size for inference
    if not entity_texts:
        return []

    # Extract resources needed
    model = resources['model']
    tokenizer = resources['tokenizer']
    index = resources['index']
    mesh_uis = resources['mesh_uis']
    mesh_terms = resources['mesh_terms']
    device = resources['device']

    print(f"Linking {len(entity_texts)} entities (top_k={top_k})...")
    start_time = time.time()

    # 1. Compute embeddings for the input entities
    query_embeddings = get_embeddings(entity_texts, model, tokenizer, device,
                                      batch_size)

    # 2. Normalize the query embeddings
    query_embeddings_normalized = query_embeddings.copy()
    faiss.normalize_L2(query_embeddings_normalized)
    print("Query embeddings computed and normalized.")

    # 3. Search the FAISS index
    print(f"Searching FAISS index for {top_k} nearest neighbors...")
    D, I = index.search(query_embeddings_normalized, top_k)
    print("FAISS search complete.")

    # 4. Format the results
    all_results = []
    for i, entity_text in enumerate(entity_texts):
        entity_results = []
        indices = I[i]
        scores = D[i]
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            if idx == -1:
                continue
            mesh_ui = mesh_uis[idx]
            mesh_term = mesh_terms[idx]
            entity_results.append({
                'rank': rank + 1,
                'mesh_ui': mesh_ui,
                'mesh_term': mesh_term,
                'score': float(score)
            })
        all_results.append(entity_results)

    end_time = time.time()
    print(f"Linking finished in {end_time - start_time:.2f} seconds.")
    return all_results


def medmcqa_normalize_query_entity():
    """
    对 query 中提取的实体进行标准化
    """
    entity_file = "./experiments/medmcqa/jsons/doubao_query_entities.json"
    normalized_entity_file = "./experiments/medmcqa/jsons/doubao_query_entities_normalized.json"
    min_id = 1
    max_id = 1000

    with open(entity_file, "r", encoding="utf-8") as f:
        entity_data = json.load(f)

    if os.path.exists(normalized_entity_file):
        with open(normalized_entity_file, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = []

    # 提前准备好使用 link_entities 函数所需资源
    loaded_resources = initialize_linker_resources()

    for item in entity_data:
        query_id = item["id"]
        if query_id < min_id:
            query_id += 1
            continue
        if query_id > max_id:
            break

        logger.info("-" * 50)
        logger.info("processing query %d", query_id)

        query = item["query"]
        # 提取实体
        all_entities = set()
        all_entities.update(item["query_entities"])
        all_entities.update(item["opa_entities"])
        all_entities.update(item["opb_entities"])
        all_entities.update(item["opc_entities"])
        all_entities.update(item["opd_entities"])
        all_entities = list(all_entities)

        # 使用 faiss 链接到 mesh term
        link_results = link_entities(all_entities, loaded_resources, top_k=1)

        # 一个 dict, 键为原始实体，值为对应的 mesh term
        all_entity_to_mesh = {}
        for entity, link_result in zip(all_entities, link_results):
            # link_result 包含 mesh_ui, mesh_term, score
            if link_result[0]["score"] < 0.9:
                continue
            all_entity_to_mesh[entity] = link_result[0]["mesh_term"]

        # 将实体和对应的 mesh term 分配给每个选项
        query_entity_to_mesh = {}
        for entity in item["query_entities"]:
            if entity not in all_entity_to_mesh:
                continue
            query_entity_to_mesh[entity] = all_entity_to_mesh[entity]
        opa_entity_to_mesh = {}
        for entity in item["opa_entities"]:
            if entity not in all_entity_to_mesh:
                continue
            opa_entity_to_mesh[entity] = all_entity_to_mesh[entity]
        opb_entity_to_mesh = {}
        for entity in item["opb_entities"]:
            if entity not in all_entity_to_mesh:
                continue
            opb_entity_to_mesh[entity] = all_entity_to_mesh[entity]
        opc_entity_to_mesh = {}
        for entity in item["opc_entities"]:
            if entity not in all_entity_to_mesh:
                continue
            opc_entity_to_mesh[entity] = all_entity_to_mesh[entity]
        opd_entity_to_mesh = {}
        for entity in item["opd_entities"]:
            if entity not in all_entity_to_mesh:
                continue
            opd_entity_to_mesh[entity] = all_entity_to_mesh[entity]

        results.append({
            "id": query_id,
            "query": query,
            "options": item["options"],
            "cop": item["cop"],
            "query_entity_to_mesh": query_entity_to_mesh,
            "opa_entity_to_mesh": opa_entity_to_mesh,
            "opb_entity_to_mesh": opb_entity_to_mesh,
            "opc_entity_to_mesh": opc_entity_to_mesh,
            "opd_entity_to_mesh": opd_entity_to_mesh,
        })

        with open(normalized_entity_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)


def main():
    """
    主程序入口
    """
    # compute_mesh_embeddings()
    medmcqa_normalize_query_entity()


if __name__ == "__main__":
    main()

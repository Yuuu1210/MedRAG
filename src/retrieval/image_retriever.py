"""
获取 query 相关的 image
"""
import json
import os

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.retrieval import dpr, text_retriever

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model.to(device)


def get_images_from_docs(query_id: int) -> list[dict]:
    """
    (Helper) 在文档中获取与 query_id 相关的图像

    Returns:
        list[dict]: dict 键包含 title, path, source_type
    """
    docs_dir = f"./experiments/medmcqa/docs/{query_id}/"
    content_lists = text_retriever.find_content_list(docs_dir)

    results = []
    for content_list in content_lists:
        par_dir = os.path.dirname(content_list)
        if text_retriever.is_pubmed_content_list(content_list):
            source_type = "pubmed"
        else:
            source_type = "ddg"

        with open(content_list, "r", encoding="utf-8") as file:
            content = json.load(file)

        for item in content:
            if item["type"] == "image" and item["img_path"] != "":
                results.append({
                    "title":
                        item["img_caption"][0] if item["img_caption"] else "",
                    "path":
                        os.path.join(par_dir, item["img_path"]),
                    "source_type":
                        source_type
                })
            elif item["type"] == "table" and item["img_path"] != "":
                results.append({
                    "title":
                        item["table_caption"][0] \
                        if item["table_caption"] else "",
                    "path":
                        os.path.join(par_dir, item["img_path"]),
                    "source_type":
                        source_type
                })

    return results


def calc_clip_score(query: str, image: dict) -> float:
    """
    (Helper) 使用 clip 模型计算 query 与 image 的相似度

    Args:
        query (str): 查询文本
        image (dict): dict 包含 title, path, source_type
    """
    image = Image.open(image["path"]).convert("RGB")

    # 注意：即使是单个查询和单个图像，processor也期望文本是列表形式
    inputs = clip_processor(
        text=[query],  # 文本查询，作为列表元素
        images=image,  # 单个PIL Image对象
        return_tensors="pt",  # 返回PyTorch张量
        padding=True,  # 对文本进行填充
        truncation=True  # 如果文本过长则截断
    )

    # 将输入数据移到与模型相同的设备
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 4. 模型推理
    with torch.no_grad():  # 在推理时不需要计算梯度
        outputs = clip_model(**inputs)

    # 5. 获取相似度得分
    # outputs.logits_per_image 的形状是 (batch_size_images, num_texts)
    # 在这里，batch_size_images = 1, num_texts = 1
    # 所以 logits_per_image 的形状是 (1, 1)
    # 我们需要的是这个 (1,1) 张量中的标量值
    # .item() 将单元素张量转换为Python标量
    raw_logit = outputs.logits_per_image[0, 0]

    # 归一化到 [0, 1] 范围
    logit_scale = clip_model.logit_scale.exp()
    cosine_similarity = raw_logit / logit_scale
    # 将cosine similarity从[-1, 1]映射到[0, 1]
    # clamp to avoid potential floating point issues making it slightly out of [-1,1]
    normalized_score = (torch.clamp(cosine_similarity, -1.0, 1.0) + 1.0) / 2.0

    return normalized_score.item()


def calc_query_image_similarity(query: str,
                                image: dict,
                                alpha: float = 0.3) -> float:
    """
    计算 query 与 image 的相似度

    Args:
        query (str): 查询文本
        image (dict): dict 包含 title, path, source_type
    """
    # 根据 image["title"] 计算和 query 的相似度
    # title_similarity 在 [0,1] 范围内
    title_similarity = dpr.calc_query_text_similarity(query, image["title"])

    # 根据 image 计算和 query 的相似度
    clip_score = calc_clip_score(query, image)

    # 加权组合
    return alpha * title_similarity + (1 - alpha) * clip_score


def filter_top_k_images(query: str,
                        options: str,
                        images: list[dict],
                        k: int = 10) -> list[dict]:
    """
    过滤出与查询最相关的前 k 张图像
    """
    for image in images:
        # 对每个 option, 组合 query 和 option, 计算 similarity 后取平均
        similarities = []
        for _, option_text in options.items():
            combined_query = query + " " + option_text
            similarities.append(
                calc_query_image_similarity(combined_query, image))

        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            image["similarity"] = avg_similarity

    # 按照相似度排序
    images = sorted(images, key=lambda x: x["similarity"], reverse=True)

    # 取前 k 张图像
    return images[:k]


def medmcqa_retrieve_images():
    """
    从 MedMCQA 数据集中检索与 query 相关的图像
    """
    query_file = "./experiments/medmcqa/jsons/doubao_query.json"
    min_id = 12
    max_id = 12

    with open(query_file, "r", encoding="utf-8") as f:
        query_data = json.load(f)

    for item in query_data:
        query_id = item["id"]
        query = item["query"]
        options = item["options"]

        if query_id < min_id:
            continue
        if query_id > max_id:
            break

        print("processing query id:", query_id)
        images = get_images_from_docs(query_id)
        if not images:
            print(f"No images found for query id {query_id}")
            continue

        filtered_images = filter_top_k_images(query, options, images, k=10)

        # 将结果保存到文件
        image_file = f"./experiments/medmcqa/retrieval/{query_id}_image.json"
        with open(image_file, "w", encoding="utf-8") as f:
            json.dump(filtered_images, f, ensure_ascii=False, indent=4)


def main():
    """
    主函数
    """
    medmcqa_retrieve_images()


if __name__ == "__main__":
    main()

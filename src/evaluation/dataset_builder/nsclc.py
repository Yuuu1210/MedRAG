"""
在 ./data/nsclc/ 目录下有 100 篇论文
论文经过 MinerU 解析获得了 content_list.json
Step 1: 从 content_list.json 中获得论文的不同章节
Step 2: 调用 gpt-4o 针对不同的章节生成不同类型的问题
"""
import json
import os

from src.llm_clients import gpt_client
from src.utils import logger_config

logger = logger_config.get_logger(__name__)
llm = gpt_client.GPTClient()


def find_pmid_jsons(root_dir: str) -> list[str]:
    """
    找到 root_dir 中所有的 pmid.json 文件

    Args:
        root_dir: 根目录

    Returns:
        pmid_jsons: 所有的 pmid.json 文件
    """
    pmid_jsons = []

    for pmid_folder_name in os.listdir(root_dir):
        pmid_folder_path = os.path.join(root_dir, pmid_folder_name)
        json_file_name = pmid_folder_name + ".json"
        json_file_path = os.path.join(pmid_folder_path, json_file_name)

        # 检查 json 文件是否存在
        if os.path.exists(json_file_path):
            pmid_jsons.append(json_file_path)

    return pmid_jsons


def find_sections_jsons(root_dir: str) -> list[str]:
    """
    找到 root_dir 中所有的 sections.json 文件

    Args:
        root_dir: 根目录

    Returns:
        sections_jsons: 所有的 sections.json 文件
    """
    sections_jsons = []

    for sections_folder_name in os.listdir(root_dir):
        sections_folder_path = os.path.join(root_dir, sections_folder_name)
        json_file_name = "sections.json"
        json_file_path = os.path.join(sections_folder_path, json_file_name)

        # 检查 json 文件是否存在
        if os.path.exists(json_file_path):
            sections_jsons.append(json_file_path)

    return sections_jsons


def find_text_questions_jsons(root_dir: str) -> list[str]:
    """
    找到 root_dir 中所有的 text_questions.json 文件

    Args:
        root_dir: 根目录

    Returns:
        text_questions_jsons: 所有的 text_questions.json 文件
    """
    text_questions_jsons = []

    for pmid_folder_name in os.listdir(root_dir):
        pmid_folder_path = os.path.join(root_dir, pmid_folder_name)
        json_file_name = "text_questions.json"
        json_file_path = os.path.join(pmid_folder_path, json_file_name)

        # 检查 json 文件是否存在
        if os.path.exists(json_file_path):
            text_questions_jsons.append(json_file_path)

    return text_questions_jsons


def find_img_questions_jsons(root_dir: str) -> list[str]:
    """
    找到 root_dir 中所有的 img_questions.json 文件

    Args:
        root_dir: 根目录

    Returns:
        img_questions_jsons: 所有的 img_questions.json 文件
    """
    img_questions_jsons = []

    for pmid_folder_name in os.listdir(root_dir):
        pmid_folder_path = os.path.join(root_dir, pmid_folder_name)
        json_file_name = "img_questions.json"
        json_file_path = os.path.join(pmid_folder_path, json_file_name)

        # 检查 json 文件是否存在
        if os.path.exists(json_file_path):
            img_questions_jsons.append(json_file_path)

    return img_questions_jsons


def find_content_list(root_dir: str) -> list[str]:
    """
    从 root_dir 中找到 content_list.json

    Args:
        root_dir: 目录
    
    Returns:
        list[str]: content_list.json 的路径列表
    """
    content_list_paths = []

    for dir_path, dir_names, file_names in os.walk(root_dir):
        for file_name in file_names:
            if file_name.endswith("content_list.json"):
                file_path = os.path.join(dir_path, file_name)
                content_list_paths.append(file_path)

    return content_list_paths


def find_pmid_from_sections_json(sections_json: str) -> str:
    """
    通过 sections.json 文件找到 pmid
    """
    return os.path.dirname(sections_json).split(os.sep)[-1]


def delete_text_questions_jsons(root_dir: str) -> None:
    """
    删除 root_dir 中所有的 text_questions.json 文件

    Args:
        root_dir: 根目录
    """
    text_questions_jsons = find_text_questions_jsons(root_dir)

    logger.info("text_questions.json files: %s", text_questions_jsons)
    input("are you sure to delete all text_questions.json files? (y/n)")

    if input().lower() == "y":
        for text_questions_json in text_questions_jsons:
            os.remove(text_questions_json)

    logger.info("text_questions.json files deleted")


def generate_text_questions() -> None:
    """
    基于 sections.json 生成 text_questions.json, 跑一次就行
    """
    nsclc_dir = "./data/nsclc"
    sections_jsons = find_sections_jsons(nsclc_dir)

    for sections_json in sections_jsons:
        # 找到正在处理的论文 pmid
        pmid = find_pmid_from_sections_json(sections_json)
        logger.info("Processing pmid: %s", pmid)

        # 如果目录下存在 text_questions.json, 则跳过
        text_questions_json = os.path.join(
            os.path.dirname(sections_json),
            "text_questions.json",
        )
        if os.path.exists(text_questions_json):
            logger.info("text_questions.json already exists, skip")
            continue

        # Step 1: 读取章节
        with open(sections_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        text_questions = []
        for section, content in data.items():
            if not content or not content.strip():
                continue

            # Step 2: 生成不同类型的 qa
            text_question = llm.generate_text_questions_from_sections(
                section,
                content,
            )

            for question in text_question:
                question["pmid"] = pmid

            text_questions.extend(text_question)

        # Step 3: 保存到 text_questions.json
        questions_file = os.path.join(
            os.path.dirname(sections_json),
            "text_questions.json",
        )
        with open(questions_file, "w", encoding="utf-8") as f:
            json.dump(text_questions, f, indent=4)


def merge_text_question_jsons() -> None:
    """
    把 llm 生成的 text_questions.json 合并到一个文件, 跑一次就行
    """
    nsclc_dir = "./data/nsclc"
    text_question_jsons = find_text_questions_jsons(nsclc_dir)

    all_text_questions = []
    id_counter = 1
    for text_question_json in text_question_jsons:
        with open(text_question_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            item["id"] = id_counter
            all_text_questions.append(item)
            id_counter += 1

    file_name = os.path.join(nsclc_dir, "all_text_questions.json")
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(all_text_questions, f, indent=4)


def generate_img_questions() -> None:
    """
    生成多模态问题, 跑一次就行
    """
    nsclc_dir = "./data/nsclc"
    # Step 1: 找到 content_list.json
    content_lists = find_content_list(nsclc_dir)

    for content_list in content_lists:
        imgs = []
        img_questions = []
        with open(content_list, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Step 2: 提取其中的 figure 和 table
        for content in data:
            if content["type"] == "image" or content["type"] == "table":
                # 把 img_path 转换为绝对路径
                content["img_path"] = os.path.join(
                    os.path.dirname(content_list), content["img_path"])
                imgs.append(content)

        # Step 3: 生成问题
        for img in imgs:
            img_path = img["img_path"]
            img_caption = ""
            if "table_caption" in img:
                img_caption = "".join(img["table_caption"])
            else:
                img_caption = "".join(img["img_caption"])

            img_question = llm.generate_img_questions(img_path, img_caption)
            img_question["img_path"] = img_path
            img_questions.append(img_question)

        # Step 4: 保存到 img_questions.json
        questions_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(content_list))),
            "img_questions.json",
        )
        with open(questions_file, "w", encoding="utf-8") as f:
            json.dump(img_questions, f, indent=4)


def merge_img_question_jsons() -> None:
    """
    把 llm 生成的 img_questions.json 合并到一个文件, 跑一次就行
    """
    nsclc_dir = "./data/nsclc"
    img_question_jsons = find_img_questions_jsons(nsclc_dir)

    all_img_questions = []
    id_counter = 1
    for img_question_json in img_question_jsons:
        with open(img_question_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            question_dict = {
                "question_id": id_counter,
                "question": item.get("question", ""),
                "options": item.get("options", []),
                "analysis": item.get("reference", ""),
                "correct_answer": item.get("correct_answer", ""),
                "img_path": item.get("img_path", "")
            }
            all_img_questions.append(question_dict)
            id_counter += 1

    file_name = os.path.join(nsclc_dir, "all_img_questions.json")
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(all_img_questions, f, indent=4)


def extract_sections_from_content_list(content_list: str) -> None:
    """
    从 content_list.json 中提取不同章节的内容, 保存到 sections.json

    Args:
        content_list: content_list.json 的路径
    """
    with open(content_list, "r", encoding="utf-8") as f:
        data = json.load(f)

    sections_to_extract = ["methods", "results", "discussion"]
    sections = {section_name: None for section_name in sections_to_extract}
    current_section = None
    section_content = ""

    for content in data:
        if content.get("type") == "text" and content.get("text_level") == 1:
            # 发现新的章节标题
            section_title = content["text"].strip()  # 获取章节标题并去除首尾空格

            # 保存上一个章节的内容 (如果存在)
            if current_section and current_section in sections:
                sections[current_section] = section_content.strip()  # 保存并去除首尾空格

            # 检查新的章节标题是否在要提取的列表中 (需要精确匹配)
            for section_name in sections_to_extract:
                # 忽略大小写，并允许部分匹配
                if (section_title.lower() in section_name.lower() or
                        section_name.lower() in section_title.lower()):
                    current_section = section_name  # 记录当前章节
                    section_content = ""  # 重置章节内容
                    break  # 找到匹配的章节后跳出循环
            else:
                current_section = None  # 如果不在要提取的列表中，则设置为 None
                section_content = ""  # 重置章节内容

        elif content.get("type") == "text" and current_section is not None:
            # 如果是文本内容并且在要提取的章节中，则添加到章节内容
            section_content += content["text"] + " "  # 添加文本内容和空格，保持段落分隔

    # 处理最后一个章节的内容 (循环结束后可能还有内容未保存)
    if current_section and current_section in sections:
        sections[current_section] = section_content.strip()

    section_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(content_list))),
        "sections.json",
    )
    with open(section_file, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=4)


def main():
    """
    主程序入口
    """
    merge_text_question_jsons()


if __name__ == "__main__":
    main()

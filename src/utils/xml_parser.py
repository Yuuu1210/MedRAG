"""
(preprocess) 从 mesh 的原始 xml 文件中提取 mesh 术语
"""
import os
import xml.etree.ElementTree as ET

import pandas as pd
from tqdm import tqdm


def extract_mesh_terms_desc(xml_file_path):
    """
    解析 MeSH 描述符 XML 文件 (例如 desc202x.xml)，并提取
    MeSH UI、首选术语和入口术语。

    Args:
        xml_file_path (str): MeSH 描述符 XML 文件的路径。

    Returns:
        list: 一个字典列表，其中每个字典代表一个术语，
              包含键：'mesh_ui', 'term_string', 'is_preferred'。
              如果文件无法解析或找不到，则返回空列表。
    """
    if not os.path.exists(xml_file_path):
        print(f"错误：文件未找到 '{xml_file_path}'")
        return []

    print(f"正在处理描述符文件：{xml_file_path}...")
    mesh_data = []
    # 使用 iterparse 以提高内存效率，监听 'end' 事件
    context = ET.iterparse(xml_file_path, events=('end',))
    context = iter(context)
    event, root = next(context)  # 获取根元素

    record_count = 0  # 用于 tqdm 的计数器

    # 如果需要精确的总数，可以预先统计，但这会扫描文件一次，可能较慢
    # total_records = sum(1 for event, elem in ET.iterparse(xml_file_path, events=('end',)) if elem.tag == 'DescriptorRecord')
    # print(f"正在估算总记录数... (可能需要一些时间)")
    # 如果估算太慢，可以使用一个大概的数字，或者不给 tqdm 设置 total
    # total_records = 30000 # 描述符的大约数量

    # 初始化 tqdm（不带总数）
    pbar = tqdm(desc="解析描述符中")

    try:
        for event, elem in context:
            # 当找到 DescriptorRecord 元素的结束标签时进行处理
            if event == 'end' and elem.tag == 'DescriptorRecord':
                record_count += 1
                pbar.update(1)  # 更新进度条

                mesh_ui_elem = elem.find('DescriptorUI')
                descriptor_name_elem = elem.find('DescriptorName/String')

                if mesh_ui_elem is not None and descriptor_name_elem is not None:
                    mesh_ui = mesh_ui_elem.text
                    preferred_term = descriptor_name_elem.text

                    # 将首选术语添加到列表中
                    mesh_data.append({
                        'mesh_ui': mesh_ui,
                        'term_string': preferred_term,
                        'is_preferred': True  # 标记为首选术语
                    })

                    # 查找与此描述符关联的所有概念 (concepts)
                    concept_list = elem.find('ConceptList')
                    if concept_list is not None:
                        for concept in concept_list.findall('Concept'):
                            term_list = concept.find('TermList')
                            if term_list is not None:
                                for term in term_list.findall('Term'):
                                    term_string_elem = term.find('String')
                                    if term_string_elem is not None:
                                        entry_term = term_string_elem.text
                                        # 添加入口术语，确保不与首选术语完全重复
                                        # （虽然 MeSH 结构通常在此处避免了这种情况，但以防万一）
                                        if entry_term != preferred_term:
                                            # 在描述符文件中，DescriptorName 是总的首选名称。
                                            # Concept 下的 Term 可能有自己的 'ConceptPreferredTermYN' 属性，
                                            # 但对于链接目的，我们通常关心的是与 DescriptorUI 关联的主要首选名称。
                                            # 我们将所有其他术语标记为非首选（相对于此描述符而言）。
                                            mesh_data.append({
                                                'mesh_ui': mesh_ui,
                                                'term_string': entry_term,
                                                'is_preferred':
                                                    False  # 标记为入口术语 (相对于此 DescriptorUI)
                                            })

                # 从内存中清除已处理的元素以节省空间
                elem.clear()
                # 如果内存仍然是个问题，也可以定期清除根引用，
                # 但如果 iterparse 使用得当，通常不是必需的。
                # 如果内存是大问题，可以考虑使用 lxml，它在这方面处理得更好。

    except ET.ParseError as e:
        print(f"\n解析 XML 时出错: {e}")
        pbar.close()
        return []  # 出错时返回空列表
    except Exception as e:
        print(f"\n发生意外错误: {e}")
        pbar.close()
        return []

    pbar.close()  # 关闭进度条
    print(f"处理完成。提取了 {record_count} 条描述符记录的数据。")
    # 可选：以防万一去除重复项，尽管在此结构下不太可能出现
    # mesh_data = [dict(t) for t in {tuple(d.items()) for d in mesh_data}]
    return mesh_data


def extract_mesh_terms_supp(xml_file_path):
    """
    解析 MeSH 补充概念记录 XML 文件 (例如 supp202x.xml)，并提取
    MeSH UI (SCR UI)、首选术语 (SCR Name) 和入口术语。

    Args:
        xml_file_path (str): MeSH 补充记录 XML 文件的路径。

    Returns:
        list: 一个字典列表，其中每个字典代表一个术语，
              包含键：'mesh_ui', 'term_string', 'is_preferred'。
              如果文件无法解析或找不到，则返回空列表。
    """
    if not os.path.exists(xml_file_path):
        print(f"错误：文件未找到 '{xml_file_path}'")
        return []

    print(f"正在处理补充记录文件：{xml_file_path}...")
    mesh_data = []
    # 使用 iterparse 以提高内存效率
    context = ET.iterparse(xml_file_path, events=('end',))
    context = iter(context)
    event, root = next(context)  # 获取根元素

    record_count = 0

    # 初始化 tqdm（不带总数，因为 SCR 数量变化更大）
    pbar = tqdm(desc="解析补充记录中")

    try:
        for event, elem in context:
            # 当找到 SupplementalRecord 元素的结束标签时进行处理
            if event == 'end' and elem.tag == 'SupplementalRecord':
                record_count += 1
                pbar.update(1)

                # 补充记录使用 SupplementalRecordUI 和 SupplementalRecordName
                mesh_ui_elem = elem.find('SupplementalRecordUI')
                preferred_term_elem = elem.find('SupplementalRecordName/String')

                if mesh_ui_elem is not None and preferred_term_elem is not None:
                    mesh_ui = mesh_ui_elem.text  # 这是 SCR UI，如 C123456
                    preferred_term = preferred_term_elem.text

                    # 将首选术语添加到列表中
                    mesh_data.append({
                        'mesh_ui': mesh_ui,
                        'term_string': preferred_term,
                        'is_preferred': True  # 标记为首选术语
                    })

                    # 补充记录通常也有 ConceptList -> TermList 结构来包含所有相关术语
                    # （包括首选术语本身和其他入口术语）
                    concept_list = elem.find('ConceptList')
                    if concept_list is not None:
                        for concept in concept_list.findall('Concept'):
                            term_list = concept.find('TermList')
                            if term_list is not None:
                                for term in term_list.findall('Term'):
                                    term_string_elem = term.find('String')
                                    if term_string_elem is not None:
                                        entry_term = term_string_elem.text
                                        # 添加入口术语，前提是它不与我们已添加的首选术语完全相同
                                        if entry_term != preferred_term:
                                            mesh_data.append({
                                                'mesh_ui': mesh_ui,
                                                'term_string': entry_term,
                                                'is_preferred': False  # 标记为入口术语
                                            })

                # 从内存中清除已处理的元素
                elem.clear()

    except ET.ParseError as e:
        print(f"\n解析 XML 时出错: {e}")
        pbar.close()
        return []
    except Exception as e:
        print(f"\n发生意外错误: {e}")
        pbar.close()
        return []

    pbar.close()
    print(f"处理完成。提取了 {record_count} 条补充记录的数据。")
    return mesh_data


def extract_mesh_qualifiers(xml_file_path):
    """
    解析 MeSH 限定符 XML 文件 (例如 qual202x.xml)，并提取
    限定符 UI、名称、缩写和入口术语。

    Args:
        xml_file_path (str): MeSH 限定符 XML 文件的路径。

    Returns:
        pd.DataFrame: 包含限定符信息的 Pandas DataFrame，
                      列包括：'qualifier_ui', 'name', 'abbreviation', 'term_string', 'is_preferred'。
                      如果文件无法解析或找不到，则返回空 DataFrame。
    """
    if not os.path.exists(xml_file_path):
        print(f"错误：文件未找到 '{xml_file_path}'")
        return pd.DataFrame(columns=[
            'qualifier_ui', 'name', 'abbreviation', 'term_string',
            'is_preferred'
        ])

    print(f"正在处理限定符文件：{xml_file_path}...")
    qualifier_data = []
    context = ET.iterparse(xml_file_path, events=('end',))
    context = iter(context)
    event, root = next(context)

    record_count = 0
    # 限定符数量不多（比如 76），可以给 tqdm 设置一个总数
    pbar = tqdm(desc="解析限定符中", total=76)

    try:
        for event, elem in context:
            if event == 'end' and elem.tag == 'QualifierRecord':
                record_count += 1
                pbar.update(1)

                qual_ui_elem = elem.find('QualifierUI')
                qual_name_node = elem.find('QualifierName')
                qual_name_elem = None
                if qual_name_node is not None:
                    qual_name_elem = qual_name_node.find('String')

                # 确保能获取到 UI 和首选名称
                if qual_ui_elem is None or qual_name_elem is None:
                    print(
                        f"警告：记录 {record_count} 缺少 QualifierUI 或 QualifierName/String。跳过..."
                    )
                    elem.clear()
                    continue  # 跳过此记录

                qual_ui = qual_ui_elem.text
                preferred_name = qual_name_elem.text
                abbreviation = None  # 初始化缩写为 None
                found_terms = []  # 用于临时存储此记录的所有术语及其首选状态

                # 现在深入查找缩写和所有术语
                concept_list = elem.find('ConceptList')
                if concept_list is not None:
                    for concept in concept_list.findall('Concept'):
                        term_list = concept.find('TermList')
                        if term_list is not None:
                            for term in term_list.findall('Term'):
                                term_string_elem = term.find('String')
                                if term_string_elem is not None:
                                    current_term_string = term_string_elem.text
                                    # 检查这个 Term 是否是整个记录的首选术语 (根据 RecordPreferredTermYN)
                                    # 同时我们也关心它是否是当前 Concept 的首选术语 (ConceptPreferredTermYN)，但主要依据 Record 来找缩写
                                    is_record_preferred = term.get(
                                        'RecordPreferredTermYN') == 'Y'

                                    # 如果是记录级别的首选术语，尝试从中提取缩写
                                    if is_record_preferred:
                                        abbrev_elem = term.find('Abbreviation')
                                        if abbrev_elem is not None:
                                            abbreviation = abbrev_elem.text  # 找到了！
                                        # 将(首选术语字符串, is_preferred=True) 添加到临时列表
                                        # 即使 QualifierName/String 和这里的 String 相同，也通过这里添加
                                        # （避免重复添加，并确保 is_preferred 标记正确）
                                        if not any(
                                                t[0] == current_term_string
                                                for t in found_terms
                                        ):  # 避免因Concept结构重复添加同一个 preferred term string
                                            found_terms.append(
                                                (current_term_string, True))
                                    else:
                                        # 将(入口术语字符串, is_preferred=False) 添加到临时列表
                                        if not any(t[0] == current_term_string
                                                   for t in
                                                   found_terms):  # 避免重复添加入口术语
                                            found_terms.append(
                                                (current_term_string, False))

                # 在处理完一个 QualifierRecord 的所有术语后，将它们与找到的（或为None的）缩写一起添加到最终列表
                if not found_terms:
                    # 如果 ConceptList/TermList 为空或无法解析，至少添加顶层的首选名称自身
                    print(
                        f"警告：记录 {record_count} ({qual_ui}) 未在其 TermList 中找到术语。仅添加 QualifierName。"
                    )
                    qualifier_data.append({
                        'qualifier_ui': qual_ui,
                        'name': preferred_name,  # 限定符的官方名称
                        'abbreviation': abbreviation,  # 可能仍然是 None
                        'term_string': preferred_name,  # 使用官方名称作为术语
                        'is_preferred': True  # 标记为首选
                    })
                else:
                    # 使用找到的（或为None的）缩写为所有收集到的术语添加条目
                    for term_str, is_pref in found_terms:
                        qualifier_data.append({
                            'qualifier_ui': qual_ui,
                            'name': preferred_name,  # 所有术语都关联到这个首选名称
                            'abbreviation':
                                abbreviation,  # 对该 QualifierRecord 的所有术语使用同一个缩写
                            'term_string': term_str,
                            'is_preferred': is_pref  # 使用之前判断好的标记
                        })

                elem.clear()  # 清理内存

    except ET.ParseError as e:
        print(f"\n解析 XML 时出错: {e}")
        pbar.close()
        return pd.DataFrame(columns=[
            'qualifier_ui', 'name', 'abbreviation', 'term_string',
            'is_preferred'
        ])
    except Exception as e:
        print(f"\n发生意外错误: {e}")
        pbar.close()
        return pd.DataFrame(columns=[
            'qualifier_ui', 'name', 'abbreviation', 'term_string',
            'is_preferred'
        ])

    pbar.close()
    print(f"处理完成。提取了 {record_count} 条限定符记录的数据。")

    # 转换为 DataFrame
    if not qualifier_data:  # 做个最终检查
        print("警告：最终 qualifier_data 列表为空。")
        return pd.DataFrame(columns=[
            'qualifier_ui', 'name', 'abbreviation', 'term_string',
            'is_preferred'
        ])

    df_qual = pd.DataFrame(qualifier_data)

    # 可选：清理 - 移除基于 (qualifier_ui, term_string, is_preferred) 完全相同的重复行
    # 这可以处理 XML 中潜在的冗余或解析逻辑可能引入的重复
    initial_len = len(df_qual)
    df_qual = df_qual.drop_duplicates(
        subset=['qualifier_ui', 'term_string', 'is_preferred'])
    if len(df_qual) < initial_len:
        print(f"清理：移除了 {initial_len - len(df_qual)} 条重复的术语条目。")

    return df_qual


def main():
    """
    主程序入口
    """
    # ===========================
    # 处理 desc2025.xml 文件
    # ===========================
    # mesh_data_dir = './data/mesh'  # 确保路径正确
    # desc_file = os.path.join(mesh_data_dir, 'desc2025.xml')

    # descriptor_terms = extract_mesh_terms_from_desc(desc_file)

    # # 打印一些结果来检查
    # if descriptor_terms:
    #     print(
    #         f"\n成功从 {os.path.basename(desc_file)} 提取了 {len(descriptor_terms)} 个术语。"
    #     )
    #     print("示例数据 (前 5 条):")
    #     for i, item in enumerate(descriptor_terms[:5]):
    #         print(item)
    #     print("\n示例数据 (后 5 条):")
    #     for i, item in enumerate(descriptor_terms[-5:]):
    #         print(item)

    #     # 你可能想将这些数据保存到文件（例如 CSV）中供以后使用
    #     df_desc = pd.DataFrame(descriptor_terms)
    #     df_desc.to_csv(os.path.join(mesh_data_dir, 'mesh_descriptor_terms.csv'),
    #                    index=False,
    #                    encoding='utf-8')  # 注意编码
    #     print("\n数据已保存到 mesh_descriptor_terms.csv")

    # else:
    #     print(f"\n未能从 {os.path.basename(desc_file)} 提取术语。")

    # ===========================
    # 处理 supp2025.xml 文件
    # ===========================
    # mesh_data_dir = './data/mesh/'  # 确保路径正确
    # supp_file = os.path.join(mesh_data_dir, 'supp2025.xml')

    # supplemental_terms = extract_mesh_terms_from_supp(supp_file)

    # # 打印一些结果来检查
    # if supplemental_terms:
    #     print(
    #         f"\n成功从 {os.path.basename(supp_file)} 提取了 {len(supplemental_terms)} 个术语。"
    #     )
    #     print("示例数据 (前 5 条):")
    #     for i, item in enumerate(supplemental_terms[:5]):
    #         print(item)
    #     print("\n示例数据 (后 5 条):")
    #     for i, item in enumerate(supplemental_terms[-5:]):
    #         print(item)

    #     # 可以将这些数据合并到之前的数据中或单独保存
    #     df_supp = pd.DataFrame(supplemental_terms)
    #     df_supp.to_csv(os.path.join(mesh_data_dir, 'supplemental_terms.csv'),
    #                    index=False,
    #                    encoding='utf-8')
    #     print("\n数据已保存到 supplemental_terms.csv")

    # else:
    #     print(f"\n未能从 {os.path.basename(supp_file)} 提取术语。")

    # ===========================
    # 合并 descriptor_terms.csv 和 supplemental_terms.csv
    # ===========================
    mesh_data_dir = './data/mesh/'  # 确保路径正确

    # 从 CSV 加载 descriptor_terms
    desc_csv_path = os.path.join(mesh_data_dir, 'descriptor_terms.csv')
    if os.path.exists(desc_csv_path):
        print(f"从 {desc_csv_path} 加载描述符数据...")
        df_desc = pd.read_csv(desc_csv_path)
    else:
        print(f"警告：找不到 CSV 文件 '{desc_csv_path}'。")
        df_desc = pd.DataFrame(
            columns=['mesh_ui', 'term_string', 'is_preferred'])  # 创建空 DataFrame

    # 从 CSV 加载 supplemental_terms
    supp_csv_path = os.path.join(mesh_data_dir, 'supplemental_terms.csv')
    if os.path.exists(supp_csv_path):
        print(f"从 {supp_csv_path} 加载补充记录数据...")
        df_supp = pd.read_csv(supp_csv_path)
    else:
        print(f"找不到 CSV 文件 '{supp_csv_path}'。")
        df_supp = pd.DataFrame(
            columns=['mesh_ui', 'term_string', 'is_preferred'])  # 创建空 DataFrame

    # 只有当两个 DataFrame 都至少有一个有效时才进行合并
    if not df_desc.empty or not df_supp.empty:
        # 合并 DataFrame
        df_combined = pd.concat([df_desc, df_supp], ignore_index=True)
        print(f"\n成功合并数据。总条目数: {len(df_combined)}")

        # 打印一些信息和示例
        print(df_combined.info())  # 查看列类型和非空值数量
        print("\n合并后数据示例 (头部):")
        print(df_combined.head())
        print("\n合并后数据示例 (尾部):")
        print(df_combined.tail())

        # 可选：清理和检查
        # 1. 去除完全重复的行 (同一 UI，同一术语字符串，同一首选状态)
        initial_len = len(df_combined)
        df_combined = df_combined.drop_duplicates()
        if len(df_combined) < initial_len:
            print(f"\n移除了 {initial_len - len(df_combined)} 条完全重复的行。")

        # 2. 检查是否有空值或 NaN
        if df_combined.isnull().values.any():
            print("\n警告：合并后的数据中存在空值 (NaN)！")
            print(df_combined[df_combined.isnull().any(axis=1)])
            # 可以考虑填充或删除这些行，取决于具体情况
            # df_combined = df_combined.dropna() # 简单粗暴地删除含 NaN 的行

        # 3. 检查是否有同一个术语字符串映射到不同的 MeSH UI (歧义词)
        ambiguous_terms = df_combined[df_combined.duplicated(
            subset=['term_string'], keep=False)]
        num_ambiguous = ambiguous_terms['term_string'].nunique()
        if num_ambiguous > 0:
            print(f"\n注意：发现 {num_ambiguous} 个术语字符串映射到多个 MeSH UI (潜在歧义)。")
            # 在进行实体链接时，这需要特别处理（例如，返回所有可能的链接让下游决定，或根据上下文选择）

        # 保存合并后的数据
        combined_csv_path = os.path.join(mesh_data_dir, 'combined_terms.csv')
        df_combined.to_csv(combined_csv_path, index=False, encoding='utf-8')
        print(f"\n合并后的数据已保存到 {combined_csv_path}")

    else:
        print("\n错误：描述符数据和补充记录数据均为空，无法进行合并。")

    # ===========================
    # 处理 qual2025.xml 文件
    # ===========================
    # mesh_data_dir = './data/mesh/'  # 确保路径正确
    # qual_file = os.path.join(mesh_data_dir, 'qual2025.xml')

    # df_qualifiers = extract_mesh_qualifiers(qual_file)

    # # 打印一些结果来检查
    # if not df_qualifiers.empty:
    #     print(
    #         f"\n成功从 {os.path.basename(qual_file)} 提取了 {len(df_qualifiers)} 个限定符相关术语。"
    #     )
    #     print(f"共有 {df_qualifiers['qualifier_ui'].nunique()} 个唯一的限定符。")
    #     print("限定符 DataFrame 信息:")
    #     print(df_qualifiers.info())
    #     print("\n限定符数据示例 (前 5 条):")
    #     print(df_qualifiers.head())
    #     print("\n限定符数据示例 (后 5 条):")
    #     print(df_qualifiers.tail())

    #     # 保存限定符数据
    #     qual_csv_path = os.path.join(mesh_data_dir, 'qualifiers.csv')
    #     df_qualifiers.to_csv(qual_csv_path, index=False, encoding='utf-8')
    #     print(f"\n限定符数据已保存到 {qual_csv_path}")

    # else:
    #     print(f"\n未能从 {os.path.basename(qual_file)} 提取限定符术语。")


if __name__ == "__main__":
    main()

"""
存放全局变量及配置信息
"""
# kimi api
MOONSHOT_API_KEY = "sk-Pn6XzJfaLRzKtFRYP9GQTQTUs3YkNRqdy2OinIz44YqBvJkd"
MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"

# 火山引擎 api
# 调用 doubao 模型
ARK_API_KEY = "e0eaf79d-2d9c-4ec5-b800-65c4a3ab8808"
ARK_ENDPOINT_ID = "ep-20250215165654-tf6gh"
ARK_ENDPOINT_ID2 = "ep-20250512122835-q8vx2"
DEEPSEEK_ENDPOINT_ID = "ep-20250525104928-dvcjf"
ARK_VISION_ENDPOINT_ID = "ep-20250216182848-2lr76"

# 阿里百炼 api
# 调用 qwen 模型
DASHSCOPE_API_KEY = "sk-4dbc03ad554748aa8d2bf307a190bcd2"

OPENAI_API_KEY = ("sk-proj-A-vV2sICQogIqd8x1jJdDIGDEMP8WBap81qYyhxw9WHHIyaq"
                  "4LTRfJy2PQlAr2vEVvywbrGDeaT3BlbkFJ2TGEHandCwzSxpGQ8ogN"
                  "2QHvk9XIGsQzF4qWiUhu1GwizC_hPpjwOSG7LGPa2YB7pgubd90F0A")

ANTHROPIC_API_KEY = ("sk - ant - api03 - RZxD_UGGccWivHdgfiU_zuJ5V6_"
                     "eCImAQeerpO_X0tUrXGdlnAmxZsuwtnnQhqlm_"
                     "y4o3GE0GZ4xNvoUzyDoEQ - ESHOxgAA")

# pubmed api
NCBI_API_KEY = "e6c416942cef2b3f727af51fb2bc1f324208"
NCBI_EMAIL = "3178804266@qq.com"

# 百度智能云格式转换 api
BAIDU_API_KEY = "mSWjFqBMfx2KlAeSeuwrsIhQ"
BAIDU_SECRET_KEY = "dE2ijU9drSTEaF6cVagIQxrWrFhDuXRr"

# neo4j api
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "rssychldcj962464"

# Serper api
SERPER_API_KEY = "111da9a4a638b74690eef703e976131c9bcfca9d"

# 特定 API 返回结果数
PUBMED_API_RETMAX = 5
DDG_API_RETMAX = 5

# 构建 NSCLC 数据集使用的 example
METHODS_QUESTION_EXAMPLE = {
    "Methodological Details":
        "Could you elaborate on the specific inclusion and exclusion criteria used for patient selection in this study, and how these criteria might influence the generalizability of the findings to a broader NSCLC population?",
    "Critical Evaluation":
        "Considering the study's methodology, what are the potential limitations or biases inherent in the chosen study design (e.g., retrospective analysis, single-arm trial), and how might these factors affect the interpretation of the results?",
    "Clinical Application":
        "Based on the methodology described for biomarker assessment (e.g., liquid biopsy, IHC), how feasible and practical would it be to implement these techniques in routine clinical practice for NSCLC patient management?"
}
RESULTS_QUESTION_EXAMPLE = {
    "Key Findings":
        "What were the primary and secondary endpoints reported in the results section, and what were the key statistical measures (e.g., Hazard Ratio, Confidence Interval, p-value) associated with the main findings regarding treatment efficacy or biomarker association?",
    "Interpretation & Significance":
        "How do the authors interpret the magnitude of the observed treatment effect or biomarker association in the context of clinically meaningful outcomes for NSCLC patients? Is the reported effect size clinically significant, and what is the clinical relevance of these findings?",
    "Comparative Analysis":
        "When comparing the results across different treatment arms or patient subgroups, what were the notable differences or similarities observed in terms of efficacy or safety outcomes? Are there any statistically significant differences between groups, and what are the potential clinical implications of these comparisons?"
}
DISCUSSION_QUESTION_EXAMPLE = {
    "Author's Interpretation":
        "How do the authors contextualize their findings within the existing body of literature on NSCLC? Do they highlight areas of agreement or disagreement with prior studies, and how do they explain any discrepancies or novel insights?",
    "Limitations & Future Directions":
        "What limitations of the study do the authors acknowledge in the discussion section, and are these limitations adequately addressed in their interpretation of the results? Furthermore, what specific future research directions do they propose to further validate or expand upon their current findings?",
    "Clinical and Research Implications":
        "Based on the study's findings and discussion, what are the potential implications for current clinical practice guidelines or future research strategies in NSCLC? How might these results influence treatment decision-making, patient stratification, or the development of novel therapeutic approaches?"
}

# medqa 数据集提供的教科书
EN_TEXTBOOKS = [{
    "id": 1,
    "category": "Anatomy",
    "file_path": ["./data/dataset/medqa/textbooks/en/Anatomy_Gray.txt"]
}, {
    "id": 2,
    "category": "Histology and Embryology",
    "file_path": ["./data/dataset/medqa/textbooks/en/Histology_Ross.txt"]
}, {
    "id": 3,
    "category": "Physiology",
    "file_path": ["./data/dataset/medqa/textbooks/en/Physiology_Levy.txt"]
}, {
    "id":
        4,
    "category":
        "Biochemistry",
    "file_path": [
        "./data/dataset/medqa/textbooks/en/Biochemistry_Lippincott.txt"
    ]
}, {
    "id": 5,
    "category": "Cell Biology",
    "file_path": ["./data/dataset/medqa/textbooks/en/Cell_Biology_Alberts.txt"]
}, {
    "id": 6,
    "category": "General Pathology",
    "file_path": ["./data/dataset/medqa/textbooks/en/Pathology_Robbins.txt"]
}, {
    "id": 7,
    "category": "Pathophysiology",
    "file_path": ["./data/dataset/medqa/textbooks/en/Pathoma_Husain.txt"]
}, {
    "id": 8,
    "category": "Internal Medicine",
    "file_path": ["./data/dataset/medqa/textbooks/en/InternalMed_Harrison.txt"]
}, {
    "id": 9,
    "category": "Surgery",
    "file_path": ["./data/dataset/medqa/textbooks/en/Surgery_Schwartz.txt"]
}, {
    "id":
        10,
    "category":
        "Obstetrics and Gynecology",
    "file_path": [
        "./data/dataset/medqa/textbooks/en/Gynecology_Novak.txt",
        "./data/dataset/medqa/textbooks/en/Obstentrics_Williams.txt"
    ]
}, {
    "id": 11,
    "category": "Pediatrics",
    "file_path": ["./data/dataset/medqa/textbooks/en/Pediatrics_Nelson.txt"]
}, {
    "id":
        12,
    "category":
        "Neurology and Psychiatry",
    "file_path": [
        "./data/dataset/medqa/textbooks/en/Neurology_Adams.txt",
        "./data/dataset/medqa/textbooks/en/Psichiatry_DSM-5.txt"
    ]
}, {
    "id": 13,
    "category": "Pharmacology",
    "file_path": ["./data/dataset/medqa/textbooks/en/Pharmacology_Katzung.txt"]
}, {
    "id": 14,
    "category": "Immunology",
    "file_path": ["./data/dataset/medqa/textbooks/en/Immunology_Janeway.txt"]
}, {
    "id":
        15,
    "category":
        "First Aid",
    "file_path": [
        "./data/dataset/medqa/textbooks/en/First_Aid_Step1.txt",
        "./data/dataset/medqa/textbooks/en/First_Aid_Step2.txt"
    ]
}]
EN_TEXTBOOKS_INVERTED_INDEX = {
    'Anatomy': ['./data/textbook/en_inverted_index/Anatomy_Gray.json'],
    'Histology and Embryology': [
        './data/textbook/en_inverted_index/Histology_Ross.json'
    ],
    'Physiology': ['./data/textbook/en_inverted_index/Physiology_Levy.json'],
    'Biochemistry': [
        './data/textbook/en_inverted_index/Biochemistry_Lippincott.json'
    ],
    'Cell Biology': [
        './data/textbook/en_inverted_index/Cell_Biology_Alberts.json'
    ],
    'General Pathology': [
        './data/textbook/en_inverted_index/Pathology_Robbins.json'
    ],
    'Pathophysiology': [
        './data/textbook/en_inverted_index/Pathoma_Husain.json'
    ],
    'Internal Medicine': [
        './data/textbook/en_inverted_index/InternalMed_Harrison.json'
    ],
    'Surgery': ['./data/textbook/en_inverted_index/Surgery_Schwartz.json'],
    'Obstetrics and Gynecology': [
        './data/textbook/en_inverted_index/Gynecology_Novak.json',
        './data/textbook/en_inverted_index/Obstentrics_Williams.json'
    ],
    'Pediatrics': ['./data/textbook/en_inverted_index/Pediatrics_Nelson.json'],
    'Neurology and Psychiatry': [
        './data/textbook/en_inverted_index/Neurology_Adams.json',
        './data/textbook/en_inverted_index/Psichiatry_DSM-5.json'
    ],
    'Pharmacology': [
        './data/textbook/en_inverted_index/Pharmacology_Katzung.json'
    ],
    'Immunology': ['./data/textbook/en_inverted_index/Immunology_Janeway.json'],
    'First Aid': [
        './data/textbook/en_inverted_index/First_Aid_Step1.json',
        './data/textbook/en_inverted_index/First_Aid_Step2.json'
    ]
}

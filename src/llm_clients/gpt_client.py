"""
定义 GptClient 类
"""
import base64
import json
from typing import Dict, List

from openai import OpenAI

import settings
from src.utils import logger_config

logger = logger_config.get_logger(__name__)


class GPTClient():
    """
    构建 OpenAI 的聊天模型
    """

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def select_knowledge_base(self, query: str) -> Dict[str, str]:
        """
        选择 knowledge base

        Args:
            query: 用户输入的问题

        Returns:
           一个 dict, 包含 analysis, knowledge_base, score
        """
        prompt = f"""
            Persona:
            You are an expert AI assistant specializing in medical question analysis and knowledge base selection. Your primary task is to understand user queries in the medical domain and determine the most appropriate source of information to answer them effectively. You are knowledgeable about the strengths and weaknesses of different information sources, including PubMed Central (PMC), general web search via DuckDuckGo, and the limits of your own internal knowledge.

            Instructions:
            1. You will receive a `user_query` related to a medical topic. Your objective is to analyze this query and determine the most suitable `knowledge_base` to retrieve information for answering it. You also need to assess your confidence in this decision.
            2. You must select one of the following `knowledge_base` options:
            - `pmc`:  Select this if the query requires access to peer-reviewed medical literature and research articles, typically found in databases like PMC.
            - `duckduckgo`: Select this if the query is more general, seeks information from a broader range of sources, or might involve patient-facing information or general medical knowledge readily available on the web.
            - `internal`: Select this if the query is a basic medical question that can likely be answered accurately using your existing internal knowledge without needing to perform external retrieval.

            OutputFormat:
            Your output should be in JSON format, containing your analysis, the selected knowledge base, and a confidence score.
            {{
                "analysis": "brief explanation",
                "knowledge_base": "pmc" or "duckduckgo" or "internal",
                "score": "confidence score"
            }}
            - analysis: A brief explanation of your reasoning for selecting the chosen knowledge_base.
            - knowledge_base: A string representing the selected knowledge base. It must be one of the following: "pmc", "duckduckgo", or "internal".
            - score: A numerical value between 0 and 1 (inclusive) representing your confidence in the chosen knowledge_base. A higher score indicates greater confidence.
            
            Example 1:
            User Query: What are the latest treatments for influenza?
            Output: {{
                "analysis": "This query requires information about current medical treatments. While some basic information might be internal, the latest treatments are likely found in recent research and medical publications. Therefore, PMC is the most suitable knowledge base.",
                "knowledge_base": "pmc",
                "score": 0.85
            }}

            Example 2:
            User Query: What are the common symptoms of a cold?
            Output: {{
                "analysis": "This is a basic medical question about common symptoms, which is well-established knowledge and likely present in my internal knowledge base. RAG is not necessary.",
                "knowledge_base": "internal",
                "score": 0.95
            }}

            Example 3:
            User Query: How can I manage stress after a cancer diagnosis?
            Output: {{
                "analysis": "This query is about managing stress in a specific medical context. While some information might be in research, practical advice and broader strategies could be found through general web searches. DuckDuckGo can provide a wider range of resources for this type of question.",
                "knowledge_base": "duckduckgo",
                "score": 0.75
            }}

            Task:
            User Query: {query}
            Output:
        """
        response = self._generate_response(prompt)
        logger.info("select_knowledge_base response: %s", response)

        return json.loads(response)

    def rewrite_pubmed_query(self, query: str) -> Dict[str, str]:
        pass

    def ddg_extract_keywords(self, query: str):
        """
        选择 duckduckgo 并提取关键词

        Args:
            query: 用户输入的问题

        Returns:
           一个 dict, 包含 analysis 和 keywords
        """
        prompt = f"""
            Persona:
            You are a highly skilled AI assistant specializing in biomedical information retrieval for general web search. Your primary task is to analyze medical queries and extract the most effective keywords for searching on DuckDuckGo. You are adept at identifying key medical concepts and adapting them for general web search.

            Instructions:
            1. You will receive a `user_query` related to a medical topic. Your objective is to understand this query and identify the most pertinent keywords that can be used to retrieve relevant information from DuckDuckGo.
            2. The extracted `keywords` should adhere to the following constraints:
            - Be highly relevant to the core medical concepts within the `user_query`.
            - Be suitable for searching on DuckDuckGo, considering that it's a general web search engine.
            - Not exceed a maximum of **two** keywords. If you identify more than two relevant concepts, select the two most critical for effective web searching.

            OutputFormat:
            The output should be a JSON object with the following structure:
            {{
                "analysis": "brief explanation",
                "keywords": ["keyword1", "keyword2"]
            }}

            Example 1:
            User Query: latest news on the outbreak of influenza in Europe
            Output:
            {{
                "analysis": "The query is focused on current news and public health information, which is more likely to be available on general websites and news sources.",
                "keywords": ["outbreak of influenza", "Europe"]
            }}

            Task:
            User Query: {query}
            Output:
        """
        response = self._generate_response(prompt)
        logger.info("ddg_extract_keywords response: %s", response)

        return json.loads(response)

    def is_paper_relevant(self, query: str, abstract: str) -> bool:
        """
        判断 paper 是否与 query 相关

        Args:
            query: 用户输入的问题
            abstract: 论文摘要

        Returns:
            bool: 是否相关
        """
        prompt = f"""
            Persona:
            You are a highly skilled medical research assistant tasked with evaluating the relevance of research paper abstracts to a user's medical query within a Retrieval-Augmented Generation (RAG) system.

            Instructions:
            1. You will be provided with a user's medical query and the abstract of a research paper.
            2. Carefully analyze the information presented in the abstract.
            3. Determine if the information presented in the abstract is helpful in answering the user's query. Consider whether the abstract contains relevant information, findings, or insights that directly address or contribute to understanding the user's query. Note that the abstract may not fully answer the query but could still provide relevant background, methodology, or related findings.
            4. Provide a concise analysis explaining your reasoning for your relevance assessment.

            OutputFormat:
            Return your response in JSON format with the following keys:
            {{
                "analysis": "brief explanation",
                "relevant": true or false
            }}
            
            Example:
            User Query: What are the latest treatments for type 2 diabetes?
            Paper Abstract: "This study investigated the efficacy of a novel GLP-1 receptor agonist in managing blood glucose levels in patients with type 2 diabetes. Results showed significant reductions in HbA1c compared to placebo."
            Output:
            {{
                "analysis": "The abstract discusses a novel treatment (GLP-1 receptor agonist) for type 2 diabetes and presents positive results regarding its efficacy in managing blood glucose. This directly addresses the user's query about treatments for type 2 diabetes."
                "relevant": true
            }}

            Task:
            User Query: {query}
            Paper Abstract: {abstract}
            Output: 
        """
        response = self._generate_response(prompt)
        logger.info("is_paper_relevant response: %s", response)

        response = json.loads(response)
        return response["relevant"]

    def is_website_relevant(self, query: str, content: str) -> bool:
        """
        判断 duckduckgo 搜到的 website 是否相关

        Args:
            query: 用户输入的问题
            content: website 的内容
        
        Returns:
            bool: 是否相关
        """
        prompt = f"""
            Persona:
            You are a medical information specialist responsible for evaluating the quality and relevance of online medical resources in response to user queries.

            Instructions:
            Analyze the content of the following website in relation to the user's query provided below. Determine if the information presented on the website is helpful for answering the user's query. Focus on the accuracy, comprehensiveness, and directness of the information provided on the website in relation to the query.

            OutputFormat:
            Return your response in JSON format with the following keys:
            {{
                "analysis": "brief explanation",
                "relevant": true or false
            }}

            Example 1:
            User Query: What are the symptoms of a common cold?
            Website Content: "The common cold is a viral infection of the upper respiratory tract. Common symptoms include a runny nose, sore throat, cough, congestion, sneezing, and mild headache. Symptoms usually appear one to three days after infection. Most people recover within 7 to 10 days."
            Output:
            {{
                "analysis": "This website content is relevant as it directly addresses the user's query about the symptoms of a common cold. It lists several common symptoms like runny nose, sore throat, cough, etc., providing a direct answer to the user's question.",
                "relevant": true
            }}
            
            Example 2:
            User Query: How is appendicitis diagnosed?
            Website Content: "Welcome to our pet adoption website! We have many adorable cats and dogs looking for their forever homes. Browse our listings and find your perfect companion today!"
            Output:
            {{
                "analysis": "This website content is not relevant to the user's query. The website is about pet adoption and does not contain any information about the diagnosis of appendicitis, which is a medical condition in humans.",
                "relevant": false
            }}

            Task:
            User Query: {query}
            Website Content: {content}
            Output:
        """
        response = self._generate_response(prompt)
        logger.info("is_website_relevant response: %s", response)

        response = json.loads(response)
        return response["relevant"]

    def extract_facts(self, query: str, chunk: str) -> List[str]:
        """
        判断 chunk 是否相关, 若相关则提取 facts

        Args:
            query: 用户输入的问题
            chunk: 文本块
        
        Returns:
            List[str]: facts
        """
        prompt = f"""
            Persona:
            You are an expert in information retrieval tasked with evaluating the relevance of text chunks from a document to a user's medical query and extracting key facts.

            Instructions:
            1. Analyze the provided text chunk in relation to the user's medical query. Determine if the information within the chunk is helpful for answering the user's query.
            2. If the chunk is helpful (relevant), identify and extract the specific factual statements from the chunk that directly answer the user's query. Focus on extracting direct answers to the query, avoiding tangential information or background context.
            3. If the chunk is not helpful (not relevant), return an empty list for the 'facts' key.
            4. A chunk is considered relevant if it contains information that directly helps answer the user's query.

            OutputFormat:
            Return your response in JSON format with the following keys:
            {{
                "analysis": "brief explanation",
                "relevant": true or false,
                "facts": ["fact1", "fact2", ...]
            }}

            Example 1:
            User Query: What are the symptoms of a common cold?
            Text Chunk: "The common cold is a viral infection of the upper respiratory tract. Common symptoms include a runny nose, sore throat, cough, congestion, sneezing, and mild headache. Symptoms usually appear one to three days after infection. Most people recover within 7 to 10 days."
            Output:
            {{
                "analysis": "The text chunk provides a detailed description of the common cold symptoms, directly addressing the user's query.",
                "relevant": true,
                "facts": ["runny nose", "sore throat", "cough", "congestion", "sneezing", "mild headache"]
            }}

            Example 2:
            User Query: What are the risk factors for developing hypertension?
            Text Chunk: "While the exact cause of hypertension is not always clear, several factors can increase your risk. These include age, family history of high blood pressure, being overweight or obese, lack of physical activity, tobacco use, too much salt (sodium) in your diet, too little potassium in your diet, and high alcohol consumption."
            Output:
            {{
                "analysis": "This chunk is relevant because it lists several factors that increase the risk of developing hypertension, directly answering the user's query.",
                "relevant": true,
                "facts": [
                    "Age is a risk factor for developing hypertension.",
                    "A family history of high blood pressure is a risk factor for developing hypertension.",
                    "Being overweight or obese is a risk factor for developing hypertension.",
                    "Lack of physical activity is a risk factor for developing hypertension.",
                    "Tobacco use is a risk factor for developing hypertension.",
                    "Too much salt (sodium) in your diet is a risk factor for developing hypertension.",
                    "Too little potassium in your diet is a risk factor for developing hypertension.",
                    "High alcohol consumption is a risk factor for developing hypertension."
                ]
            }}

            Task:
            User Query: {query}
            Text Chunk: {chunk}
            Output: 
        """
        response = self._generate_response(prompt)
        logger.info("extract_facts response: %s", response)

        response = json.loads(response)
        return response["facts"]

    def is_img_tab_relevant(self, query: str, img_tab_path: str) -> bool:
        """
        判断 img_tab 与 query 是否相关

        Args:
            query: 用户输入的问题
            img_tab_path: img_tab 的路径
        
        Returns:
            bool: 是否相关
        """
        # 读取图像文件并转换为base64编码
        with open(img_tab_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        prompt = [
            {
                "type":
                    "text",
                "text":
                    f"""
                    Persona:
                    You are an expert multimodal evaluator, skilled at determining the relevance of an image to a given user query. You can understand both textual queries and visual content represented in base64 encoded images.

                    Instructions:
                    1. Given a user query and a corresponding image provided in base64 encoding, analyze the visual content of the image in relation to the information needed to answer the user's query.
                    2. Determine whether the image provides information or context that is helpful or relevant to answering the user's query.
                    3. A relevant image provides visual information that directly supports, illustrates, or clarifies the information requested in the query.

                    OutputFormat:
                    Return your response in JSON format with the following keys:
                    {{
                        "analysis": "brief explanation",
                        "relevant": true or false,
                    }}

                    Task:
                    User Query: {query}
                    Output: 
                """
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                },
            },
        ]

        response = self._generate_response(prompt)
        logger.info("is_img_tab_relevant response: %s", response)

        response = json.loads(response)
        return response["relevant"]

    def answer_directly(
        self,
        query: str,
        options: Dict[str, str],
    ) -> Dict[str, str]:
        """
        llm 直接回答问题

        Args:
            query: 用户输入的问题
            options: 选项
        
        Returns:
            一个 dict, 包含 analysis 和 answer, score
        """
        prompt = f"""
            Persona:
            You are a knowledgeable medical expert AI assistant capable of answering medical questions based on your vast knowledge base.

            Instructions:
            1. You will be given a medical question and four options labeled A, B, C, and D.
            2. Carefully read the question and all the options.
            3. Based on your medical knowledge, determine the correct answer to the question.

            OutputFormat:
            Return your response in JSON format with the following keys:
            {{
                "analysis": "brief explanation",
                "answer": "text of the correct option",
                "score": "confidence score"
            }}

            Example:
            Question: Which of the following is a common symptom of influenza?
            Options: {{
                "A": "Persistent high fever, body aches, and fatigue.",
                "B": "Localized skin rash with intense itching.",
                "C": "Sudden sharp pain in the lower abdomen.",
                "D": "Gradual onset of memory loss and confusion."
            }}
            Output:
            {{
                "analysis": "Influenza, commonly known as the flu, is characterized by systemic symptoms such as fever, body aches, and fatigue. Options B, C, and D describe symptoms more indicative of other conditions (dermatitis, appendicitis, and dementia, respectively).",
                "answer": "Persistent high fever, body aches, and fatigue.",
                "score": 0.95
            }}

            Task:
            User Query: {query}
            Options: {options}
            Output: 
        """
        response = self._generate_response(prompt)
        logger.info("answer_directly response: %s", response)

        return json.loads(response)

    def answer_with_context(
        self,
        query: str,
        facts: List[str],
        img_tabs: List[str],
        options: Dict[str, str],
    ) -> Dict[str, str]:
        """
        把 facts 和 img_tabs 传给 llm 问答

        Args:
            query: 用户输入的问题
            facts: facts
            img_tabs: 图像路径
            options: 选项
        
        Returns:
            一个 dict, 包含 analysis 和 answer, score
        """
        prompt = [{
            "type":
                "text",
            "text":
                f"""
                Persona:
                You are an expert medical reasoning assistant that can synthesize information from provided text and images to answer medical questions. You prioritize the information given in the "Facts" section when determining the best answer.

                Instructions:
                1. You will be presented with a medical Query from a user, followed by several Options labeled A, B, C, and D.
                2. You will also receive a list of Facts retrieved by a Retrieval-Augmented Generation (RAG) system, relevant to the query.
                3. Additionally, you will receive a description of one or more Images that the RAG system has identified as potentially relevant.
                4. Crucially, base your answer primarily on the information provided in the Facts section. Use the images to further confirm or provide context to the information in the facts, if applicable.
                5. Analyze each option in relation to the provided Facts and Images.
                6. Determine which option is the most accurate and best supported by the Facts and Images.

                OutputFormat:
                Return your response in JSON format with the following keys:
                {{
                    "analysis": "brief explanation",
                    "answer": "text of the correct option",
                    "score": "confidence score"
                }}

                Task:
                User Query: {query}
                Options: {options}
                Facts: {facts}
                Output: 
                """
        }]

        for img_tab_path in img_tabs:
            # 读取图像文件并转换为base64编码
            with open(img_tab_path, "rb") as image_file:
                encoded_image = base64.b64encode(
                    image_file.read()).decode("utf-8")

            prompt.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
            })

        response = self._generate_response(prompt)
        logger.info("answer_with_context response: %s", response)

        return json.loads(response)

    def generate_text_questions(self, abstract: str) -> None:
        """
        调用 llm 基于 abstract 生成 3 个 text_qa
        """
        prompt = f"""
            Persona:
            You are a highly skilled biomedical text expert, specifically in the field of lung small cell cancer research. Your expertise lies in understanding scientific abstracts and formulating insightful multiple-choice questions based on their content. You are meticulous and pay close attention to detail, ensuring that all questions are directly answerable from the provided abstract and adhere to the specified output format.

            Instructions:
            Your task is to generate **three** multiple-choice questions based **solely** on the content of the provided scientific abstract about lung small cell cancer.
            For each question, you must:
            1. Formulate a clear and concise question that is directly relevant to the abstract's content and tests understanding of key information.
            2. Create four plausible options for each question. Only **one** of these options should be the correct answer based on the information presented in the abstract. The other three options should be distractors – plausible but incorrect based on the abstract.
            3. Identify the 'reference' for each question. This 'reference' should be the specific sentence or phrase **directly from the abstract** that contains the information needed to answer the question correctly.  If the answer is implicitly derived from multiple sentences, point to the sentences that collectively support the answer.
            4. Clearly indicate the 'correct_answer' for each question. This should be the option letter (e.g., "A", "B", "C", or "D") corresponding to the correct answer within the 'options' list.

            It is crucial that **all questions, options, references, and correct answers are derived exclusively from the provided abstract.** Do not introduce external knowledge or information. Focus on testing comprehension of the given abstract itself.

            OutputFormat:
            You must output your response in JSON format. The JSON should contain the following keys, structured for three questions:
            {{
                "question1": "...",
                "options1": ["A. ...", "B. ...", "C. ...", "D. ..."],
                "reference1": "...",
                "correct_answer1": "...",
                "question2": "...",
                "options2": ["A. ...", "B. ...", "C. ...", "D. ..."],
                "reference2": "...",
                "correct_answer2": "...",
                "question3": "...",
                "options3": ["A. ...", "B. ...", "C. ...", "D. ..."],
                "reference3": "...",
                "correct_answer3": "..."
            }}
            Each key should correspond to the question number (1, 2, or 3). 'question' should contain the question text, 'options' should be a list of four strings representing the options, 'reference' should be a string containing the sentence or phrase from the abstract as the reference, and 'correct_answer' should be a string indicating the letter of the correct option (e.g., "A").

            Example:
            Abstract: Small cell lung cancer (SCLC) is a highly aggressive malignancy characterized by rapid proliferation, early metastasis, and poor prognosis. Treatment options for SCLC are limited, and platinum-based chemotherapy remains the standard of care for extensive-stage disease. However, resistance to chemotherapy frequently develops, leading to treatment failure. Recent research has focused on identifying novel therapeutic targets and strategies to overcome chemoresistance in SCLC. Immunotherapy, particularly immune checkpoint inhibitors, has shown promising results in a subset of SCLC patients. Ongoing clinical trials are evaluating the efficacy of novel agents and combinations to improve outcomes for patients with SCLC.
            Output:
            {{
                "question1": "According to the abstract, what is a key characteristic of small cell lung cancer (SCLC)?",
                "options1": ["A. Slow proliferation rate", "B. Late metastasis", "C. Highly aggressive malignancy", "D. Favorable prognosis"],
                "reference1": "Small cell lung cancer (SCLC) is a highly aggressive malignancy characterized by rapid proliferation, early metastasis, and poor prognosis.",
                "correct_answer1": "C",
                "question2": "What is mentioned as the standard of care for extensive-stage small cell lung cancer (SCLC) in the abstract?",
                "options2": ["A. Immunotherapy", "B. Targeted therapy", "C. Platinum-based chemotherapy", "D. Surgery"],
                "reference2": "Treatment options for SCLC are limited, and platinum-based chemotherapy remains the standard of care for extensive-stage disease.",
                "correct_answer2": "C",
                "question3": "What therapeutic approach has shown 'promising results' in some small cell lung cancer (SCLC) patients, as mentioned in the abstract?",
                "options3": ["A. Platinum-based chemotherapy", "B. Immunotherapy with immune checkpoint inhibitors", "C. Targeted therapy", "D. Surgery"],
                "reference3": "Immunotherapy, particularly immune checkpoint inhibitors, has shown promising results in a subset of SCLC patients.",
                "correct_answer3": "B"
            }}

            Task:
            Abstract: {abstract}
            Output: 
        """
        response = self._generate_response(prompt, max_completion_tokens=2000)
        logger.info("generate_text_questions response: %s", response)

        return json.loads(response)

    def generate_img_questions(self, img_path: str,
                               img_caption: str) -> Dict[str, str]:
        """
        调用 llm 生成图片问题
        """
        prompt = [{
            "type":
                "text",
            "text":
                f"""
                Persona:
                You are an expert question generator that specializes in creating insightful and relevant questions based on images and their captions. Your goal is to assess understanding of visual content and its description through well-structured multiple-choice questions. You are meticulous, logical, and aim for clarity and accuracy in your questions and explanations.

                Instructions:
                1. Input Analysis: You will receive an image and its corresponding caption as input. Thoroughly analyze both the visual information suggested by the caption and the caption text itself to grasp the core content and context.
                2. Question Generation: Based on your understanding of the image and caption, generate one question that is relevant to both. The question should test the understanding of the combined information from the image and caption. The question should be clear, concise, and have only one definitively correct answer among the provided options.
                3. Option Creation: Develop four plausible answer options (A, B, C, D) for the generated question. Only one of these options should be the correct answer according to the image and caption. The other three options should be distractors – plausible but incorrect answers that someone who hasn't fully understood the image and caption might choose.
                4. Analysis Generation: Explain your thought process in generating the question and the options. This "analysis" should detail how you arrived at the question, why you chose these specific options, and how you identified the correct answer based on the image and caption. Essentially, walk through your reasoning step-by-step.
                5. Correct Answer Identification: Clearly identify the correct answer from the options (A, B, C, or D).

                OutputFormat:
                Present your output in JSON format, strictly adhering to the following structure:
                {{
                    "question": "...",
                    "options": {{
                        "A": "...",
                        "B": "...",
                        "C": "...",
                        "D": "..."
                    }},
                    "analysis": "...",
                    "correct_answer": "..."
                }}
                
                Example:
                Image Caption: "A golden retriever puppy enthusiastically chases a bright red ball in the green grass of a sunny park."
                Output:
                {{
                    "question": "According to the image and caption, what is the golden retriever puppy doing?",
                    "options": {{
                        "A": "Sleeping peacefully on a mat",
                        "B": "Eating food from a bowl",
                        "C": "Chasing a red ball",
                        "D": "Swimming in a lake"
                    }},
                    "analysis": "The caption explicitly states the puppy is 'chasing a bright red ball'. The image (as described by the caption) depicts a puppy in a park, actively engaged in playing fetch. Option A and B are unrelated to the action of chasing and the park setting. Option D, while plausible for a dog, is not mentioned in the caption and less directly related to 'ball' and 'park' in this context. Option C directly reflects the core action described in the caption.",
                    "correct_answer": "C"
                }}

                Task:
                Image Caption: {img_caption}
                Output:
                """
        }]

        with open(img_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        prompt.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}"
            }
        })

        response = self._generate_response(prompt, max_completion_tokens=2000)
        logger.info("generate_img_questions response: %s", response)
        return json.loads(response)

    def generate_text_questions_from_sections(
        self,
        section: str,
        content: str,
    ) -> List[Dict[str, str]]:
        """
        针对不同章节生成 qa 问题

        Args:
            section: 章节
            content: 章节内容
        
        Returns:
            一个 List[dict], dict 中包含 question, opitons, analysis 和 answer, score
        """
        logger.info("Generating questions for section: %s", section)

        if section == "methods":
            QUESTION_EXAMPLE = settings.METHODS_QUESTION_EXAMPLE
        elif section == "results":
            QUESTION_EXAMPLE = settings.RESULTS_QUESTION_EXAMPLE
        elif section == "discussion":
            QUESTION_EXAMPLE = settings.DISCUSSION_QUESTION_EXAMPLE
        else:
            raise ValueError("Invalid section provided.")

        responses = []
        for category, example in QUESTION_EXAMPLE.items():
            question_example = {}
            question_example[category] = example

            prompt = f"""
                Persona:
                Expert NSCLC researcher. Generate insightful, **stand-alone** multiple-choice questions for research paper sections.  Questions should be **professional and in-depth.**

                Instructions:
                Generate **one** multiple-choice question from:
                1. **Section Title** (e.g., METHODS, RESULTS, DISCUSSION).
                2. **Section Content** (NSCLC research paper section text).
                3. **Question Example** (JSON of example questions by section/type). Use examples for **style, depth, and focus**. **Do NOT copy examples.**
                Steps:
                1. **Context:** Understand section title (for question type guidance).
                2. **Analyze Content:** Read section content. Question MUST be **derived from** this content.
                3. **Question Example:** Use `question_example` for question type/complexity.
                4. **Generate Question:** Create 1 **stand-alone**, in-depth multiple-choice NSCLC question. Test understanding/critical thinking. **Avoid section-specific prefixes and context-dependent phrases.**
                5. **Options:** 4 plausible options. One correct. Three Distractors.
                6. **Analysis:** Explain rationale & relevance to content.
                7. **Answer:** State correct option.
                8. **Confidence Score:** 0.0-1.0 score (clarity, options, relevance).

                OutputFormat:
                You MUST output your response in JSON format, with the following keys:
                {{
                    "question": "...",
                    "options": {{
                        "A": "...",
                        "B": "...",
                        "C": "...",
                        "D": "..."
                    }},
                    "answer": "...",
                    "type": "...",
                    "analysis": "...",
                    "score": ...
                }}
                question: question text.
                options: options text.
                answer: correct option letter.
                type: question type (from example).
                analysis: question rationale & content relevance.
                score: 0.0-1.0 confidence.

                Example:
                Section Title: METHODS
                Section Content: Patients with stage IV NSCLC who progressed on first-line platinum-based chemotherapy were enrolled.  Tumor PD-L1 expression was assessed using immunohistochemistry (IHC) with the 22C3 antibody.  Next-generation sequencing (NGS) was performed on tumor tissue to identify EGFR mutations.
                Question Example: {{
                    "Methodological Details": "Could you elaborate on the specific inclusion and exclusion criteria used for patient selection in this study, and how these criteria might influence the generalizability of the findings to a broader NSCLC population?",
                    "Critical Evaluation": "Considering the study's methodology, what are the potential limitations or biases inherent in the chosen study design (e.g., retrospective analysis, single-arm trial), and how might these factors affect the interpretation of the results?",
                    "Clinical Application": "Based on the methodology described for biomarker assessment (e.g., liquid biopsy, IHC), how feasible and practical would it be to implement these techniques in routine clinical practice for NSCLC patient management?"
                }},
                Output:
                {{
                    "question": "What is a common method used to evaluate PD-L1 expression in tumor samples from NSCLC patients?",
                    "options": {{
                        "A": "Flow Cytometry",
                        "B": "Immunohistochemistry (IHC) with 22C3 antibody",
                        "C": "Enzyme-linked immunosorbent assay (ELISA)",
                        "D": "Western Blotting"
                    }},
                    "answer": "B",
                    "type": "Methodological Details",
                    "analysis": "This question focuses on the methodology of PD-L1 assessment in NSCLC.  While derived from the 'METHODS' section content describing IHC, the question is framed to be stand-alone, asking about a common method generally applicable in NSCLC research. The options and correct answer (IHC with 22C3 antibody) are directly informed by the example content, aligning with 'Methodological Details' for procedural understanding.",
                    "score": 0.9
                }}

                Task:
                Section Title: {section}
                Section Content: {content}
                Question Example: {question_example}
                Output:        
            """

            response = self._generate_response(prompt,
                                               max_completion_tokens=2000)
            logger.info("generate_text_questions response: %s", response)
            responses.append(json.loads(response))

        return responses

    def _generate_response(self, prompt: str, **args) -> str:
        params = {
            "model": "gpt-4o",
            "temperature": 0.5,
            "max_completion_tokens": 1000,
            "response_format": {
                "type": "json_object"
            },
        }
        params.update(args)
        logger.info("model name: %s", params["model"])

        completion = self.client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": prompt
            }],
            **params,
        )

        return completion.choices[0].message.content


def main():
    """
    主程序入口
    """
    pass


if __name__ == "__main__":
    main()

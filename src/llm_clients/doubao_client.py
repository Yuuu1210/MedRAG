"""
完成与豆包的交互
"""
import json

from volcenginesdkarkruntime import Ark

import settings
from src.utils import logger_config

logger = logger_config.get_logger(__name__)


class DoubaoClient():
    """
    构建豆包聊天模型
    """

    def __init__(self):
        self.client = Ark(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=settings.ARK_API_KEY,
        )

    def extract_biomedical_entities(self, text: str):
        """
        提取生物医学实体

        Args:
            text: 输入文本
        
        Returns:
            一个 dict, 包含 entities 和 score
        """
        system_prompt = """
            You are an advanced AI assistant specialized in biomedical natural language processing. Your primary task is to identify biomedical entities (such as diseases, genes, proteins, drugs, medical procedures, anatomical parts, etc.) from a given text.

            A critical part of your task is to expand any identified biomedical abbreviations to their full names using the provided context.
            - If the full name is already present alongside the abbreviation in the text (e.g., "Chronic Kidney Disease (CKD)"), prioritize using the full name.
            - If an abbreviation is used and its full form is commonly known or can be inferred from the broader biomedical context (even if not explicitly defined in the immediate input), you should expand it.
            - If an abbreviation cannot be confidently expanded or if its full form is highly ambiguous without further context not present in the input, you may return the abbreviation itself but should note this challenge in the 'analysis' field.

            The output MUST be a JSON object with the following exact structure:
            {
                "analysis": "A brief textual summary of the findings, including any challenges encountered (e.g., ambiguous abbreviations, successful expansions, or entities that could not be expanded).",
                "entities": [
                    "string_entity_1_full_name",
                    "string_entity_2_full_name",
                    // ... more entities
                ],
                "score": "A numerical value between 0.0 and 1.0 representing your overall confidence in the accuracy and completeness of the entity extraction and expansion for the given text."
            }

            Focus on accuracy and contextual understanding for abbreviation expansion. Only include distinct entities in the `entities` list, preferring full names.
        """
        user_prompt = f"""
            Please process the following biomedical text. Identify all relevant biomedical entities. If an entity is presented as an abbreviation, expand it to its full name based on the context within the text or common biomedical knowledge.

            Return the results strictly in the JSON format specified by your system instructions, including the 'analysis', 'entities' (as a list of full-name strings), and 'score' keys.

            Text: {text}
        """
        response = self._generate_response(system_prompt, user_prompt)
        # 去除转义字符, 防止 json.loads 报错
        response = response.replace("\\", "")
        # logger.info("extract_biomedical_entities response: %s", response)

        return json.loads(response)

    def qa_rewrite_pubmed_query(
        self,
        query: str,
        option_text: str,
        query_entity_to_mesh: dict,
        option_entity_to_mesh: dict,
    ):
        """
        (qa dataset) 重写 user query 为 pubmed_query
        """
        system_prompt = """
            You are an expert biomedical researcher tasked with converting multiple-choice question (MCQ) options into effective PubMed search queries. The goal is to generate queries that can verify the truthfulness of each option.

            You will receive:
            1.  The original MCQ query.
            2.  The specific option text.
            3.  Entity normalization results (MeSH terms).

            Your process:
            1.  Analyze Option: Identify the core assertion.
            2.  Utilize Entity Normalization:
                Prioritize MeSH terms from `Option Entity to MeSH` if they accurately represent key concepts.
                If an option's MeSH term is unsuitable (e.g., too broad/narrow, misaligned), you may adapt it, use a keyword from the option text itself.
                If no MeSH term is suitable for a key concept, use the most appropriate general biomedical keyword.
            3.  Formulate PubMed Query:
                Construct a `pubmed_query` as a list of 2 to 3 keywords.
                Keywords should be optimized for PubMed relevance and specificity.

            Output Format:
            {
                "analysis": Briefly explain your keyword choices, detailing how entity normalization was used or adapted. Justify any deviations from provided MeSH terms,
                "pubmed_query": The list of selected keywords,
                "score": A float (0.0-1.0) indicating your confidence in the query's effectiveness
            }

            Example:
            Query: "Which of the following is a primary mechanism of action for metformin in treating type 2 diabetes?",
            Option: "Metformin increases insulin secretion from pancreatic beta cells.",
            Query Entity to MeSH: {
                "metformin": "Metformin",
                "type 2 diabetes": "Diabetes Mellitus, Type 2"
            },
            Option to MeSH: {
                "Metformin": "Metformin",
                "insulin secretion": "Insulin Secretion",
                "pancreatic beta cells": "Insulin-Secreting Cells"
            }
            Output:
            {
                "analysis": "The option proposes that metformin's mechanism involves increasing insulin secretion from pancreatic beta cells. Key concepts are 'Metformin', 'insulin secretion', and 'pancreatic beta cells'. Entity normalization provided 'Metformin', 'Insulin Secretion', and 'Insulin-Secreting Cells' (for pancreatic beta cells). These MeSH terms are accurate and directly relevant for verifying this specific claim about metformin's action.",
                "pubmed_query": ["Metformin", "Insulin Secretion", "Insulin-Secreting Cells"],
                "score": 0.95
            }
        """
        user_prompt = f"""
            Query: {query}
            Option: {option_text}
            Query Entity to MeSH: {query_entity_to_mesh}
            Option Entity to MeSH: {option_entity_to_mesh}
        """
        response = self._generate_response(system_prompt, user_prompt)
        # 去除转义字符, 防止 json.loads 报错
        response = response.replace("\\", "")
        # logger.info("qa_rewrite_pubmed_query response: %s", response)

        return json.loads(response)

    def qa_rewrite_ddg_query(
        self,
        query: str,
        option: str,
        query_entities: list[str],
        option_entities: list[str],
    ):
        """
        (qa dataset) 重写 user query 为 duckduckgo_query

        Args:
            query: 用户输入的问题
            option_text: 选项文本
        
        Returns:
            一个 dict, 包含 analysis 和 query, score
        """
        system_prompt = """
            You are an expert DDG query rewriter for biomedical MCQs. Your goal is to create a DDG query to verify a given option.

            Inputs:
            1.  Original multiple-choice question.
            2.  A single option to verify.
            3.  Extracted biomedical entities (suggestions).

            Output: A single JSON object with:
            -   `analysis`: (string) Your reasoning for the `ddg_query`, including how you used or ignored the provided biomedical entities.
            -   `ddg_query`: (string) The rewritten query for DuckDuckGo, optimized for natural language and relevant keywords.
            -   `score`: (float, 0.0-1.0) Your confidence in the `ddg_query`'s effectiveness to find information for verifying the option.

            Entity Handling for `ddg_query`:
            From the provided biomedical entities, select the most relevant ones. Ignore noisy or irrelevant entities. The primary aim of the `ddg_query` is to help verify the specific claim made by the option.

            Rewriting Strategy for Negative Query (e.g., "Which is NOT TRUE...", "All are true EXCEPT..."):
            1.  Focus on Option's Claim: Identify the core assertion of the option you are rewriting for.
            2.  Create Direct Query: Formulate a DDG query to directly verify *that option's claim* within the question's context.
            3.  Avoid Double Negatives: Do NOT carry "NOT" or "EXCEPT" from the original question into the rewritten query.
            4.  Aim for Verifiability: The query should help find facts confirming or denying the option's statement.

            Example 1:
            Query: "What is the primary mechanism of action for metformin in the treatment of type 2 diabetes?"
            Option: "Decreasing hepatic glucose production"
            Extracted Query Entities: ["metformin","treatment","type 2 diabetes"]
            Extracted Option Entities: ["hepatic glucose production"]
            Output:
            {
                "analysis": "The question asks for metformin's mechanism in type 2 diabetes. The option claims it involves decreasing hepatic glucose production. The query combines 'metformin', 'hepatic glucose production', and 'type 2 diabetes' to verify this specific mechanism. 'mechanism of action' is implied. The query is keyword-focused for DDG.",
                "ddg_query": "metformin decrease hepatic glucose production type 2 diabetes mechanism",
                "score": 0.9
            }

            Example 2 (Negative Query):
            Query: "All of the following are known side effects of prolonged steroid use EXCEPT:"
            Option: "Osteoporosis"
            Extracted Query Entities: ["side effects","prolonged steroid use","steroid use","Osteoporosis"]
            Extracted Option Entities: ["Osteoporosis"]
            Output:
            {
                "analysis": "The original question is negative. The query focuses on verifying the option's claim directly: whether osteoporosis is a side effect of prolonged steroid use. Entities 'Osteoporosis' and 'prolonged steroid use' (or 'steroid use') are key. 'side effects' provides context. The query is phrased as a direct question suitable for DDG.",
                "ddg_query": "is osteoporosis a side effect of prolonged steroid use",
                "score": 0.95
            }
        """
        user_prompt = f"""
            Query: {query}
            Option: {option}
            Extracted Query Entities: {query_entities}
            Extracted Option Entities: {option_entities}
            Output: 
        """
        response = self._generate_response(system_prompt, user_prompt)
        # 去除转义字符, 防止 json.loads 报错
        response = response.replace("\\", "")
        # logger.info("qa_rewrite_ddg_query response: %s", response)

        return json.loads(response)

    def qa_rewrite_pubmed_query_v1(self, query: str, option_text: str):
        """
        (qa dataset) 重写 user query 为 pubmed_query

        Args:
            query: 用户输入的问题
            option_text: 选项文本
        
        Returns:
            一个 dict, 包含 analysis 和 keywords, score
        """
        prompt = f"""
            Role:
            You are a medical information specialist skilled in evidence-based search strategies, with expertise in:
            1. MeSH terminology mapping
            2. PubMed search syntax optimization
            3. Clinical negation handling
            4. Anatomical terminology validation

            Instructions:
            1. Analyze the user's query and provided option to identify core medical entities (e.g., diseases, drugs, symptoms).
            2. Rewrite the query into a PubMed-optimized search string using ≤3 keywords, prioritizing MeSH terms.
            3. Return a JSON with:
                analysis: Brief reasoning for keyword selection.
                keywords: list of keywords (max 3).
                score: Confidence score (0-1) based on relevance to the option.
            
            Output Format:
            {{  
                "analysis": "str",  
                "keywords": ["str", "str", "str"],  
                "score": float  
            }}

            Example:
            Query: Which medication is most effective for type 2 diabetes in elderly patients with cardiovascular risks?
            Option: Metformin
            Output:
            {{  
                "analysis": "Focused on 'Metformin' (drug entity) and 'Diabetes Mellitus, Type 2' (MeSH term). Added 'cardiovascular risks' as context.",  
                "keywords": ["Metformin", "Diabetes Mellitus, Type 2", "cardiovascular diseases"],  
                "score": 0.9  
            }}  

            Task:
            User Query: {query}
            Option: {option_text}
            Output: 
        """
        response = self._generate_response(prompt)
        # 去除转义字符, 防止 json.loads 报错
        response = response.replace("\\", "")
        # logger.info("qa_rewrite_pubmed_query response: %s", response)

        return json.loads(response)

    def fc_rewrite_pubmed_query(self, query: str):
        """
        (fact checking dataset) 将 fact checking 问题重写为 pubmed_query
        """
        prompt = f"""
            Role:
            You are a medical research assistant specializing in evidence-based literature review. You are adept at translating layperson queries into precise PubMed search strategies using Medical Subject Headings (MeSH terms) and technical medical terminology.

            Instructions:
            1. Analyze the user's fact-checking query to identify core medical concepts, diseases, interventions, or relationships.
            2. Reformulate the query into a PubMed-optimized query by prioritizing MeSH terms, clinical terminology, and Boolean logic.
            3. Extract **up to 3** keywords/phrases that precisely represent the query. Avoid overly broad terms (e.g., "cancer" → "colorectal neoplasms").
            4. Assign a confidence score between 0 and 1 reflecting your certainty that the query aligns with PubMed's biomedical scope, where:
                Closer to 0: Ambiguous/vague terms (e.g., "natural remedies for pain")
                Closer to 1: Specific biomedical concepts with clear MeSH mappings (e.g., "aspirin AND myocardial infarction prevention")

            OutputFormat:
            Return a JSON object with:
            {{  
                "analysis": "describe how you parsed the query and selected keywords",  
                "keywords": ["str", "str", "str"],  
                "score": 0.0-1.0
            }}

            Example:
            Query: "Does the COVID-19 vaccine cause heart inflammation in teenagers?"
            Output:
            {{  
                "analysis": "Focused on causal relationship between COVID-19 vaccines and myocarditis/pericarditis in adolescents. Prioritized MeSH terms for vaccine adverse effects and pediatric age groups.",  
                "keywords": ["COVID-19 vaccines/adverse effects", "Myocarditis/chemically induced", "Adolescent"],  
                "score": 0.9
            }}

            Task:
            Query: {query}
            Output: 
        """
        response = self._generate_response(prompt)
        # 去除转义字符, 防止 json.loads 报错
        response = response.replace("\\", "")
        # logger.info("fc_rewrite_pubmed_query response: %s", response)

        return json.loads(response)

    def qa_rewrite_ddg_query_v1(self, query: str, option_text: str):
        """
        重写 user query 为 duckduckgo_query

        Args:
            query: 用户输入的问题
            option_text: 选项文本

        Returns:
            一个 dict, 包含 analysis 和 query, score
        """
        prompt = f"""
            Role:
            You are a search expert skilled in crafting effective DuckDuckGo queries for medical topics, balancing natural language and keyword precision.

            Instructions:
            1. Identify core medical entities (e.g., diseases, drugs) in the user's query and provided option.
            2. Rewrite the query into a concise DuckDuckGo search string (1-2 phrases), prioritizing clarity and broad relevance (avoid advanced syntax like site:).
            3. Include synonyms or context to reduce ambiguity.

            Output Format:
            {{  
                "analysis": "str",  
                "query": "str",  
                "score": 0.0-1.0  
            }}  

            Example:
            Query: What's the first-line treatment for uncomplicated UTIs in non-pregnant adults?
            Option: Nitrofurantoin
            Output:
            {{  
                "analysis": "Focused on 'Nitrofurantoin' (drug) and 'uncomplicated UTI' (condition). Added 'first-line treatment' and 'non-pregnant adults' for context.",  
                "query": "Nitrofurantoin first-line treatment uncomplicated UTI non-pregnant adults",  
                "score": 0.95  
            }}  

            Task:
            User Query: {query}
            Option: {option_text}
            Output:
        """
        response = self._generate_response(prompt)
        # 去除转义字符, 防止 json.loads 报错
        response = response.replace("\\", "")
        # logger.info("qa_rewrite_ddg_query response: %s", response)

        return json.loads(response)

    def fc_rewrite_ddg_query(self, query: str):
        """
        对 fact checking 问题重写为 duckduckgo_query
        """
        prompt = f"""
            Role:
            You are an information verification specialist skilled in transforming fact-checking questions into optimized search queries for DuckDuckGo. You focus on clarity, source credibility, and efficient keyword extraction.

            Instructions:
            1. Analyze the user's query to identify core claims, entities, and context requiring verification.
            2. Reformulate the query into a concise DuckDuckGo search query:
                Use natural language phrases (e.g., does X cause Y?) without quotation marks or special operators.
                Replace technical terms with layperson synonyms (e.g., "myocardial infarction" → "heart attack").
                Keep the query under 8 words unless critical to the claim.
            3. Assign a confidence score between 0 and 1 to reflect your certainty in the query's effectiveness, based on:
                Specificity: Can the claim be tested with public data? (Higher specificity → closer to 1)
                Ambiguity: Are terms too vague to search effectively? (Higher ambiguity → closer to 0)
            
            OutputFormat:
            Return a JSON object with:
            {{
                "analysis": "explain how you simplified terms and structured the query",  
                "query": "clean search string without symbols or domain filters",
                "score": 0.0-1.0
            }}

            Example:
            Query: "Does the COVID-19 vaccine cause heart inflammation in teenagers?"
            Output:
            {{
                "analysis": "Targeted causal claim about COVID-19 vaccines and heart inflammation in adolescents. Simplified 'myocarditis' to public term 'heart inflammation' and removed medical jargon.",
                "query": "does covid vaccine cause heart inflammation in teens",
                "score": 0.75
            }}

            Task:
            Query: {query}
            Output: 
        """
        response = self._generate_response(prompt)
        # 去除转义字符, 防止 json.loads 报错
        response = response.replace("\\", "")
        # logger.info("fc_rewrite_ddg_query response: %s", response)

        return json.loads(response)

    def qa_is_paper_relevant(
        self,
        query: str,
        option: str,
        title: str,
        abstract: str,
    ):
        """
        根据论文 title, abstract 判断是否与 query, option 相关
        """
        system_prompt = """
            Role
            You are an academic research assistant with expertise in medical analysis. Your task is to objectively evaluate whether a given paper provides meaningful evidence or insights to address a specific multiple-choice query.

            Instructions
            1. Understand the Query and Options:
                Carefully read the user-provided query and its 4 options. Identify the core query and the key concepts each option represents.
            2. Analyze the Paper:
                Read the provided paper title and abstract. Extract the main research goals, methods, findings, and conclusions.
                Focus on explicit or implicit connections to the query's topic, terminology, or options.
            3. Determine Relevance:
                If the paper directly addresses the query's topic, supports/refutes an option, or provides critical context/theory/data, mark "relevant": true.
                If the paper's content is unrelated, too vague, or lacks actionable insights for the query, mark "relevant": false.
            4. Assign Confidence Score:
                Score ranges from 0 (lowest) to 1 (highest). Base this on the strength of the paper's alignment with the query (e.g., explicit mentions = higher score; tangential links = lower score).
            
            Output Format
            {  
                "analysis": "string // Concise rationale for your judgment (e.g., The paper explicitly states that X causes Y, directly supporting Option C)", 
                "relevant": "boolean // true if the content is helpful for evaluating the option, false otherwise",  
                "score": "float // Confidence score between 0 and 1"  
            }

            Example
            Query: "What is the primary driver of antibiotic resistance in bacteria?"
            Option: "Overuse in agriculture"
            Paper Title: "Agricultural Antibiotic Use and Resistance Gene Proliferation in Soil Microbiomes"
            Paper Abstract: "This study demonstrates that tetracycline application in livestock farming increases antibiotic-resistance gene transfer rates among soil bacteria by 300%..."
            Output:
            {  
                "analysis": "The paper directly links agricultural antibiotic use (Option) to increased resistance gene transfer in bacteria, providing experimental evidence. It strongly supports Option.",  
                "relevant": true,  
                "score": 0.9  
            }
        """
        user_prompt = f"""
            Query: {query}
            Option: {option}
            Paper Title: {title}
            Paper Abstract: {abstract}
            Output: 
        """
        response = self._generate_response(system_prompt, user_prompt)
        # 去除转义字符, 防止 json.loads 报错
        response = response.replace("\\", "")
        # logger.info("is_paper_relevant response: %s", response)

        return json.loads(response)

    def fc_is_paper_relevant(self, query: str, title: str, abstract: str):
        """
        fact checking 数据集使用, 根据论文 title, abstract 判断是否与 query 相关
        """
        prompt = f"""
            Role
            You are an academic research assistant with expertise in medical analysis. Your task is to objectively evaluate whether a given paper provides meaningful evidence or insights to address the specific research query in the query.

            Instructions
            1. Understand the Query:
                Carefully analyze the user-provided query to identify its core research query and key scientific concepts.
                Extract explicit or implicit requirements for evidence types (e.g., causal mechanisms, clinical outcomes, biological processes).
            2. Analyze the Paper:
                Read the provided paper title and abstract systematically.
                Identify:  
                a) Research objectives aligned with the query's focus  
                b) Methodology relevant to answering the query  
                c) Findings directly addressing the query's core mechanisms  
                d) Conclusions explicitly linking to the query's domain
            3. Determine Relevance:
                Mark "relevant": true ONLY IF:
                The paper _directly investigates_ the phenomenon/relationship stated in the query
                Provides _empirical evidence_ (clinical, experimental, or observational) about the query's subject
                Establishes _theoretical frameworks_ critical for interpreting the query's core concepts
                Mark "relevant": false IF:
                Focuses on tangential phenomena
                Uses methodologies incompatible with the query's required evidence tier
                Discusses broader/narrower scope than the query's specific focus
            4. Assign Confidence Score:
                0.8-1.0: Explicitly tests/evaluates the exact relationship in the query (e.g., RCT on the query's intervention-outcome pair)
                0.5-0.7: Examines closely related mechanisms or provides secondary evidence (e.g., animal models of the query's disease process)
                0.2-0.4: Mentions key terms without substantive investigation
                0.0-0.1: No conceptual/methodological overlap

            Output Format
            {{
                "analysis": "string // Concise rationale (e.g., 'The paper experimentally verifies X-induced Y pathogenesis through knockout models')",
                "relevant": boolean,
                "score": float
            }}

            Example
            Query: Does long-term proton pump inhibitor use increase gastric cancer risk?
            Paper Title: A 10-Year Cohort Study on Acid Suppression Therapy and Gastrointestinal Malignancies
            Paper Abstract: Longitudinal analysis of 45,000 patients showed prolonged PPI usage (>5 years) correlated with 2.3-fold higher gastric adenocarcinoma incidence (95% CI 1.7-3.1) after adjusting for H. pylori status...
            Output:
            {{
                "analysis": "The paper directly examines the causal relationship in the query through longitudinal clinical data, providing adjusted risk estimates specifically for gastric cancer.",
                "relevant": true,
                "score": 0.95
            }}

            Task
            Query: {query}
            Paper Title: {title}
            Paper Abstract: {abstract}
            Output: 
        """
        response = self._generate_response(prompt)
        # 去除转义字符, 防止 json.loads 报错
        response = response.replace("\\", "")
        # logger.info("fc_is_paper_relevant response: %s", response)

        return json.loads(response)

    def qa_is_website_relevant(
        self,
        query: str,
        option: str,
        title: str,
        content: str,
    ):
        """
        根据网站 title, content 判断是否与 query, option 相关
        """
        system_prompt = """
            Role
            You are a Search Relevance Analyst specializing in evaluating the utility of web content for answering specific questions. Your task is to critically assess whether a given webpage's title and content provide sufficient information to support or refute a specific option in a multiple-choice query.

            Instructions
            1. Understand the Query and Option:
                Read the user's query (with four options) and the single provided option they want to verify.
                Identify the key claims, keywords, or reasoning required to validate the option.
            2. Analyze Webpage Content:
                Review the title and content of the webpage retrieved from DuckDuckGo.
                Check if the content directly confirms, contradicts, or is irrelevant to the option's validity.
            3. Determine Relevance:
                true: The content explicitly supports or refutes the option (e.g., provides evidence, statistics, or logical reasoning).
                false: The content is unrelated, ambiguous, or lacks actionable information for evaluating the option.
            4. Assign Confidence Score:
                Score (0-1): Reflect how certain you are in your judgment (e.g., 0.9 for strong evidence, 0.5 for partial relevance).

            OutputFormat
            Return a JSON object with:
            {  
                "analysis": "string // Concise rationale for your judgment (e.g., The webpage explicitly states that X causes Y, directly supporting Option C)",  
                "relevant": "boolean // true if the content is helpful for evaluating the option, false otherwise",  
                "score": "float // Confidence score between 0 and 1"  
            }

            Example
            Query: "What causes solar eclipses?"
            Option: "The Moon blocks sunlight from reaching Earth."
            Webpage Title: "NASA Solar Eclipse Guide"
            Webpage Content: "A solar eclipse occurs when the Moon passes between the Sun and Earth, casting a shadow on Earth."
            Output:
            {  
                "analysis": "The webpage explicitly states that the Moon's position between the Sun and Earth causes solar eclipses, confirming Option C.",  
                "relevant": true,  
                "score": 0.95  
            }
        """
        user_prompt = f"""
            Query: {query}
            Option: {option}
            Webpage Title: {title}
            Webpage Content: {content}
            Output: 
        """
        response = self._generate_response(system_prompt, user_prompt)
        # 去除转义字符, 防止 json.loads 报错
        response = response.replace("\\", "")
        # logger.info("is_website_relevant response: %s", response)

        return json.loads(response)

    def fc_is_website_relevant(self, query: str, title: str, content: str):
        """
        fact checking 数据集使用, 根据网站 title, content 判断是否与 query 相关
        """
        prompt = f"""
            Role  
            Search Relevance Analyst evaluating webpage utility for answering queries

            Instructions  
            1. Analyze Query:  
            - Identify core query and key information requirements from the query  
            2. Assess Webpage:  
            - Check if title/content directly addresses the query's central query  
            - Verify whether content provides conclusive evidence (support/refute) or is irrelevant  
            3. Relevance Criteria:  
            true: Content explicitly answers query with evidence/data/logical reasoning  
            false: Content lacks focus, shows partial/incomplete coverage, or discusses unrelated topics  
            4. Confidence Scoring:  
            0.9-1.0: Comprehensive direct answer (e.g., official data)  
            0.6-0.8: Partial/indirect answer (e.g., related studies)  
            0.3-0.5: Surface-level mention  
            0-0.2: No meaningful connection  

            Output Format  
            {{  
                "analysis": string // 20-30 character conclusion (e.g., Provides clinical evidence about X effects),  
                "relevant": boolean,  
                "score": float  
            }}  

            Example  
            Query: "Does regular aspirin use reduce colon cancer risk?"  
            Webpage Title: "NIH Study on Aspirin and Colorectal Cancer"  
            Webpage Content: "A 10-year cohort study of 50,000 adults showed daily low-dose aspirin users had 40% lower incidence of colorectal cancer (HR 0.60, 95% CI 0.52-0.69)."  
            Output:  
            {{  
                "analysis": Provides longitudinal clinical trial data with risk ratios,  
                "relevant": true,  
                "score": 0.97  
            }}

            Task:  
            Query: {query}  
            Webpage Title: {title}  
            Webpage Content: {content}  
            Output:
        """
        response = self._generate_response(prompt)
        # 去除转义字符, 防止 json.loads 报错
        response = response.replace("\\", "")
        # logger.info("fc_is_website_relevant response: %s", response)

        return json.loads(response)

    def qa_answer_query(self, query: str, options: dict[str, str]):
        """
        llm 直接回答问题

        Args:
            query: 问题
            options: 选项
        
        Returns:
            一个 dict, 包含 analysis 和 answer, score
        """
        prompt = f"""
            Role:
            You are a knowledgeable medical expert AI assistant capable of answering medical questions based on your vast knowledge base.

            Instructions:
            1. You will be given a medical query and four options labeled A, B, C, and D.
            2. Carefully read the query and all the options.
            3. Based on your medical knowledge, determine the correct answer to the query.

            OutputFormat:
            Return your response in JSON format with the following keys:
            {{
                "analysis": "brief explanation",
                "answer": "letter of the correct option",
                "score": "confidence score"
            }}

            Example:
            Query: Which of the following is a common symptom of influenza?
            Options: {{
                "A": "Persistent high fever, body aches, and fatigue.",
                "B": "Localized skin rash with intense itching.",
                "C": "Sudden sharp pain in the lower abdomen.",
                "D": "Gradual onset of memory loss and confusion."
            }}
            Output:
            {{
                "analysis": "Influenza, commonly known as the flu, is characterized by systemic symptoms such as fever, body aches, and fatigue. Options B, C, and D describe symptoms more indicative of other conditions (dermatitis, appendicitis, and dementia, respectively).",
                "answer": "A",
                "score": 0.95
            }}

            Task:
            Query: {query}
            Options: {options}
            Output: 
        """
        response = self._generate_response(prompt)
        # 去除转义字符, 防止 json.loads 报错
        response = response.replace("\\", "")
        logger.info("answer_qa_directly response: %s", response)

        return json.loads(response)

    def fc_answer_query(self, query: str):
        """
        llm 直接回答问题

        Args:
            query: 问题
        
        Returns:
            一个 dict, 包含 analysis 和 answer, score
        """
        prompt = f"""
            Role:
            You are a Fact-Checking Analyst specializing in the biomedical domain.

            Instructions:
            1.  Evaluate the claim within the `Query` based on your internal knowledge of biology and medicine.
            2.  Determine if the claim is `Supported` (0), `Refuted` (2), or if there is `Not enough information` (1) to make a definitive judgment based on established scientific understanding.
            3.  Provide a concise `analysis` explaining your reasoning, referencing relevant biological or medical principles that lead to your conclusion.
            4.  Assign a confidence `score` (a float between 0.0 and 1.0) indicating your certainty in the assigned answer based on the strength of scientific evidence.
            5.  Respond ONLY with a JSON object strictly following the `OutputFormat` specified below.

            Answer Definitions:
            `0: Supported` - Established scientific evidence clearly supports the claim.
            `1: Not enough information` - Evidence is insufficient, conflicting, or the topic is outside established scientific consensus, preventing a definitive judgment.
            `2: Refuted` - Established scientific evidence clearly contradicts the claim.

            OutputFormat:
            Required Output Format (JSON):
            {{
                "analysis": "Your reasoning and evidence assessment based on biomedical knowledge.",
                "answer": <Integer: 0, 1, or 2>,
                "score": <Float: 0.0 to 1.0>
            }}


            Example:
            Query: Can antibiotics be effectively used to treat infections caused by the influenza virus?
            Output:
            {{
                "analysis": "The claim is that antibiotics are effective against influenza virus infections. Influenza is caused by a virus. Antibiotics are antimicrobial drugs designed specifically to target bacteria, typically by disrupting their cell walls, protein synthesis, or DNA replication. Viruses have fundamentally different structures and replication mechanisms (they replicate inside host cells) and lack the specific targets that antibiotics act upon. Therefore, antibiotics are not effective against viral infections like influenza. Standard antiviral medications are used for influenza treatment. The claim is clearly refuted by basic principles of microbiology and pharmacology.",
                "answer": 2,
                "score": 1.0
            }}

            Task:
            Query: {query}
            Output: 
        """
        response = self._generate_response(prompt)
        # 去除转义字符, 防止 json.loads 报错
        response = response.replace("\\", "")
        logger.info("answer_fc_directly response: %s", response)

        return json.loads(response)

    def qa_answer_query_rag(
        self,
        query: str,
        options: dict[str, str],
        context: list[str],
    ):
        """
        使用 rag 方法回答问题

        Args:
            query: 问题
            options: 选项
            context: rag 方法获取的相关文本
        
        Returns:
            一个 dict, 包含 analysis 和 answer, score
        """
        prompt = f"""
        Role
        You are an analytical assistant trained to solve multiple-choice questions by synthesizing retrieved information and internal knowledge. You carefully evaluate the credibility of external context, cross-reference it with your own expertise, and make informed decisions even when conflicting data arises.

        Instructions
        1. Input Components:
            Receive query, options, and context.
            If context is empty/non-relevant, rely on internal knowledge.
        2. Conflict Handling:
            If retrieved text conflicts with your internal knowledge:
                Assess the reliability of the external context (e.g., factual consistency, source quality).
                Choose the most plausible answer by prioritizing either external context or internal knowledge.
        
        Output Format
        Provide a JSON with keys: analysis, answer, source, score.
        analysis: Detailed reasoning about why the answer was chosen, including contradictions or uncertainties.
        source: "external" if context is used, "internal" otherwise.
        score: Confidence score (0-1) based on clarity of evidence and consistency of reasoning. 

        Example
        Query: "What causes tides?"
        Options: {{"A": "Moon's gravity", "B": "Earth's rotation", "C": "Solar winds", "D": "Atmospheric pressure"}}
        Context: ["Tides are primarily caused by the gravitational pull of the Moon and the Sun."]
        Output:
        {{  
            "analysis": "The context states that tides result from the Moon's and Sun's gravity. My internal knowledge confirms that the Moon's gravitational pull is the dominant factor, aligning with option A. The Sun's role is secondary but not listed, so no conflict arises.",  
            "answer": "A",  
            "source": "external",  
            "score": 0.95  
        }}  

        Task:
        Query: {query}
        Options: {options}
        Context: {context}
        Output:
        """
        response = self._generate_response(prompt)
        # 去除转义字符, 防止 json.loads 报错
        response = response.replace("\\", "")
        # logger.info("answer_rag response: %s", response)

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.info("error: %s", e)
            return {}

    def fc_answer_query_rag(self, query: str, context: list[str]):
        """
        使用 rag 方法回答问题

        Args:
            query: 问题
            context: rag 方法获取的相关文本
        
        Returns:
            一个 dict, 包含 analysis 和 answer, score
        """
        prompt = f"""
            Role:
            You are an AI Fact-Checker.

            Instructions:
            1.  Evaluate: Analyze the `Query` against the `Context` (if provided). If `Context` is absent or insufficient, use internal knowledge. Determine if the `Query` is: `0` (Supported), `1` (Not Enough Information), or `2` (Refuted).
            2.  Explain: Write a brief `analysis` justifying your evaluation.
            3.  Source: Set `source` to `"external"` if the `Context` was primarily used, `"internal"` otherwise.
            4.  Confidence: Provide a `score` (float, 0.0-1.0) indicating your certainty.

            Output Format:
            Respond *only* with a single JSON object:
            {{
                "analysis": "Your concise reasoning based on the evaluation.",
                "answer": "[0, 1, or 2]",
                "source": "["external" or "internal"]",
                "score": "[float between 0.0 and 1.0]"
            }}

            Example:
            Query: mRNA vaccines work by introducing a weakened or inactivated virus into the body.
            Context: ["Unlike traditional vaccines which may use weakened or inactivated viruses, mRNA vaccines utilize messenger RNA (mRNA) sequences. These sequences instruct the body's cells to produce a specific antigen (like the spike protein of a virus), triggering an immune response without introducing the actual virus."]
            Output:
            {{
                "analysis": "The context explicitly contrasts mRNA vaccines with traditional vaccines, stating mRNA vaccines use mRNA sequences to instruct cell protein production, rather than introducing weakened or inactivated viruses. This directly refutes the claim.",
                "answer": 2,
                "source": "external",
                "score": 1.0
            }}

            Task:
            Query: {query}
            Context: {context}
            Output:
        """
        response = self._generate_response(prompt)
        # 去除转义字符, 防止 json.loads 报错
        response = response.replace("\\", "")
        logger.info("answer_fc_rag response: %s", response)

        return json.loads(response)

    def qa_select_en_textbooks(self, query: str, options: dict[str, str]):
        """
        (qa dataset) 选择 query 相关的 en 教科书
        """
        textbooks = {
            item["id"]: item["category"] for item in settings.EN_TEXTBOOKS
        }

        prompt = f"""
            Role
            You are a medical education specialist with expertise in selecting appropriate textbooks for specific topics. Your role is to identify the 1 to 3 most relevant textbooks based on the content of a given query and the provided multiple-choice options.

            Instructions
            1. Read the query and the four options carefully to understand the core concepts and topics involved.
            2. Select the **1 to 3 most relevant** textbooks that cover the subject matter relevant to the query and options. Prioritize categories directly addressing the core concepts.
            3. Provide a brief explanation in the 'analysis' field detailing your reasoning. This should include: 1) Key concepts identified from the query/options. 2) Justification for selecting *each* chosen textbook category. 3) Rationale for excluding other potentially relevant categories or stopping the selection.

            OutputFormat
            Return a JSON object with the following keys:
            {{
                "analysis": "<Three-part reasoning: 1) Key concept extraction 2) Rationale for selecting each chosen discipline 3) Rationale for exclusion/stopping>",
                "textbooks": ["<Category Name>", ...],  // 1 to 3 items
                "score": <0.0-1.0> // Confidence score for the relevance of the selected textbooks
            }}

            Example
            Query: "A 65-year-old man with a history of smoking presents with chest pain radiating to the left arm, diaphoresis, and ECG changes showing ST elevation. What is the most likely diagnosis and initial management step involving medication?"
            Options: {{
                "A": "Pulmonary embolism, administer heparin",
                "B": "Myocardial infarction, administer aspirin and nitroglycerin",
                "C": "Aortic dissection, control blood pressure",
                "D": "Pericarditis, administer NSAIDs"
            }}
            Textbooks: {textbooks}
            Output:
            {{
                "analysis": "1) Core concepts: Acute chest pain, ECG ST elevation, smoking history point to cardiovascular pathology, likely acute myocardial infarction (MI), asking for diagnosis and initial drug management. 2) Requires understanding of MI pathophysiology and clinical presentation (Pathophysiology, Internal Medicine) and the drugs used (Pharmacology). Internal Medicine covers the clinical context and diagnosis broadly. Pathophysiology explains the underlying disease process. Pharmacology details the drug actions (aspirin, nitroglycerin). 3) Excluded Surgery as initial management is medical; excluded basic sciences like Anatomy/Physiology as the focus is clinical pathology and treatment.",
                "textbooks": ["Internal Medicine", "Pathophysiology", "Pharmacology"],
                "score": 0.9
            }}

            Task:
            Query: {query}
            Options: {options}
            Textbooks: {textbooks}
            Output:
        """
        response = self._generate_response(prompt)
        # 去除转义字符, 防止 json.loads 报错
        response = response.replace("\\", "")
        logger.info("qa_select_en_textbooks response: %s", response)

        return json.loads(response)

    def fc_select_en_textbooks(self, query: str):
        """
        fact checking 数据集使用, 选择 query 相关的 en 教科书
        """
        textbooks = {
            item["id"]: item["category"] for item in settings.EN_TEXTBOOKS
        }

        prompt = f"""
            Role:
            You are a medical education specialist with expertise in selecting appropriate textbooks for specific topics. Your role is to identify the most relevant textbooks based solely on the content of a given clinical query.

            Instructions:
            1. Analyze the query to identify key medical concepts, disease entities, and anatomical structures
            2. Determine the primary discipline(s) required to address the core knowledge in the query
            3. Select ONE most relevant textbook category from the provided list
            4. Provide reasoning including:
            - Key concept extraction from query
            - Discipline matching logic
            - Exclusion rationale for other categories

            Output Format:
            {{
                "analysis": "<Three-part reasoning>",
                "textbooks": ["<Textbook Category>"],  // Strictly 1 item
                "score": <0.0-1.0 confidence score>
            }}

            Example:
            Query: "A 58-year-old male with chronic hypertension presents with sudden hematuria..."
            Textbooks: {textbooks}
            Output:
            {{
                "analysis": "1) Core concepts: hypertension complications, hematuria pathophysiology 2) Requires understanding of end-organ damage mechanisms (Pathophysiology) 3) Excluded Anatomy as query focuses on disease mechanism not structure",
                "textbooks": ["Pathophysiology"],
                "score": 0.85
            }}

            Task:
            Query: {query}
            Textbooks: {textbooks}
            Output: 
        """
        response = self._generate_response(prompt)
        # 去除转义字符, 防止 json.loads 报错
        response = response.replace("\\", "")
        # logger.info("fc_select_en_textbooks response: %s", response)

        return json.loads(response)

    def summarize_chunks(
        self,
        query: str,
        options: dict[str, str],
        chunks_and_source: list[dict[str, str]],
    ):
        """
        总结 chunks
        """
        system_prompt = """
            Role:
            Expert text analyst extracting key information to answer questions.

            Instructions:
            1. Input: You receive a query, 4 options and a chunk list (dictionaries with "chunk" and "source").
            2. Relevance Analysis: Read the query, options and chunks. Identify chunks relevant to answering the query and selecting the correct option.
            3. Summarize & Source: Summarize key information from relevant chunks. Append the source in parentheses for each summarized point. Combine summaries into a single string.
            4. Analysis & Confidence: Explain your reasoning: how the summarized chunks help answer the query and choose the correct option. Provide a confidence score (0-1) on the summary's helpfulness.

            OutputFormat:
            Don't include DOUBLE QUOTES in the values.
            Don't include DOUBLE QUOTES in the values.
            Don't include DOUBLE QUOTES in the values.
            Respond with a single JSON object:
            {
                "analysis": "...",
                "summary": "...",
                "score": "..."
            }

            Example:
            Query: "What is mitochondria's main function?"
            Options: {
                "A": "Protein Synthesis",
                "B": "Energy Production",
                "C": "DNA Replication",
                "D": "Waste Removal"
            }
            Chunks: [
                {"chunk": "Mitochondria are cell 'powerhouses'.", "source": "Biology.txt"},
                {"chunk": "They generate ATP for energy.", "source": "Cell_Journal.pdf"},
                {"chunk": "Ribosomes make proteins.", "source": "Bio_Notes.pdf"},
                {"chunk": "Nucleus replicates DNA.", "source": "Genetics.txt"}
            ]
            Output:
            {
                "analysis": "Sentences indicate mitochondria are for energy production. Sentence 1 calls them powerhouses. Sentence 2 says they make ATP (energy). Sentences 3 & 4 eliminate options A & C.",
                "summary": "Mitochondria are cell powerhouses (Biology.txt). They generate ATP for energy (Cell_Journal.pdf).",
                "score": 0.9
            }
        """
        user_prompt = f"""
            Query: {query}
            Options: {options}
            Chunks: {chunks_and_source}
            Output:
        """
        response = self._generate_response(system_prompt, user_prompt)
        # 去除转义字符, 防止 json.loads 报错
        response = response.replace("\\", "")
        # logger.info("summarize_chunks response: %s", response)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON response: %s", response)
            return {"analysis": "", "summary": "", "score": 0.0}

    def classify_ddg_query_category(
        self,
        query: str,
        option_text: str,
        ddg_query: str,
    ):
        """
        调用 llm 分类 ddg query 的类型
        """
        prompt = f"""
            Role:
            You are an expert evaluator for Biomedical RAG query rewriting. Your task is to analyze a `rewritten_ddg_query` based on an `original_query` (multiple-choice) and the specific `option_text` it aims to verify for DuckDuckGo search.

            Instruction:
            Given `original_query`, `option_text`, and `rewritten_ddg_query`, classify the `rewritten_ddg_query` into one of the categories below. Provide your analysis and a confidence score.

            Categories & Labels:
            1.  `Inadequate Negation Handling` (1): Fails to correctly process "not true," "except," etc. in the original query for option verification.
            2.  `Key Information Omission from Option` (2): Misses or inaccurately incorporates crucial details from the `option_text` into the query.
            3.  `Over-reliance on Keyword Stuffing` (3): Query is a mere concatenation of keywords, lacking natural language structure for effective search.
            4.  `Poor Handling of Question Semantics` (4): Shows insufficient understanding of the original question's type (e.g., definition, cause-effect) leading to a suboptimal search query.
            5.  `Misapplication to Non-Searchable Questions` (5): Attempts to generate a search query for questions requiring calculation, ordering, or complex logic not directly answerable by web search.
            6.  `Inadequate Handling of Implicit Visual Context` (6): Fails to effectively create a searchable query when the original question implies unprovided visual information (e.g., "marked with arrow").
            7.  `Suboptimal Strategy for Meta-Options` (7): Generates a too generic query for meta-options like "All of the above" or "None of the above," which doesn't help validate the option.
            8.  `Appropriate Query` (8): Accurately captures intent, includes necessary information, and is well-phrased for effective retrieval.
            9.  `Other Issues` (9): Exhibits problems not covered by other categories or issues are minor/unique.

            OutputFormat:
            Respond with a single JSON object:
            {{
                "analysis": "Briefly justify your classification, focusing on the query's effectiveness for retrieval and option verification.",
                "label": "Integer label from 1 to 9",
                "score": "Confidence score (0.0-1.0)"
            }}

            Task:
            Query: {query}
            Option Text: {option_text}
            Pubmed Query: {ddg_query}
            Output: 
        """
        response = self._generate_response(prompt)
        # 去除转义字符, 防止 json.loads 报错
        response = response.replace("\\", "")
        logger.info("classify_pubmed_query_category response: %s", response)

        return json.loads(response)

    def classify_empty_context_query_category(
        self,
        query: str,
        options: dict,
    ):
        """
        调用 llm 分类 ddg query 的类型
        """
        prompt = f"""
            You will be provided with a biomedical question consisting of a "query" and "options".
            Your task is to analyze the question and classify it into ONE of the following categories based SOLELY on the content of the query and options. Assume there is no external context available.

            Categories:
            1.  Basic Medical Sciences (Non-Clinical):
                *   Description: Tests fundamental knowledge of normal human body structure (anatomy), function (physiology), drug actions/properties (pharmacology), basic disease mechanisms (pathology), microorganism characteristics (microbiology), or biochemical processes, not tied to a specific patient scenario.
                *   Keywords: "structure of...", "function of...", "mechanism of...", "drug of choice for [general condition, not specific case]", "enzyme involved in...", "produced by...".
            2.  Clinical Scenarios & Management:
                *   Description: Presents a hypothetical patient case or clinical situation (even if brief) and asks for a diagnosis, next step in management, interpretation of findings, or understanding of disease features in a patient context.
                *   Keywords: "patient presents with...", "next step in management...", "most likely diagnosis...", "features of [disease]...", "complication of...".
            3.  Dental Sciences, Materials & Procedures:
                *   Description: Focuses specifically on dental anatomy, oral pathology, properties/composition/use of dental materials (e.g., amalgam, composites, impression materials, cements, orthodontic wires), or common dental/orthodontic/endodontic procedures and instruments.
                *   Keywords: "cusp...", "amalgam...", "composite...", "impression material...", "root canal...", "gingivitis...", "orthodontic wire...", "denture...".
            4.  Definitions, Classifications, Indices & Terminology:
                *   Description: Asks for the definition of a specific medical or dental term, the components/criteria of a named classification system or index (e.g., WHO classification, Miller's, CPITN, Dean's Fluorosis Index), or the meaning of a specific rule or sign.
                *   Keywords: "is defined as...", "stands for...", "components of...", "classification of...", "index...", "rule...", "syndrome [when asking about the syndrome itself]".
            5.  Epidemiology, Biostatistics & Public Health:
                *   Description: Relates to population health metrics (e.g., mortality rates, incidence, prevalence), statistical concepts/tests, study design principles, types of bias, or public health programs, data sources, and screening principles.
                *   Keywords: "mortality rate...", "sample size...", "type of error...", "odds ratio...", "screening test...", "WHO program...".
            6.  Legal, Ethical & Forensic Principles:
                *   Description: Pertains to aspects of medical law (e.g., specific sections of penal codes), ethical doctrines in healthcare, or principles of forensic science.
                *   Keywords: "IPC section...", "CrPC section...", "doctrine of...", "bullet fingerprinting...".
            7.  Others

            Based on your analysis of the query and options, provide your response in the following JSON format:
            {{
                "analysis": "Provide a brief step-by-step reasoning for your classification decision. Explain which keywords or aspects of the query/options led you to choose the specific category. If the question seems to fit multiple categories, explain why you prioritized the chosen one.",
                "label": <integer_representing_the_chosen_category_number_from_1_to_6>,
                "score": <float_between_0.0_and_1.0_indicating_your_confidence_in_the_classification_e.g.,_0.95>
            }}

            Task:
            Query: {query}
            Options: {options}
            Output: 
        """
        response = self._generate_response(prompt)
        # 去除转义字符, 防止 json.loads 报错
        response = response.replace("\\", "")
        # logger.info("classify_empty_context_query_category response: %s",
        #             response)

        return json.loads(response)

    def _generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        **args,
    ) -> str:
        params = {
            "model": settings.ARK_ENDPOINT_ID2,
            "temperature": 0.3,
            "max_tokens": 1000,
        }
        params.update(args)

        completion = self.client.chat.completions.create(
            messages=[{
                "role": "system",
                "content": system_prompt
            }, {
                "role": "user",
                "content": user_prompt
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

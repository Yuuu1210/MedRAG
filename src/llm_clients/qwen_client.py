"""
定义 QwenClient 类
"""
import json

from openai import OpenAI

import settings
from src.utils import logger_config

logger = logger_config.get_logger(__name__)


class QwenClient():
    """
    与 Qwen 聊天模型交互
    """

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def qa_answer_query(self, question: str, options: dict[str, str]):
        """
        llm 直接回答问题

        Args:
            question: 问题
            options: 选项
        
        Returns:
            一个 dict, 包含 analysis 和 answer, score
        """
        prompt = f"""
            Role:
            You are a knowledgeable medical expert AI assistant capable of answering medical questions based on your vast knowledge base.

            Instructions:
            1. You will be given a medical question and four options labeled A, B, C, and D.
            2. Carefully read the question and all the options.
            3. Based on your medical knowledge, determine the correct answer to the question.

            OutputFormat:
            Return a SINGLE JSON object with the following keys:
            analysis: A brief explanation of why the correct option is the best answer.
            answer: The letter of the correct option (A, B, C, or D).
            score: Your confidence score in the answer (0-1).

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
                "answer": "A",
                "score": 0.95
            }}

            Task:
            Question: {question}
            Options: {options}
            Output: 
        """
        response = self._generate_response(prompt)
        # logger.info("answer_qa_directly response: %s", response)

        response_type = type(json.loads(response))
        if response_type == dict:
            return json.loads(response)
        elif response_type == list:
            return json.loads(response)[0]

    def fc_answer_query(self, question: str):
        """
        llm 直接回答 fact checking 问题

        Args:
            question: 问题
        
        Returns:
            一个 dict, 包含 analysis 和 answer, score
        """
        prompt = f"""
            Role:
            You are a Fact-Checking Analyst specializing in the biomedical domain.

            Instructions:
            1.  Evaluate the claim within the `Question` based on your internal knowledge of biology and medicine.
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
            Question: Can antibiotics be effectively used to treat infections caused by the influenza virus?
            Output:
            {{
                "analysis": "The claim is that antibiotics are effective against influenza virus infections. Influenza is caused by a virus. Antibiotics are antimicrobial drugs designed specifically to target bacteria, typically by disrupting their cell walls, protein synthesis, or DNA replication. Viruses have fundamentally different structures and replication mechanisms (they replicate inside host cells) and lack the specific targets that antibiotics act upon. Therefore, antibiotics are not effective against viral infections like influenza. Standard antiviral medications are used for influenza treatment. The claim is clearly refuted by basic principles of microbiology and pharmacology.",
                "answer": 2,
                "score": 1.0
            }}

            Task:
            Question: {question}
            Output: 
        """
        response = self._generate_response(prompt)
        # logger.info("answer_fc_directly response: %s", response)

        response_type = type(json.loads(response))
        if response_type == dict:
            return json.loads(response)
        elif response_type == list:
            return json.loads(response)[0]

    def qa_answer_query_rag(self, question: str, options: dict[str, str],
                            context: list[str]):
        """
        使用 rag 方法回答问题

        Args:
            question: 问题
            options: 选项
            context: rag 方法获取的相关文本
        
        Returns:
            一个 dict, 包含 analysis 和 answer, score
        """
        prompt = f"""
            Role
            You are an analytical assistant trained to solve biomedical questions by synthesizing retrieved information and internal knowledge. You carefully evaluate the credibility of external context, cross-reference it with your own expertise, and make informed decisions even when conflicting data arises.

            Instructions
            1. Context Check: Receive question, options, context. If context is empty or clearly irrelevant to the question, rely solely on internal knowledge (go to step 3).
            2. Analyze & Compare: If relevant context exists, analyze it and compare with your internal knowledge regarding the question and options.
                No Conflict: Choose the option best supported by both. Mark source as "external".
                Conflict Exists: Proceed to conflict resolution (step 3).
            3. Conflict Resolution / Internal Knowledge Use:
                If Conflict: Assess the reliability of the context versus your internal knowledge. Prioritize the source (context or internal) deemed more reliable/accurate for the specific biomedical question. If context is prioritized and used, mark source as "external"; otherwise, mark as "internal".
                If No Relevant Context (from step 1) or Context Disregarded: Rely on internal knowledge. Mark source as "internal".
            4. Decision: Choose the best option based on the prioritized information. If all options seem incorrect, select the 'least wrong' one.
            Output Format: Return a single JSON with keys: analysis (reasoning, including conflict handling if any), answer (A/B/C/D), source ("external" or "internal"), and score (0.0-1.0 confidence). Ensure answer is strictly one option letter.
            
            Output Format
            Return your response in a SINGLE JSON object with the following keys:
            analysis: Detailed reasoning about why the answer was chosen, including contradictions or uncertainties.
            answer: MUST be exactly one of the option keys (A/B/C/D) provided in the question.
            source: "external" if context is used, "internal" otherwise.
            score: Confidence score (0.0-1.0) based on clarity of evidence and consistency of reasoning.

            Example
            Question: "What causes tides?"
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
            Question: {question}
            Options: {options}
            Context: {context}
            Output:
        """
        response = self._generate_response(prompt)
        # logger.info("answer_qa_rag response: %s", response)

        response_type = type(json.loads(response))
        if response_type == dict:
            return json.loads(response)
        elif response_type == list:
            return json.loads(response)[0]

    def fc_answer_query_rag(self, question: str, context: list[str]):
        """
        使用 rag 方法回答问题

        Args:
            question: 问题
            context: rag 方法获取的相关文本
        
        Returns:
            一个 dict, 包含 analysis 和 answer, score
        """
        prompt = f"""
            Role:
            You are an AI Fact-Checker.

            Instructions:
            1.  Evaluate: Analyze the `Question` against the `Context` (if provided). If `Context` is absent or insufficient, use internal knowledge. Determine if the `Question` is: `0` (Supported), `1` (Not Enough Information), or `2` (Refuted).
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
            Question: mRNA vaccines work by introducing a weakened or inactivated virus into the body.
            Context: ["Unlike traditional vaccines which may use weakened or inactivated viruses, mRNA vaccines utilize messenger RNA (mRNA) sequences. These sequences instruct the body's cells to produce a specific antigen (like the spike protein of a virus), triggering an immune response without introducing the actual virus."]
            Output:
            {{
                "analysis": "The context explicitly contrasts mRNA vaccines with traditional vaccines, stating mRNA vaccines use mRNA sequences to instruct cell protein production, rather than introducing weakened or inactivated viruses. This directly refutes the claim.",
                "answer": 2,
                "source": "external",
                "score": 1.0
            }}

            Task:
            Question: {question}
            Context: {context}
            Output:
        """
        response = self._generate_response(prompt)
        # logger.info("answer_fc_rag response: %s", response)

        response_type = type(json.loads(response))
        if response_type == dict:
            return json.loads(response)
        elif response_type == list:
            return json.loads(response)[0]

    def _generate_response(self, prompt: str, **args) -> str:
        params = {
            "model": "qwen2.5-7b-instruct-1m",
            "temperature": 0.3,
            "max_tokens": 1000,
            "response_format": {
                "type": "json_object"
            },
        }
        params.update(args)
        # logger.info("model name: %s", params["model"])

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

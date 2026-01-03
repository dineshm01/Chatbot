from google import genai
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "models/gemini-2.5-flash"

DIAGRAM_PROMPTS = {
    "Diagram → Notes": "Convert this diagram explanation into 5 clear study notes.",
    "Diagram → Questions": "Generate 5 exam-style questions based on this diagram.",
    "Diagram → Glossary": "Extract key terms from this diagram and define them."
}

def diagram_transform(diagram_text: str, mode: str) -> str:
    prompt = DIAGRAM_PROMPTS[mode]
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt, diagram_text]
    )
    return resp.text or ""

def hybrid_answer(diagram_text: str, doc_text: str, question: str) -> str:
    # Fallback: no documents were actually retrieved
    if not doc_text.strip():
        return diagram_text

    prompt = f"""
Using the diagram explanation and the document context below, answer the question.

Rules:
- Prefer document facts when available
- Use the diagram only for structure/flow
- Do not hallucinate missing info

Diagram:
{diagram_text}

Document:
{doc_text}

Question:
{question}
"""

    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt]
    )
    return resp.text or diagram_text

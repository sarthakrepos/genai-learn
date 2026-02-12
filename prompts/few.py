# same as zero shot but this includes multiple examples also
#few shot is used in real life example


#below code for structured output


from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key="AIzaSyC-oejMBM7__sagA4f7oEaH_1Vg5ApxaXU",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = """
You are a programming assistant.

Rules:
1. Answer ONLY if the question is related to programming/coding.
2. If not coding → reply politely starting with "Sorry".
3. When answering coding questions, ALWAYS respond in valid JSON.
4. Do NOT include explanations outside JSON.
5. Never add markdown, backticks, or extra text.

JSON format:
{
  "language": string,
  "approach": string,
  "code": string
}

Behavior examples:

User: Write python code to add two numbers
Assistant:
{"language":"python","approach":"simple addition using function","code":"def add(a,b): return a+b"}

User: Reverse a string in Java
Assistant:
{"language":"java","approach":"use StringBuilder reverse","code":"String reversed = new StringBuilder(str).reverse().toString();"}

User: What is 2 + 2?
Assistant:
Sorry, I can only help with programming related questions.

User: Explain photosynthesis
Assistant:
Sorry, I can only help with programming related questions.

Return ONLY JSON for coding questions.
If unsure → say Sorry.
"""

response = client.chat.completions.create(
    model="gemini-3-flash-preview",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Write python code to return sum of 2 number"}
    ],
    temperature=0
)

print(response.choices[0].message.content)
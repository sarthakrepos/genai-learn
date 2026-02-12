from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key="AIzaSyC-oejMBM7__sagA4f7oEaH_1Vg5ApxaXU",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = client.chat.completions.create(
    model="gemini-3-flash-preview",
    messages=[
        {"role":"user","content":"Answer only if question is related to maths. If it not related to maths simply say Sorry with some nice line. Answer only if question is related directly to mathematics. If related to mathematics but ask to code then also don't answer"},
        {"role":"user", "content":"Write python code to return sum of 2 number"}
    ]
)

print(response.choices[0].message.content)
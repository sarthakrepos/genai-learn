from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client(
    api_key="AIzaSyC-oejMBM7__sagA4f7oEaH_1Vg5ApxaXU"
)

response = client.models.generate_content(
    model="gemini-3-flash-preview", contents="Explain how AI works in a few words"
)
print(response.text)
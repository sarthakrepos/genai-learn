import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

text = "Hey There! My name is Sarthak"

tokens = enc.encode(text)

print("tokens", tokens)

decoded=enc.decode([25216, 3274, 0, 3673, 1308, 382, 336, 7087, 422])

print("Decoded", decoded)
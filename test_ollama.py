import ollama

response = ollama.chat(
    model="llama3",
    messages=[
        {"role": "user", "content": "Explain recursion simply"}
    ]
)

print(response["message"]["content"])
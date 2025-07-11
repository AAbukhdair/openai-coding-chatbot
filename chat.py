import os
from openai import OpenAI

client = OpenAI()

def ask(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print("Ask your coding question (type 'exit' to quit):")
    while True:
        q = input(">> ")
        if q.strip().lower() == "exit":
            break
        print(ask(q), "\n")

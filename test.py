import openai

client = openai.OpenAI(

api_key="",



)

completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "hi."
        }
    ]
)

print(completion.choices[0].message)
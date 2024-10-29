from openai import OpenAI

client = OpenAI(

    base_url="http://172.30.64.1:1234/v1",
    api_key="not-needed"
)


def get_chatgpt_response(prompt):
    response = client.chat.completions.create(
        model="local-model",  # Use 'gpt-4' if you have access
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    return response.choices[0].message


# Test the function
print(get_chatgpt_response('Hello, how can I help you today?'))

import openai
from openai import OpenAI


def translate_text(text):
    command = "해당 내용에서 약 이름 정보 딱 한단어만 추출해주세요. 불필요한 정보들은 다 제거하세요."
    content = f"{text} {command}"
    client = OpenAI(api_key='') # TODO key는 OpenAI api 키가 들어가야 함
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": "You are a drug expert who must explain as quickly and concisely as possible."},
                {"role": "user", "content": content},
            ]
        )
        translated_text = response.choices[0].message.content
        
        return translated_text
    except Exception as e:
        return f"An error occurred: {str(e)}"

# 사용 방법
# filter_texts = translate_text(text)



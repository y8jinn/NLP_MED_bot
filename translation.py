from openai import OpenAI


def translation_text(text):
    command = "해당 내용을 일본어로로 정확하게 번역해주세요."
    content = f"{text} {command}"
    client = OpenAI(api_key='') # TODO key는 OpenAI api 키가 들어가야 함
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "너는 이제부터 일본어 통역사야,해당 언어를 일본어로 정확하게 번역해"},
                {"role": "user", "content": content},
            ]
        )
        translated_text = response['choices'][0]['message']['content']
        return translated_text
    except Exception as e:
        return f"An error occurred: {str(e)}"
    

from openai import OpenAI


def create_output(text):

    client = OpenAI(api_key='') # TODO key는 OpenAI api 키가 들어가야 함
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": f"You are a pharmacist. You deliver information quickly and concisely based on the given {text}."},
        {"role": "user", "content": '해당 정보를 바탕으로 사용자 질문에 대해 답변을 생성해주세요. 질문 정보가 없다면 그냥 해당 약 정보를 제공하세요'}
    ]
    )
    return completion.choices[0].message.content


from openai import OpenAI


def translation_text_other_country(text,user_country):

    client = OpenAI(api_key='') # TODO key는 OpenAI api 키가 들어가야 함

    completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": f" Please translate this into {user_country} language correctly."},
                {"role": "user", "content": text},
            ],temperature=1.0
        )    # Please explain it kindly like a professional pharmacist in {} language.
            # You are a {user_country} language interpreter, translate the given content into {user_country} language very accurately and quickly.
            # Please explain it kindly like a professional pharmacist in a very accurate {user_country} language.
    return completion.choices[0].message.content


def translation_text_korea(text):

    client = OpenAI(api_key='') # TODO key는 OpenAI api 키가 들어가야 함

    completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": "You are a Korean interpreter from now on, translate the given content correctly into Korean. If the content is korean, print it out as it is"},
                {"role": "user", "content": text},
            ]
        )

    return completion.choices[0].message.content

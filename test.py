from OCR_image import *
from text_summarize import *
from create_answer import *
import imghdr
import os
from rag import *

def is_image(file_path):
    """
    이미지인지 확인 하는 함수
    """
    image_formats = ['jpeg', 'png', 'gif', 'bmp', 'tiff']
    format = imghdr.what(file_path)
    return format in image_formats

def classify_input(input_data):
    """
    텍스트인지 이미지인지 구분 / 이미지는 경로로 들어오는 것으로 예상
    """
    if os.path.isfile(input_data):
        if is_image(input_data):
            result = detect_text(input_data,'') #TODO key는 구글 비전 키가 들어가야함
            result = translate_text(result)
            image_info='image'
            return result,image_info
    else:
        return input_data


if __name__ == "__main__":
    user_country = input('choice your country: ') # 나라 선택
    user_state = input('Are you a children/pregnant/elders/None?: ') # 어린이/임산부/노인 선택
    user_input = '腹が痛すぎる場合は、どの薬を飲むべきですか？' 
    text_out = classify_input(user_input) # input 구분


    if text_out[1]=='image':
        final = process_image(text_out[0],user_state) # rag로 관련 약 찾기
        print(final)
        print("----")
        before_answer = create_output(final) # 답변 생성(한국어)
        final_answer = translation_text_other_country(before_answer,user_country) # 답변 번역(자국어)
        print(final_answer)

    else:
        translate_input = translation_text_korea(text_out) # 사용자 증상 한국어로 번역
        final = process_text(translate_input,user_state) # rag로 관련 약 찾기
        print(final)
        print("----")
        before_answer = create_output(final) # 답변 생성(한국어)
        final_answer = translation_text_other_country(before_answer,user_country)#  # 답변 번역(자국어)
        print(final_answer)


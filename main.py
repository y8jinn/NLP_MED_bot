import os
import imghdr
import requests
from PIL import Image
from flask import Flask, request, jsonify
import threading
import warnings
from OCR_image import *
from text_summarize import *
from create_answer import *
from rag import *

warnings.filterwarnings('ignore')

app = Flask(__name__)

user_info = {
    "country": None,
    "state": None,
    "symptoms": None
}

state_tracker = {
    "expecting_country": True,
    "expecting_state": False,
    "expecting_symptoms": False
}

def is_image(file_path):
    """
    이미지인지 확인하는 함수
    """
    image_formats = ['jpeg', 'png', 'gif', 'bmp', 'tiff']
    format = imghdr.what(file_path)
    return format in image_formats

def download_image(image_url):
    """
    URL로부터 이미지를 다운로드하여 임시 파일로 저장하는 함수
    """
    response = requests.get(image_url)
    if response.status_code == 200:
        temp_image_path = "temp_image.png"
        with open(temp_image_path, 'wb') as f:
            f.write(response.content)
        return temp_image_path
    else:
        raise Exception("Failed to download image")

def classify_input(input_data):
    """
    텍스트인지 이미지인지 구분 / 이미지는 경로로 들어오는 것으로 예상
    """
    print(f"Classifying input: {input_data}")
    if os.path.isfile(input_data):
        if is_image(input_data):
            try:
                # 이미지 파일을 읽어와서 처리
                image = Image.open(input_data)
                temp_image_path = "temp_image.png"
                image.save(temp_image_path)
                print(f"Image saved to temporary path: {temp_image_path}")
                
                result = detect_text(temp_image_path, '') # TODO key는 구글 비전 키가 들어가야 함
                print(f"OCR result: {result}")
                
                result = translate_text(result)
                print(f"Translated text: {result}")
                
                image_info = 'image'
                return result, image_info
            except Exception as e:
                print(f"Error processing image: {e}")
                return None, 'error'
    elif input_data.startswith("http://") or input_data.startswith("https://"):
        try:
            temp_image_path = download_image(input_data)
            print(f"Downloaded image saved to: {temp_image_path}")
            
            result = detect_text(temp_image_path, '') # TODO key는 구글 비전 키가 들어가야 함
            print(f"OCR result: {result}")
            
            result = translate_text(result)
            print(f"Translated text: {result}")
            
            image_info = 'image'
            return result, image_info
        except Exception as e:
            print(f"Error downloading or processing image: {e}")
            return None, 'error'
    else:
        print(f"Input is text: {input_data}")
        return input_data, 'text'

def process_request_async(symptoms, callback_url):
    """비동기로 긴 처리 과정을 수행하고 콜백 URL을 통해 결과를 전송하는 함수."""
    try:
        text_out, input_type = classify_input(symptoms)  # 입력 분류
        if input_type == 'image':
            final = process_image(text_out, user_info["state"])  # rag로 관련 약 찾기
        else:
            translate_input = translation_text_korea(text_out)  # 사용자 증상 한국어로 번역
            final = process_text(translate_input, user_info["state"])  # rag로 관련 약 찾기
        
        before_answer = create_output(final)  # 답변 생성(한국어)
        final_answer = translation_text_other_country(before_answer, user_info["country"])  # 답변 번역(자국어)
        send_callback_response(callback_url, final_answer)  # 콜백 URL로 결과 전송
    except Exception as e:
        print(f"Error in processing request asynchronously: {e}")

def send_callback_response(callback_url, text):
    """비동기 처리 후 콜백 URL로 결과를 보내는 함수."""
    callback_data = {
        "version": "2.0",
        "useCallback": True,
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": text
                    }
                }
            ]
        }
    }
    response = requests.post(callback_url, json=callback_data)
    print("Callback POST response:", response.status_code, response.text)

@app.route("/keyword", methods=['POST'])
def keyword():
    req = request.get_json()
    text_ck = req['userRequest']['utterance']
    callback_url = req['userRequest'].get('callbackUrl')

    if state_tracker["expecting_country"]:
        state_tracker["expecting_country"] = False
        state_tracker["expecting_state"] = True
        return jsonify({
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "Which country are you from?"
                        }
                    }
                ]
            }
        })
    elif state_tracker["expecting_state"]:
        user_info["country"] = text_ck
        state_tracker["expecting_state"] = False
        state_tracker["expecting_symptoms"] = True
        return jsonify({
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "Please provide your state (children/pregnant/elders/None)."
                        }
                    }
                ]
            }
        })
    elif state_tracker["expecting_symptoms"]:
        user_info["state"] = text_ck
        state_tracker["expecting_symptoms"] = False  # 모든 질문이 완료되었음을 표시
        return jsonify({
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": "Please describe your symptoms or share medication image."
                        }
                    }
                ]
            }
        })
    else:
        user_info["symptoms"] = text_ck
        if callback_url:  # 콜백 URL이 제공되면 비동기 처리를 시작
            threading.Thread(target=process_request_async, args=(text_ck, callback_url)).start()
            return jsonify({
                "version": "2.0",
                "useCallback": True,
                "data": {
                    "text": "Processing your request..."
                }
            })
        else:
            text_out, input_type = classify_input(text_ck)
            if input_type == 'image':
                final = process_image(text_out, user_info["state"])  # rag로 관련 약 찾기
            else:
                translate_input = translation_text_korea(text_out)  # 사용자 증상 한국어로 번역
                final = process_text(translate_input, user_info["state"])  # rag로 관련 약 찾기
            
            before_answer = create_output(final)  # 답변 생성(한국어)
            final_answer = translation_text_other_country(before_answer, user_info["country"])  # 답변 번역(자국어)
            return jsonify({
                "version": "2.0",
                "template": {
                    "outputs": [
                        {
                            "simpleText": {
                                "text": final_answer
                            }
                        }
                    ]
                }
            })

if __name__ == "__main__":
    state_tracker["expecting_country"] = True
    state_tracker["expecting_state"] = False
    state_tracker["expecting_symptoms"] = False
    app.run(host='0.0.0.0', port=5000, threaded=True)

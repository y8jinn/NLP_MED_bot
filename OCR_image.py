from google.cloud import vision
from google.oauth2 import service_account
import io
import json

def detect_text(path, json_path):
    """
    이미지 파일에서 텍스트를 검출하는 함수.
    
    파라미터:
    - path: 이미지 파일의 경로
    - credentials_json: 서비스 계정 키 파일의 내용을 포함한 JSON 객체
    
    return:
    - 텍스트
    """
    with open(json_path, 'r') as json_file:
        credentials_json = json.load(json_file)

    credentials = service_account.Credentials.from_service_account_info(credentials_json)
    client = vision.ImageAnnotatorClient(credentials=credentials)

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    # 첫 번째 검출된 텍스트 반환(전체 텍스트)
    if texts:
        return texts[0].description
    else:
        return 'No text found.'
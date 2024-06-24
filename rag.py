from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import PromptTemplate

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def process_text(query,user_state):

    # Create embeddings model
    embeddings_model = HuggingFaceInferenceAPIEmbeddings(
        api_key='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', # # TODO key는 huggingface api 키가 들어가야 함
        model_name='bespin-global/klue-sroberta-base-continue-learning-by-mnr'
    )
        
    # Load vector database
    db = FAISS.load_local(
        './drug_info', 
        embeddings_model, 
        'drug_db', 
        allow_dangerous_deserialization=True
    )

    # Retrieve documents
    info = db.similarity_search(query,k=1)


    # Create prompt
    효능 = info[0].page_content.split(':')[1].strip()
    제품명 = info[0].metadata['제품명']
    사용방법 = info[0].metadata['사용방법']
    숙지사항 = info[0].metadata['숙지사항']
    주의사항 = info[0].metadata['사용상 주의사항']
    음식 = info[0].metadata['사용 시 주의해야 할 음식 또는 약']
    이상반응 = info[0].metadata['이상반응']

###########################################################################################################
    if user_state != "None":
        db_path = '{}_db'.format(user_state)
        additional_db = FAISS.load_local(
        './drug_info', 
        embeddings_model, 
        db_path, 
        allow_dangerous_deserialization=True)
        add_info=additional_db.similarity_search_with_score(제품명,k=1)

        content, score = add_info[0] # q.metadata['상세정보']
        if score <= 115:
            추가_주의사항= content.metadata['상세정보']
        else:
            추가_주의사항 = "관련 약 정보가 없습니다"
    elif user_state == "None":
        추가_주의사항 = "관련 약 정보가 없습니다"
###########################################################################################################

    multiple_input_prompt = PromptTemplate(
        input_variables=["효능","제품명", "사용방법", "숙지사항", "주의사항",'음식','이상반응','추가_주의사항','질문'],
        template="다음 정보만을 사용해서 사용자 질의에 대해 정확하고 구체적인 답변을 생성해주세요.\n\n"
                 "제품명: {제품명}\n\n"
                 "효능: {효능}\n\n"
                 "사용방법: {사용방법}\n\n"
                 "숙지사항: {숙지사항}\n\n"
                 "사용시 주의사항: {주의사항}\n\n"
                 "사용 시 주의해야 할 음식 또는 약: {음식}\n\n"
                 "사용 시 이상반응: {이상반응}\n\n"
                 "추가_주의사항: {추가_주의사항}\n\n"
                 "질문: {질문}"
            
    )

    final_prompt = multiple_input_prompt.format(
        제품명=제품명,
        효능=효능,
        사용방법=사용방법, 
        숙지사항=숙지사항, 
        주의사항=주의사항,
        음식=음식,
        이상반응=이상반응,
        추가_주의사항 = 추가_주의사항 if 추가_주의사항 else "정보 없음",
        질문=query
    )

    return final_prompt


def process_image(query,user_state):
    # Create embeddings model
    embeddings_model = HuggingFaceInferenceAPIEmbeddings(
        api_key='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',
        model_name='bespin-global/klue-sroberta-base-continue-learning-by-mnr'
    )

    # Load vector database
    db = FAISS.load_local(
        './drug_info', 
        embeddings_model, 
        'drug_db_for_image', 
        allow_dangerous_deserialization=True
    )

    # Retrieve documents
    info = db.similarity_search(query,k=1)


    # 주성분, 사용방법, 숙지사항, 사용상 주의사항, 사용 시 주의해야 할 음식 또는 약, 이상반응, 보관방법, 효능


    # Create prompt
    제품명 = info[0].page_content.split(':')[1].strip()
    효능 = info[0].metadata['효능']
    사용방법 = info[0].metadata['사용방법']
    숙지사항 = info[0].metadata['숙지사항']
    주의사항 = info[0].metadata['사용상 주의사항']
    음식 = info[0].metadata['사용 시 주의해야 할 음식 또는 약']
    이상반응 = info[0].metadata['이상반응']
    보관방법 = info[0].metadata['보관방법']

###########################################################################################################
    if  user_state != "None":
        db_path = '{}_db'.format(user_state)
        additional_db = FAISS.load_local(
        './drug_info', 
        embeddings_model, 
        db_path, 
        allow_dangerous_deserialization=True)
        add_info=additional_db.similarity_search_with_score(제품명,k=1)

        content, score = add_info[0] # q.metadata['상세정보']
        if score <= 118:
            추가_주의사항= content.metadata['상세정보']
        else:
            추가_주의사항 = "관련 약 정보가 없습니다"

    elif user_state == "None":
        추가_주의사항 = "관련 약 정보가 없습니다"
###########################################################################################################

    multiple_input_prompt = PromptTemplate(
        input_variables=["효능","제품명", "사용방법", "숙지사항", "주의사항",'음식','이상반응','보관방법','추가_주의사항'],
        template="다음 정보를 기반으로 해당 약에 대한 정보를 사용자한테 친절하게 알려주세요.\n\n"
                 "제품명: {제품명}\n\n"
                 "효능: {효능}\n\n"
                 "사용방법: {사용방법}\n\n"
                 "숙지사항: {숙지사항}\n\n"
                 "사용시 주의사항: {주의사항}\n\n"
                 "사용 시 주의해야 할 음식 또는 약: {음식}\n\n"
                 "사용 시 이상반응: {이상반응}\n\n"
                 "보관방법: {보관방법}\n\n"
                 "추가_주의사항: {추가_주의사항}"
    )

    final_prompt = multiple_input_prompt.format(
        제품명=제품명,
        효능=효능,
        사용방법=사용방법, 
        숙지사항=숙지사항, 
        주의사항=주의사항,
        음식=음식,
        이상반응=이상반응,
        보관방법=보관방법,
        추가_주의사항 = 추가_주의사항 if 추가_주의사항 else "정보 없음",
        질문=query
    )

    return final_prompt
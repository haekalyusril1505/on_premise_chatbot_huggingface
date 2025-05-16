import os
from langchain.llms import HuggingFacePipeline
from torch import cuda, bfloat16
import transformers
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain import PromptTemplate, LLMChain

from dotenv import load_dotenv
load_dotenv(".env")

HUGGING_FACE_KEY = os.getenv('HUGGING_FACE_KEY')


custom_prompt = '''
Nama anda adalah GueBOT. Anda adalah asisten yang suka membantu, penuh hormat, dan jujur. Selalu jawab semaksimal mungkin, sambil tetap aman.
Jawaban Anda tidak boleh berisi konten berbahaya, tidak etis, rasis, seksis, beracun, atau ilegal.
Jika Anda tidak mengetahui jawaban atas sebuah pertanyaan, mohon jangan membagikan informasi palsu dan arahkan user untuk berkonsultasi lewat aplikasi bernama DKONSUL.
Gunakan informasi di bawah ini untuk menjawab pertanyaan dari user.

Context: {context}
Question: {question}

Pastikan hanya memberikan jawaban di bawah ini.
Jawaban:

'''

access_token = HUGGING_FACE_KEY
model_id = 'indonlp/cendol-llama2-7b-chat'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

bnb_config = transformers.BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_quant_type='nf4',
  bnb_4bit_use_double_quant=True,
  bnb_4bit_compute_dtype=bfloat16
)

hf_auth = access_token
model_config = transformers.AutoConfig.from_pretrained(
  model_id,
  use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
  model_id,
  trust_remote_code=True,
  config=model_config,
  quantization_config=bnb_config,
  device_map='auto',
  use_auth_token=hf_auth
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
  model_id,
  use_auth_token=hf_auth
)

generate_text = transformers.pipeline(
  model=model,
  tokenizer=tokenizer,
  return_full_text=True,
  task='text-generation',
  temperature=0.1,
  max_new_tokens=512,
  repetition_penalty=1.1
)

model = HuggingFacePipeline(pipeline=generate_text)

embed = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                    model_kwargs={'device':'cuda'})

def set_custom_prompt():
    prompt = PromptTemplate(input_variables = ['context', 'question'], template = custom_prompt)
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm = model,
        chain_type = 'stuff',
        retriever = db.as_retriever(search_kwargs={'k':2}),
        #return_source_documents = True,
        chain_type_kwargs = {'prompt':prompt}
    )
    return qa_chain

def qa_bot():
#    try:
    print('Data masuk')
    embeddings = embed
    db = vectorstore
    llm = model
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
#    except Exception as e:
#        print('gak masok pak eko')
#        embeddings = embed
#        vector_path = '/root/chatbot/db_faiss'
#        db = FAISS.load_local(vector_path, embeddings, allow_dangerous_deserialization = True)
#        llm = model
#        qa_prompt = set_custom_prompt()
#        qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def results(query):
    qa_result = qa_bot()
    response = qa_result({'query':query})
    return response

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from streamlit_chat import message

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = '\n',
        chunk_size = 500,
        chunk_overlap = 50,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks
    
def get_vectorstore(text_chunks):
    embeddings = embed
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def main():
    st.set_page_config(page_title = 'Chat with PDF')
    
    st.header('Prototype Chat with PDF')
    if 'user_input' not in st.session_state:
        st.session_state['user_input'] = []

    if 'response' not in st.session_state:
        st.session_state['response'] = []
    with st.sidebar:
        st.subheader('Dokumen yang telah diupload')
        pdf_docs = st.file_uploader('Upload dokumen', accept_multiple_files = True)
        if st.button('Upload and Process'):
            with st.spinner('Processing'):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.write(text_chunks)

    def get_text():
        input_text = st.text_input("Tulis disini", key="input")
        return input_text

    user_input = get_text()
    if user_input:
        output = results(user_input)
        output = output['result'].split('Jawaban:\n\n')[-1]
    
    st.session_state.user_input.append(output)
    st.session_state.response.append(user_input)

    message_history = st.empty()

    if st.session_state['user_input']:
        for i in range(len(st.session_state['user_input']) - 1, -1, -1):
            message(st.session_state["user_input"][i], key=str(i),avatar_style="icons")
            message(st.session_state['response'][i], avatar_style="miniavs",is_user=True, key=str(i) + 'data_by_user')
    
if __name__ == '__main__':
    main()

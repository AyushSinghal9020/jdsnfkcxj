import json
from langchain_community.vectorstores import FAISS
from langchain.schema.document import Document 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage

import streamlit as st 

sample_json = json.load(open('base.json'))

documents = []

for doc in sample_json : 

    documents.append(Document(
        page_content = doc['page_content'] , 
        metadata = {
            'type' : doc['type'] , 
            'url' : doc['url']
        }
    ))

embedding = HuggingFaceEmbeddings(
        model_name = 'all-MiniLM-L6-v2'
    )

base_vc = FAISS.from_documents(
    documents , 
    embedding = embedding
)

base_vc.save_local('vc')

json_file = st.file_uploader('Upload the json file' , type = ['json'])
query = st.text_input('Query')

if st.button('Ask') : 

    ret_vc = FAISS.load_local('vc' , embeddings = embedding , allow_dangerous_deserialization = True)

    if json_file : 

        documents = []

        for doc in json.load(json_file) : 

            documents.append(Document(
                page_content = doc['page_content'] , 
                metadata = {
                    'type' : doc['type'] , 
                    'url' : doc['url']
                }
            ))

        ret_vc = FAISS.from_documents(
            documents , 
            embedding = HuggingFaceEmbeddings(
                model_name = 'all-MiniLM-L6-v2'
            )
        )

    bsdocs = base_vc.similarity_search(query)
    rsdocs = ret_vc.similarity_search(query)

    context = ' '.join([val.page_content for val in bsdocs]) + ' '.join([val.page_content for val in rsdocs])

    images = [val.metadata['url'] for val in bsdocs if val.metadata['type'] == 'image'] + [val.metadata['url'] for val in rsdocs if val.metadata['type'] == 'image']

    prompt = '''
You are a conversational chatbot, your task is to answer questions based on the context provided.

If the provided context does not match with query - just output 'No Specific context was provided for this query' and do not answer the query further

Context : {}

Query : {}
    '''
    chat = ChatCohere(cohere_api_key = 'vJZr4T4bWAJMn0kOdkSN1pmjxzrqLlPOy1YaA3fa')
    prompt = prompt.format(context , query)

    messages = [HumanMessage(content = prompt)]
    
    response = chat.invoke(messages).content

    st.write(response)

    for img in images : 

        st.markdown(
            f'''
            ![Image]({img})
            '''
        )

    mark_image = '\n'.join([
        f'''
        ![Image]({img})
        '''
        for img in images
    ])

    st.sidebar.markdown(response)
    st.sidebar.markdown(
        f'''
        ```
        {mark_image}
        ```
        '''
    )

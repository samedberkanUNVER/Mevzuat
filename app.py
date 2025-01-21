import streamlit as st
import openai
import faiss
import numpy as np
import json
from typing import List, Dict

# Must be the first Streamlit command
st.set_page_config(page_title="Mevzuat Tabanlı Soru-Cevap Sistemi", layout="wide")

# Initialize session state for chat history if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

def setup_sidebar() -> bool:
    """Setup sidebar and return whether the app is ready to process questions"""
    st.sidebar.header("Ayarlar")
    
    # API key input
    api_key = st.sidebar.text_input("OpenAI API Anahtarı", type="password")
    if api_key:
        openai.api_key = api_key
    
    # Debug mode toggle
    st.session_state.debug_mode = st.sidebar.checkbox("Debug Modu", value=False)
    
    # Embedding file upload
    uploaded_file = st.sidebar.file_uploader("Embedding Dosyası Yükle", type="json")
    
    if uploaded_file:
        try:
            # Load embeddings directly from uploaded file
            embedding_data = json.load(uploaded_file)
            st.session_state.embeddings = embedding_data["embeddings"]
            st.session_state.chunks = embedding_data["chunks"]
            st.session_state.embeddings_loaded = True
            st.sidebar.success("Embedding dosyası başarıyla yüklendi!")
            
            # Display some statistics
            st.sidebar.info(f"""
            Yüklenen Embedding Bilgileri:
            - Toplam Chunk Sayısı: {len(st.session_state.chunks)}
            - Embedding Boyutu: {len(st.session_state.embeddings[0])}
            """)
        except Exception as e:
            st.sidebar.error(f"Embedding dosyası yüklenirken hata oluştu: {str(e)}")
            st.session_state.embeddings_loaded = False
            
    return uploaded_file is not None and api_key and st.session_state.get('embeddings_loaded', False)

def get_embeddings(text: str) -> List[float]:
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response['data'][0]['embedding']

def answer_question(question: str, relevant_texts: List[str]) -> str:
    context = "\n\n".join(relevant_texts)
    prompt = f"Belge içeriği:\n{context}\n\nSoru: {question}\nCevap:"
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """
            Sen yardımsever bir asistansın. 
            Yorum yapma, 
            daha net cevap ver. 
            Hükmün nereden çıktığını bilmek istiyoruz. 
            Sana verilen dokümandaki hükme referans bulunarak cevaplar ver.
            VERMENİ İSTEDİĞİM ÖRNEK CEVAP:
            1- "Yönetmeliğinin "Köşe Başı Parsellerde Kotlandırma" başlıklı 13.maddesinde belirtildiği üzere..."
            2- "YAPI DENETİMİ HAKKINDA KANUNun "Amaç, kapsam ve tanımlar" başlıklı, 1.maddesinin ç bendinde belirtildiği üzere..."
            """},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

def process_question(question: str) -> str:
    # Use FAISS for vector similarity search
    embeddings = st.session_state.embeddings
    chunks = st.session_state.chunks
    
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    query_embedding = get_embeddings(question)
    D, I = index.search(np.array([query_embedding]), k=60)
    relevant_chunks = [chunks[i] for i in I[0]]
    
    # Add debug information if needed
    if st.session_state.get('debug_mode', False):
        st.write("Debug: Found relevant chunks:", relevant_chunks)

    return answer_question(question, relevant_chunks)

def main():
    st.title("Mevzuat Tabanlı Soru-Cevap Sistemi")
    
    # Setup sidebar and check if app is ready
    is_ready = setup_sidebar()
    
    # Chat container
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Question input - only enable if embeddings are loaded
    if is_ready:
        if question := st.chat_input("Sorunuzu yazın..."):
            # Display user message
            with chat_container:
                with st.chat_message("user"):
                    st.write(question)
                
                # Add user message to history
                st.session_state.messages.append({"role": "user", "content": question})
                
                # Get and display assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Düşünüyorum..."):
                        response = process_question(question)
                    st.write(response)
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info("Lütfen OpenAI API anahtarınızı girin ve embedding dosyasını yükleyin.")

if __name__ == "__main__":
    main()
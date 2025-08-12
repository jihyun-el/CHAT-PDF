from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import streamlit as st
from langchain_openai import ChatOpenAI
import tempfile
from dotenv import load_dotenv, find_dotenv  # ✅ 추가

# ✅ .env 로드 (앱 시작 시 1회)
load_dotenv(find_dotenv(), override=False)

embedding_model = HuggingFaceEmbeddings(model_name="distiluse-base-multilingual-cased-v1")

def vectorize(pdf):
    # BytesIO를 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf.getvalue())
        file_path = tmp_file.name
    
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        text_splitter = CharacterTextSplitter(
            separator='.',
            chunk_size=500,
            chunk_overlap=100,
            length_function=len
        )

        texts = text_splitter.split_documents(documents)

        db = FAISS.from_documents(texts, embedding_model)
        retriever = db.as_retriever(search_type='similarity', search_kwargs={"k": 3})
        return retriever
    finally:
        # 임시 파일 반드시 삭제
        try:
            os.unlink(file_path)
        except OSError:
            pass

@st.cache_resource(show_spinner=False)
def load_openai_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.0):
    # ✅ .env에서 올라온 환경변수만 사용
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY를 찾을 수 없습니다. "
            ".env에 OPENAI_API_KEY=... 형태로 넣고 실행하세요."
        )
    return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)

def search(query: str, retriever):
    docs = retriever.invoke(query)
    return docs or []

def build_prompt(query: str, docs):
    lines = []
    lines.append("아래 '자료'만 근거로 한국어로 간결히 답하세요.")
    lines.append("- 자료 밖 정보를 추측하지 마세요.")
    lines.append("- 답할 수 없으면 '제공된 문서에서 찾지 못했습니다.'라고 말하세요.\n")
    lines.append(f"질문:\n{query}\n")
    lines.append("자료:")
    for i, d in enumerate(docs, 1):
        lines.append(f"[문서{i}]\n{d.page_content}\n")
    lines.append("답변:")
    return "\n".join(lines)

def generate_with_llm(llm: ChatOpenAI, prompt: str) -> str:
    resp = llm.invoke(prompt)
    return resp.content.strip()

def main():
    st.set_page_config(page_title="🤖 CHAT pdf", page_icon="🤖", layout="centered")
    st.title("🤖 CHAT pdf")
    uploaded_file = st.file_uploader("파일을 업로드하세요", type=["pdf"])
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "llm" not in st.session_state:
        st.session_state.llm = load_openai_llm("gpt-4o-mini", temperature=0.0)

    if uploaded_file is not None:
        if "retriever" not in st.session_state:
            with st.spinner("PDF 처리 중..."):
                st.session_state.retriever = vectorize(uploaded_file)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("pdf에게 궁금한 점을 물어보세요.")
    if user_input and "retriever" in st.session_state:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        query = user_input.strip()
        docs = search(query, st.session_state.retriever)
        if not docs:
            answer = "제공된 문서에서 찾지 못했습니다."
        else:
            prompt = build_prompt(query, docs)
            answer = generate_with_llm(st.session_state.llm, prompt)
        
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()

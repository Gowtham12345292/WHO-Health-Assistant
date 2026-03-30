import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Page config
st.set_page_config(page_title="WHO Health Assistant", layout="centered")

# Constants
PDF_DIRECTORY = "data/"
CHROMA_DB_PATH = "chroma_db/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5

WHO_URLS = [
    "https://www.who.int/news-room/fact-sheets/detail/malaria",
    "https://www.who.int/news-room/fact-sheets/detail/diabetes",
    "https://www.who.int/news-room/fact-sheets/detail/cancer",
    "https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)",
    "https://www.who.int/news-room/fact-sheets/detail/hypertension",
    "https://www.who.int/news-room/fact-sheets/detail/hiv-aids",
    "https://www.who.int/news-room/fact-sheets/detail/tuberculosis",
    "https://www.who.int/news-room/fact-sheets/detail/asthma",
    "https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue",
    "https://www.who.int/news-room/fact-sheets/detail/headache-disorders",
    "https://www.who.int/news-room/fact-sheets/detail/mental-disorders",
    "https://www.who.int/news-room/fact-sheets/detail/depression",
    "https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight",
    "https://www.who.int/news-room/fact-sheets/detail/pneumonia",
    "https://www.who.int/news-room/fact-sheets/detail/cholera",
    "https://www.who.int/news-room/fact-sheets/detail/epilepsy",
    "https://www.who.int/news-room/fact-sheets/detail/food-safety",
    "https://www.who.int/news-room/fact-sheets/detail/physical-activity",
    "https://www.who.int/news-room/fact-sheets/detail/tobacco",
    "https://www.who.int/news-room/fact-sheets/detail/alcohol",
    "https://www.who.int/news-room/fact-sheets/detail/immunization-coverage",
    "https://www.who.int/news-room/fact-sheets/detail/dementia",
    "https://www.who.int/news-room/fact-sheets/detail/drinking-water",
    "https://www.who.int/news-room/fact-sheets/detail/climate-change-and-health",
]

PROMPT_TEMPLATE = """You are an AI Health Assistant powered by WHO documents.
Answer ONLY from the provided context.
If the context does not contain enough information, say:
"I don't have enough information from WHO documents to answer this."
Always mention the source. Do not provide medical diagnoses.

Context:
{context}

Question:
{question}

Answer:"""


@st.cache_resource
def build_pipeline():
    """Load documents, create embeddings, build vector DB and RAG chain."""

    all_documents = []

    # Load PDFs if available
    if os.path.exists(PDF_DIRECTORY):
        pdf_loader = DirectoryLoader(PDF_DIRECTORY, glob="**/*.pdf", loader_cls=PyPDFLoader)
        pdf_docs = pdf_loader.load()
        all_documents.extend(pdf_docs)

    # Load WHO fact sheets from web
    web_loader = WebBaseLoader(WHO_URLS)
    web_docs = web_loader.load()
    web_docs = [doc for doc in web_docs if len(doc.page_content) > 100]
    all_documents.extend(web_docs)

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(all_documents)

    # Embeddings and vector DB
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_DB_PATH)

    # RAG chain
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, len(pdf_docs) if os.path.exists(PDF_DIRECTORY) else 0, len(web_docs), len(chunks)


# Sidebar
with st.sidebar:
    st.header("Configuration")
    groq_key = st.text_input("Groq API Key:", type="password")
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        st.success("API key set")

    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown(
        "1. WHO PDFs and fact sheets are loaded\n"
        "2. Text is chunked and embedded\n"
        "3. Your question is matched with relevant chunks\n"
        "4. Llama 3.3 generates an answer from context\n"
        "5. Answer is grounded in WHO data"
    )
    st.markdown("---")
    st.markdown(
        "Built by **Vemula Gowtham**\n\n"
        "[GitHub](https://github.com/Gowtham12345292) | "
        "[LinkedIn](https://linkedin.com/in/vemula-gowtham-624206286)"
    )

# Main app
st.title("AI-Powered WHO Health Assistant")
st.markdown("Ask health questions and get verified answers from WHO documents using RAG.")
st.markdown("---")

if not os.environ.get("GROQ_API_KEY"):
    st.warning("Please enter your Groq API key in the sidebar to get started.")
    st.stop()

# Build pipeline
with st.spinner("Loading WHO documents and building knowledge base..."):
    rag_chain, pdf_count, web_count, chunk_count = build_pipeline()

st.success(f"Knowledge base ready: {pdf_count} PDF pages + {web_count} WHO fact sheets = {chunk_count} chunks indexed")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
question = st.chat_input("Ask a health question...")

if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Searching WHO documents..."):
            answer = rag_chain.invoke(question)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# Example questions
if not st.session_state.messages:
    st.markdown("### Try asking:")
    examples = [
        "What are the symptoms of malaria?",
        "How can diabetes be prevented?",
        "What causes headaches?",
        "What does WHO recommend for drinking water?",
        "How does climate change affect health?",
    ]
    for q in examples:
        if st.button(q, key=q):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

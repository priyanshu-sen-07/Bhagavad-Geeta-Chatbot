import streamlit as st
import os
import getpass
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model

# Get API key if not set
os.environ["GOOGLE_API_KEY"] = "AIzaSyBHeZdRZa3GCDQkimquTODYIe7chpkCUOQ"


# Set Streamlit page config
st.set_page_config(page_title="📜 Geeta Chatbot", layout="centered")

# Set background CSS
st.markdown(
    """
    <style>
    body {
        background-image: url("https://i.pinimg.com/originals/3e/85/ff/3e85ff0461cb00e96ff8d5b2dd044a4b.jpg");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        background-repeat: no-repeat;
        color: #222222;
    }

    .stApp {
        background-color: rgba(0, 0, 0, 0.4); /* Dark transparent background */
        padding: 2rem;
        border-radius: 12px;
        backdrop-filter: blur(4px); /* optional: adds a glassmorphism blur */
    }

    h1, h2, h3, h4, h5, h6, p, label, .css-1cpxqw2, .css-ffhzg2, .css-1v0mbdj {
        color: #ffffff !important; /* force white text */
    }

    input, textarea {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        border: 1px solid #ccc;
    }

    button {
        background-color: #6c63ff !important;
        color: white !important;
        border: none;
        border-radius: 6px;
    }

    button:hover {
        background-color: #574fd6 !important;
    }

</style>

    """,
    unsafe_allow_html=True
)

st.title("🪔 Bhagavad Geeta Chatbot")
st.caption("Ask anything from the Geeta")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load vector store + Gemini model
@st.cache_resource
def load_chain():
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    # Custom prompt
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a spiritual assistant that answers questions based on the Bhagavad Geeta.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer

Context:
{context}

Question:
{question}

Answer:"""
    )

    # Use Gemini 2.5 Flash
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt_template},
    )

    return qa_chain

chain = load_chain()

# Input box
user_query = st.chat_input("Ask something from the Geeta...")

# Handle user query
if user_query:
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.spinner("Thinking..."):
        result = chain.run(user_query)
    st.session_state.chat_history.append({"role": "ai", "content": result})

# Display messages
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])
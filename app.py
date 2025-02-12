import streamlit as st
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import os
import json

# Configuration
PERSIST_DIR = "medical_db"
DOCUMENT_DIR = "medical_books"
EMBEDDING_MODEL = "abhinand/MedEmbed-small-v0.1"
HF_API_TOKEN = "your api key from huggingface here"
PREDEFINED_RESPONSES_FILE = "predefined_responses.json"

# Load predefined responses
with open(PREDEFINED_RESPONSES_FILE, encoding='utf-8') as f:
    PREDEFINED_RESPONSES = json.load(f)

st.set_page_config(page_title="Medical Chat Assistant", page_icon="🏥")
st.title("🏥 Medical Chat Assistant")

# Inject Custom CSS
def load_css():
    css = """
    <style>
        .chat-container {
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .chat-row {
            display: flex;
            align-items: center;
            width: 100%;
            margin-bottom: 10px;
        }

        .row-reverse {
            flex-direction: row-reverse;
        }

        .chat-bubble {
            font-family: "Source Sans Pro", sans-serif, "Segoe UI", "Roboto", sans-serif;
            border: 1px solid transparent;
            padding: 12px 16px;
            margin: 5px 10px;
            max-width: 70%;
            font-size: 16px;
            font-weight: 500;
            box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.1);
            word-wrap: break-word;
        }

        .ai-bubble {
            background: rgb(240, 242, 246);
            border-radius: 12px;
            color: black;
        }

        .human-bubble {
            background: linear-gradient(135deg, rgb(0, 178, 255) 0%, rgb(0, 106, 255) 100%);
            color: white;
            border-radius: 20px;
        }

        .avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            object-fit: cover;
            margin-right: 10px;
        }

        .avatar-right {
            margin-left: 10px;
            margin-right: 0;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

load_css()

@st.cache_resource
def load_components():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    llm = HuggingFaceHub(
        repo_id="BioMistral/BioMistral-7B",
        model_kwargs={"temperature": 0.3, "max_new_tokens": 1024, "repetition_penalty": 1.2},
        huggingfacehub_api_token=HF_API_TOKEN
    )
    return embeddings, llm

def process_documents(embedding):
    if not os.path.exists(PERSIST_DIR) or len(os.listdir(PERSIST_DIR)) == 0:
        loader = DirectoryLoader(DOCUMENT_DIR, glob="*.pdf", loader_cls=PyPDFLoader, show_progress=True, use_multithreading=True)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        Chroma.from_documents(texts, embedding, persist_directory=PERSIST_DIR)

embeddings, llm = load_components()

if os.path.exists(PERSIST_DIR):
    vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
else:
    process_documents(embeddings)
    vector_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

# Load avatar images
robot_avatar = "https://cdn-icons-png.flaticon.com/512/4712/4712033.png"  # AI avatar
user_avatar = "https://cdn-icons-png.flaticon.com/512/219/219983.png"  # User avatar

MEDICAL_PROMPT_TEMPLATE = """
[INST] You are a compassionate and knowledgeable medical assistant dedicated to providing safe, clear, and supportive health advice.

Definition: A healthcare assistant is an AI-powered tool that helps patients understand their symptoms, manage their health, and determine when to seek further medical assistance.

Context: {context}
Question: {question}

Respond using the following format:
1.  **Summary:** Briefly explain what the patient's inquiry is about in plain language.
2. **Key Information:** List 2-4 bullet points outlining the most important facts or considerations.
3. **Self-Care Recommendations:** Suggest simple and safe self-care steps, if applicable.
4. **Urgency Guidance:** Advise clearly when to seek immediate medical help.

Guidelines:
- Use everyday language while including essential medical terminology (explain any technical terms in parentheses).
- Keep the response concise (no more than five sentences overall).
- Incorporate emojis where they enhance clarity and empathy.
- Focus on clarity, accuracy, and patient safety.
- Say "I'm not certain" if unsure or proper answer is not present [/INST]"""

PROMPT = PromptTemplate(template=MEDICAL_PROMPT_TEMPLATE, input_variables=["context", "question"])

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={'k': 3}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

def get_response(query):
    query_lower = query.lower().strip()
    response_categories = [
        ("greetings", PREDEFINED_RESPONSES["greetings"], False),
        ("gratitude", PREDEFINED_RESPONSES["gratitude"], False),
        ("farewells", PREDEFINED_RESPONSES["farewells"], False),
        ("common_issues", PREDEFINED_RESPONSES["common_issues"], True),
        ("disclaimers", PREDEFINED_RESPONSES["disclaimers"], False)
    ]

    for category_name, category_data, has_sources in response_categories:
        for phrase in category_data:
            if phrase in query_lower:
                response = {
                    "answer": category_data[phrase]["answer"] if has_sources else category_data[phrase],
                    "sources": category_data[phrase].get("sources", []) if has_sources else []
                }
                if category_name == "common_issues":
                    response["answer"] += f"\n\n{PREDEFINED_RESPONSES['disclaimers']['general']}"
                return response

    result = qa(query)
    raw_answer = result['result'].split('[/INST]')[-1].strip()
    sources = list({os.path.basename(doc.metadata['source']) for doc in result['source_documents']})

    return {
        "answer": raw_answer,
        "sources": sources
    }

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I assist with your medical inquiry today?"}]

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    avatar = robot_avatar if role == "assistant" else user_avatar

    div = f"""
    <div class="chat-row {'row-reverse' if role == 'user' else ''}">
        <img src="{avatar}" class="avatar {'avatar-right' if role == 'user' else ''}">
        <div class="chat-bubble {'human-bubble' if role == 'user' else 'ai-bubble'}">
            {content}
        </div>
    </div>
    """
    st.markdown(div, unsafe_allow_html=True)

if prompt := st.chat_input("Enter your medical query"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Analyzing..."):
        response = get_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

    st.rerun()  # Ensure styling applies to new messages

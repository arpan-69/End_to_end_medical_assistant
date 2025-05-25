
---

```markdown
# Medical Chat Assistant

The Medical Chat Assistant is a Streamlit-based AI chatbot designed to provide reliable and supportive responses to medical queries. It uses Retrieval-Augmented Generation (RAG) with LangChain, HuggingFace Transformers, and ChromaDB to deliver answers based on embedded medical documents.

---

## Features

- Conversational chatbot powered by BioMistral-7B
- Uses domain-specific MedEmbed embeddings
- Upload and process medical PDFs
- Predefined responses for greetings, disclaimers, and common questions
- Structured and compassionate medical answers
- Persistent vector store using ChromaDB
- Custom CSS for chat UI with avatars and chat bubbles

---

## Technologies Used

- Streamlit – for UI
- LangChain – to build the RAG pipeline
- HuggingFace Transformers – for LLM and embeddings
- ChromaDB – for vector database
- MedEmbed-small-v0.1 – for medical domain embeddings
- BioMistral-7B – for accurate biomedical question-answering

---

## Project Structure

```

.
├── medical\_chat\_assistant.py       # Main Streamlit app
├── medical\_books/                  # Directory containing medical PDFs
├── medical\_db/                     # ChromaDB persistent directory
├── predefined\_responses.json       # Predefined responses for FAQs and chat
├── .streamlit/
│   └── secrets.toml                # API secrets (excluded from version control)

````

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/medical-chat-assistant.git
cd medical-chat-assistant
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add HuggingFace API key

Create a file at `.streamlit/secrets.toml` and add:

```toml
HF_API_TOKEN = "your-huggingface-api-key"
```

### 4. Add PDF documents

Place all reference medical documents in PDF format inside the `medical_books/` folder.

### 5. Run the app

```bash
streamlit run medical_chat_assistant.py
```

---

## Example Queries

* What are the symptoms of vitamin B12 deficiency?
* I have a headache and feel dizzy. What should I do?
* Is it safe to take ibuprofen with paracetamol?

---

## Disclaimer

This chatbot is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions regarding a medical condition.

---

## Acknowledgments

* MedEmbed-small-v0.1 by Abhinand
* BioMistral-7B by BioMistral
* LangChain and Chroma for RAG integration

---

## Future Improvements

* Add PDF upload feature in the UI
* Enable voice input and response
* Multilingual support
* Save and export chat transcripts
* User feedback system

---

## License

This project is licensed under the MIT License.



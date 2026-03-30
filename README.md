# AI-Powered WHO Health Assistant

A RAG (Retrieval-Augmented Generation) chatbot that answers health queries using WHO health PDFs and 30+ WHO fact sheets. Built with LangChain, Groq (Llama 3.3), and ChromaDB.

---

## Problem Statement

Health information is scattered across hundreds of WHO documents and web pages. Users need a way to ask natural language questions and get accurate, verified answers without the AI making things up. This project solves that by grounding every answer in actual WHO data using RAG.

---

## Architecture

```
                    WHO PDFs (22 documents)
                            +
                    WHO Fact Sheets (30 web pages)
                            |
                    Document Loading
                    (PyPDFLoader + WebBaseLoader)
                            |
                    Text Chunking
                    (RecursiveCharacterTextSplitter)
                    chunk_size=500, overlap=100
                            |
                    Embedding Generation
                    (sentence-transformers/all-MiniLM-L6-v2)
                            |
                    Vector Storage (ChromaDB)
                            |
              User Query --> Similarity Search (Top-5)
                            |
                    Retrieved Context + Query
                            |
                    Custom Prompt Template
                            |
                    LLM (Groq - Llama 3.3 70B)
                            |
                    Grounded Answer
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq (Llama 3.3 70B Versatile) |
| Framework | LangChain |
| Vector Database | ChromaDB |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| PDF Loader | LangChain PyPDFLoader |
| Web Loader | LangChain WebBaseLoader |
| Text Splitter | RecursiveCharacterTextSplitter |
| Frontend | Streamlit |
| Language | Python |

---

## Key Results

| Metric | Value |
|--------|-------|
| Hallucination Reduction | ~90% (grounded in WHO context) |
| WHO PDFs Indexed | 22 documents |
| WHO Fact Sheets Indexed | 30 web pages |
| Total Chunks | 10,000+ |
| Chunk Size | 500 characters, 100 overlap |
| Retrieval | Top-5 similarity search |
| LLM Temperature | 0.2 (factual responses) |

---

## Data Sources

1. WHO Health PDFs: 22 official WHO reports covering polypharmacy, community health, malaria, and more
2. WHO Fact Sheets: 30 web pages scraped from who.int covering malaria, diabetes, cancer, cardiovascular diseases, hypertension, HIV/AIDS, tuberculosis, asthma, dengue, headache disorders, mental disorders, depression, obesity, pneumonia, diarrhoeal disease, influenza, hepatitis, cholera, epilepsy, rabies, food safety, physical activity, tobacco, alcohol, immunization, dementia, measles, antimicrobial resistance, drinking water, and climate change

---

## How to Run

### Option 1: Google Colab (Recommended)

1. Open `WHO_Health_Assistant.ipynb` in Google Colab
2. Get free API keys:
   - Groq API key from console.groq.com
3. Upload WHO PDF files when prompted
4. Run all cells sequentially
5. Start asking health questions using the `ask()` function

### Option 2: Run Streamlit App Locally

```bash
git clone https://github.com/Gowtham12345292/WHO-Health-Assistant.git
cd WHO-Health-Assistant

pip install -r requirements.txt

# Set API keys
export GROQ_API_KEY="your_groq_key"

# Place WHO PDFs in data/ folder

streamlit run app.py
```

---

## Project Structure

```
WHO-Health-Assistant/
|
├── README.md                        # Project documentation
├── WHO_Health_Assistant.ipynb        # Complete Colab notebook
├── app.py                           # Streamlit chat interface
├── requirements.txt                 # Dependencies
└── .gitignore                       # Ignored files
```

---

## RAG Pipeline Details

### Document Loading
- PDFs loaded using LangChain DirectoryLoader with PyPDFLoader
- WHO fact sheets loaded using WebBaseLoader from 30 WHO URLs
- Both sources combined into a single document collection

### Chunking Strategy
- RecursiveCharacterTextSplitter with 500 character chunks and 100 character overlap
- Split hierarchy: paragraphs first, then sentences, then words
- Preserves context across chunk boundaries

### Embedding and Retrieval
- HuggingFace sentence-transformers/all-MiniLM-L6-v2 for embeddings
- ChromaDB for persistent vector storage
- Top-5 similarity search for retrieval

### Hallucination Reduction
- Custom prompt forces LLM to answer ONLY from retrieved context
- Low temperature (0.2) for factual responses
- Explicit instruction to say "I don't have enough information" when context is insufficient
- Source document mentioned in every answer

### Prompt Template
```
You are an AI Health Assistant powered by WHO documents.
Answer ONLY from the provided context.
If the context does not contain enough information, say:
"I don't have enough information from WHO documents to answer this."
Always mention the source document.
Do not provide medical diagnoses.
```

---

## Example Queries and Responses

| Question | Topic Coverage |
|----------|---------------|
| What are the symptoms of malaria? | Infectious Diseases |
| How can diabetes be prevented? | Chronic Diseases |
| What causes headaches? | Neurological Disorders |
| What does WHO recommend for drinking water quality? | Environmental Health |
| How does climate change affect health? | Public Health |
| What are the symptoms of depression? | Mental Health |
| How does tobacco affect health? | Lifestyle and Prevention |

---

## Design Decisions

1. Groq (Llama 3.3 70B) over Gemini: Free API with generous rate limits, fast inference, no quota issues.
2. HuggingFace Embeddings over Google Embeddings: Free, no API key needed for embeddings, runs locally.
3. WebBaseLoader for WHO Fact Sheets: WHO blocks automated PDF downloads, but web pages are freely accessible and contain the same health information.
4. ChromaDB over FAISS: Easier persistence, metadata filtering, and LangChain integration.
5. Chunk Size 500: Balanced between preserving context and retrieval precision.
6. Temperature 0.2: Low temperature ensures factual, consistent responses critical for health information.
7. LCEL (LangChain Expression Language): Modern chain syntax using pipe operator for clean, readable code.

---

## What I Learned

- End-to-end RAG pipeline architecture and implementation
- Loading data from multiple sources (PDFs + web pages) into a unified pipeline
- Chunking strategies and their impact on retrieval quality
- Embedding models and semantic similarity search with ChromaDB
- Hallucination reduction through prompt engineering and grounded context
- Using Groq API with LangChain for fast, free LLM inference
- LangChain Expression Language (LCEL) for building modern chains
- Building chat interfaces with Streamlit

---

## Contact

Vemula Gowtham — [LinkedIn](https://linkedin.com/in/vemula-gowtham-624206286) | vemulagowtham7@gmail.com

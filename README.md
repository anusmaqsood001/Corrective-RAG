# RAG Chatbot with Corrective Retrieval

This project is a retrieval-augmented generation (RAG) chatbot built around a local PDF document. It uses a Google Gemini LLM, FAISS vector search, and a corrective workflow that decides whether retrieved context is good enough before generating an answer.

The project includes two entry points:

- [corrective_rag.py](corrective_rag.py): the core LangGraph-based RAG pipeline
- [corrective-streamlit-rag.py](corrective-streamlit-rag.py): a Streamlit web app that wraps the pipeline in a simple UI

---

## What this project does

The chatbot answers questions by:

1. Loading a PDF document from disk
2. Splitting it into chunks
3. Creating or loading a FAISS vector index
4. Retrieving the most relevant chunks for a user question
5. Asking an LLM to judge how relevant each retrieved chunk is
6. Deciding whether to rely on internal document context, web search, or both
7. Refining the context and generating a final answer

This is a “corrective” RAG approach because it does not blindly trust retrieved passages. Instead, it evaluates retrieval quality before answering.

---

## Key features

- PDF ingestion from a local document
- Semantic retrieval using embeddings and FAISS
- LLM-based relevance scoring for retrieved chunks
- Automatic routing between:
  - internal-document context only
  - web-search fallback
  - a combined internal + web context strategy
- Sentence-level filtering to keep only relevant context
- A simple Streamlit interface for interactive use

---

## Project structure

```text
RAG-Chatbot/
├── corrective_rag.py           # Core RAG workflow and LangGraph graph
├── corrective-streamlit-rag.py # Streamlit UI for the chatbot
├── environment.yml            # Conda environment dependencies
├── faiss_index/               # Local FAISS vector store files
├── lbdl.pdf                   # Source PDF used for retrieval
└── README.md                  # Project documentation
```

---

## How the workflow works

### 1. Document ingestion

The app loads the PDF file [lbdl.pdf](lbdl.pdf) and splits it into smaller chunks using a recursive text splitter.

### 2. Embedding and vector search

Each chunk is converted into embeddings and stored in a FAISS index. The system checks whether an existing index is already present in the [faiss_index](faiss_index) folder. If it exists, it loads it; otherwise it builds a new one.

### 3. Retrieval

For a user question, the retriever fetches the top $k$ most similar document chunks from the vector store.

### 4. Corrective evaluation

Each retrieved chunk is evaluated by an LLM using a strict scoring rubric:

- Score close to 1.0: the chunk is highly relevant
- Score close to 0.0: the chunk is not relevant

The script then classifies the retrieval result as:

- CORRECT: at least one chunk scored above a high threshold
- INCORRECT: all retrieved chunks scored below a low threshold
- AMBIGUOUS: the results are mixed or borderline

### 5. Context refinement

Depending on the verdict:

- CORRECT: use the retrieved internal chunks only
- INCORRECT: use web results only
- AMBIGUOUS: combine internal and web results

The selected context is then split into sentences and filtered again to retain only the sentences that directly answer the question.

### 6. Answer generation

Finally, the refined context is sent to the Gemini model to generate the final answer.

---

## Main components

### Core pipeline

The main logic is implemented in [corrective_rag.py](corrective_rag.py). It defines:

- a state schema for the workflow
- retrieval, evaluation, refinement, search, and generation nodes
- a LangGraph pipeline that connects them together

### Streamlit interface

The file [corrective-streamlit-rag.py](corrective-streamlit-rag.py) provides a simple chat-style interface where you can:

- type a question
- click the run button
- see the final answer
- inspect the verdict, reason, and refined context

---

## Environment requirements

This project uses Python 3.11 and a Conda environment defined in [environment.yml](environment.yml).

### Required services

You need:

- a Google Gemini API key
- a Tavily API key if you want the web-search fallback to work

---

## Setup instructions

### 1. Clone or open the project

Open the project folder in your terminal or VS Code workspace.

### 2. Create the Conda environment

From the project root, run:

```bash
conda env create -f environment.yml
conda activate rag-chatbot
```

### 3. Create a .env file

Create a file named `.env` in the project root with the following content:

```env
GEMINI_API_KEY=your_google_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
```

If you do not want to use web search, the app can still run, but the web-search path will not work properly without the Tavily key.

### 4. Make sure the PDF exists

The project expects the PDF file [lbdl.pdf](lbdl.pdf) to be present in the project root.

---

## How to run the app

### Option 1: Run the Streamlit web app (recommended)

From the project root:

```bash
streamlit run corrective-streamlit-rag.py
```

This will launch the web interface in your browser.

### Option 2: Run the core pipeline from Python

You can also import and use the pipeline directly from [corrective_rag.py](corrective_rag.py) in a Python script.

Example:

```python
from corrective_rag import app

result = app.invoke(
    {
        "question": "What is this document about?",
        "docs": [],
        "good_docs": [],
        "verdict": "",
        "reason": "",
        "strips": [],
        "kept_strips": [],
        "refined_context": "",
        "web_query": "",
        "web_docs": [],
        "answer": "",
    }
)

print(result["answer"])
```

---

## Example usage

Once the app is running:

1. Open the Streamlit UI in your browser
2. Enter a question such as:
   - “What is batch normalization?”
   - “Explain the key idea behind this document”
   - “What does the paper say about this topic?”
3. Click the run button
4. Review:
   - the final answer
   - the verdict
   - the reason for the retrieval decision
   - the refined context used to produce the answer

---

## Notes about the current implementation

The repository is functional, but a few implementation details are worth keeping in mind:

- The PDF path is currently hard-coded to a Windows-style path in [corrective_rag.py](corrective_rag.py). If you run this on another machine, you may need to update that path to match your local file location.
- The FAISS index is stored locally in [faiss_index](faiss_index), so the app can reuse previously built embeddings.
- The embedding model is loaded from Hugging Face and may need internet access or a local cache the first time it is used.
- The web-search step depends on Tavily being configured correctly.

---

## Troubleshooting

### Import errors

If you see missing package errors, make sure the Conda environment was created successfully:

```bash
conda env create -f environment.yml
conda activate rag-chatbot
```

### API key errors

If the app cannot connect to Gemini or Tavily, verify the `.env` file contains the correct keys.

### PDF not found

If the PDF cannot be loaded, make sure [lbdl.pdf](lbdl.pdf) is present in the project root and that the path in [corrective_rag.py](corrective_rag.py) is correct.

### Streamlit does not start

Make sure you are in the project directory and that the environment is activated before running:

```bash
streamlit run corrective-streamlit-rag.py
```

---

## Summary

This project is a practical example of a corrective RAG system. It combines retrieval, evaluation, optional web search, and answer generation into a single workflow that is more robust than a basic RAG setup because it verifies the retrieved context before using it.

If you want, this repository can also be extended with:

- a real chat history experience
- a file uploader instead of a fixed PDF
- a better UI with question history and source citations
- support for multiple documents and folders

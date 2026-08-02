import os
import streamlit as st
from dotenv import load_dotenv

from corrective_rag import app

load_dotenv()

st.set_page_config(page_title="Corrective RAG", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    .reportview-container, .App, .block-container {
        background: #000000 !important;
        color: #f8fafc !important;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stApp {
        background: #000000 !important;
    }
    .title-box {
        background: rgba(15, 23, 42, 0.95);
        border: 1px solid #1e293b;
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        margin-bottom: 1rem;
        color: #f8fafc;
    }
    .title-box h1, .title-box p {
        color: #f8fafc;
    }
    .card {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
        color: #f8fafc;
    }
    .css-1kyxreq .stButton>button {
        background-color: #111827;
        color: #f8fafc;
    }
    .css-1kyxreq .stButton>button:hover {
        background-color: #1f2937;
        color: #f8fafc;
    }
    .stTextInput>div>div>input {
        background: #020617;
        color: #f8fafc;
        border: 1px solid #334155;
    }
    .stTextArea>div>div>textarea {
        background: #020617;
        color: #f8fafc;
        border: 1px solid #334155;
    }
    .stMarkdown, .stText, .stCodeBlock {
        color: #f8fafc;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="title-box">
        <h1 style="margin-bottom:0.2rem;">🧠 Corrective RAG Assistant</h1>
        <p style="margin:0; color:#5b6472;">Ask a question and let the system retrieve, evaluate, and refine the answer using your documents and the web.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Setup")
    st.info("Make sure your Gemini API key is available in your .env file as GEMINI_API_KEY.")
    st.markdown("---")
    st.caption("This app uses your existing corrective RAG workflow under the hood.")

col1, col2 = st.columns([2, 1])
with col1:
    question = st.text_area(
        "Question",
        placeholder="Example: Batch normalization vs layer normalization",
        height=140,
    )

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("How it works")
    st.write("1. Retrieve relevant chunks")
    st.write("2. Judge relevance")
    st.write("3. Refine the context")
    st.write("4. Generate the final answer")
    st.markdown("</div>", unsafe_allow_html=True)

if st.button("Run RAG", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Processing your request..."):
            try:
                result = app.invoke(
                    {
                        "question": question,
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
            except Exception as e:
                st.error(f"Something went wrong: {e}")
            else:
                st.success("Completed")

                # Handle case where result itself is a string instead of dict
                if isinstance(result, str):
                    raw_answer = result
                    verdict = "Out of context / Web search"
                    reason = ""
                    web_query = ""
                    refined_context = ""
                else:
                    raw_answer = result.get("answer", "No answer generated.")
                    verdict = result.get("verdict", "")
                    reason = result.get("reason", "")
                    web_query = result.get("web_query", "")
                    refined_context = result.get("refined_context", "")

                # Clean extraction for list/dict/string formats
                if isinstance(raw_answer, list) and len(raw_answer) > 0:
                    first_item = raw_answer[0]
                    if isinstance(first_item, dict) and "text" in first_item:
                        final_answer = first_item["text"]
                    elif hasattr(first_item, "content"):
                        final_answer = first_item.content
                    else:
                        final_answer = str(first_item)
                elif isinstance(raw_answer, dict) and "text" in raw_answer:
                    final_answer = raw_answer["text"]
                else:
                    final_answer = str(raw_answer)

                st.markdown("### Final Answer")
                st.markdown(f"<div class='card'>\n\n{final_answer}\n\n</div>", unsafe_allow_html=True)

                st.markdown("---")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Verdict")
                    st.info(verdict)

                    if reason:
                        st.subheader("Reason")
                        st.write(reason)

                with col_b:
                    if web_query:
                        st.subheader("Web Query")
                        st.code(web_query)

                    if refined_context:
                        with st.expander("Refined Context"):
                            st.text(refined_context)
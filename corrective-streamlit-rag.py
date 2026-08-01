import os
import streamlit as st
from dotenv import load_dotenv

from corrective_rag import app

load_dotenv()

st.set_page_config(page_title="Corrective RAG", page_icon="🧠", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stApp {
        background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 100%);
    }
    .title-box {
        background: rgba(255,255,255,0.8);
        border: 1px solid #dfe9ff;
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 8px 24px rgba(31, 41, 55, 0.06);
        margin-bottom: 1rem;
    }
    .card {
        background: white;
        border: 1px solid #e6ebf5;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
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

                st.markdown("### Final Answer")
                st.markdown(f"<div class='card'>{result.get('answer', 'No answer generated.')}</div>", unsafe_allow_html=True)

                st.markdown("---")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.subheader("Verdict")
                    st.info(result.get("verdict", ""))

                    st.subheader("Reason")
                    st.write(result.get("reason", ""))

                with col_b:
                    st.subheader("Web Query")
                    st.code(result.get("web_query", ""))

                    with st.expander("Refined Context"):
                        st.text(result.get("refined_context", ""))

import streamlit as st
import os
import time
from rag_pipeline import process_uploaded_file, chunk_documents, setup_vector_store, create_rag_chain

# --- Page Config & Styling ---
st.set_page_config(
    page_title="NotebookLM Clone",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium look
st.markdown("""
<style>
    /* General styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Documenting Chunking Strategy Box */
    .chunk-strategy-box {
        background-color: rgba(224, 242, 254, 0.1);
        border: 1px solid #bae6fd;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.85rem;
    }
    
    /* Source citation styling */
    .source-box {
        background-color: rgba(243, 244, 246, 0.1);
        border-left: 3px solid #3b82f6;
        padding: 0.5rem 1rem;
        margin-top: 0.5rem;
        border-radius: 0 4px 4px 0;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# --- App State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Please upload a document in the sidebar, and I'll answer questions strictly based on its content."}]
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "doc_processed" not in st.session_state:
    st.session_state.doc_processed = False
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = ""

# --- Sidebar ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Google_%22G%22_logo.svg/120px-Google_%22G%22_logo.svg.png", width=60)
    st.title("NotebookLM Clone")
    st.markdown("💬 **Chat with your documents, grounded in their exact content.**")
    
    st.divider()
    
    # API Key (read from environment or Streamlit secrets)
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", "")

    st.divider()
    
    # File Upload
    st.caption("📄 Document Ingestion")
    uploaded_file = st.file_uploader("Upload Document (PDF/TXT)", type=["pdf", "txt"], help="Upload the file you want to query.")
    
    process_btn = st.button("Process Document Pipeline 🚀", type="primary", use_container_width=True)
    
    if process_btn:
        if not api_key:
            st.error("Missing GOOGLE_API_KEY. Set it in your environment or Streamlit secrets.")
        elif not uploaded_file:
            st.error("Please upload a document.")
        else:
            # God-level UI: Step-by-step progress updating
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # 1. Ingestion
                status_text.markdown("⏳ **1. Ingesting Document...**")
                documents = process_uploaded_file(uploaded_file)
                time.sleep(0.5) # Slight delay for visual UX
                progress_bar.progress(25)
                
                # 2. Chunking
                status_text.markdown(f"✂️ **2. Chunking {len(documents)} Pages...**")
                chunks = chunk_documents(documents)
                time.sleep(0.5)
                progress_bar.progress(50)
                
                # 3. Embedding & Vector Store
                status_text.markdown(f"🧠 **3. Embedding {len(chunks)} Chunks & Indexing to VectorDB...**")
                vector_store = setup_vector_store(chunks, api_key)
                time.sleep(0.5)
                progress_bar.progress(75)
                
                # 4. Create Chain
                status_text.markdown("🔗 **4. Establishing RAG Retrieval Chain...**")
                rag_chain = create_rag_chain(vector_store, api_key)
                time.sleep(0.5)
                progress_bar.progress(100)
                
                # Save to state
                st.session_state.rag_chain = rag_chain
                st.session_state.doc_processed = True
                st.session_state.current_file_name = uploaded_file.name
                
                # Reset chat logic
                st.session_state.messages = [{"role": "assistant", "content": f"Successfully indexed **{uploaded_file.name}**. I am ready to answer your questions!"}]
                
                status_text.success("✅ **Pipeline Complete! Embeddings stored in VectorDB.**")
                st.balloons()
            except Exception as e:
                st.error(f"Pipeline Error: {e}")

# --- Main Chat Interface ---
st.title("📚 Your Notebook")
if st.session_state.doc_processed:
    st.caption(f"Currently querying: **{st.session_state.current_file_name}** | VectorDB: **Chroma** | LLM: **Gemini**")
else:
    st.caption("Waiting for document ingestion...")

st.divider()

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display sources if present in the message
        if "sources" in msg and msg["sources"]:
            with st.expander("🔍 View Retrieved Context (VectorDB Results)"):
                for i, doc in enumerate(msg["sources"]):
                    st.markdown(f"<div class='source-box'><strong>Chunk {i+1} (Page {doc.metadata.get('page', 'N/A')}):</strong><br>{doc.page_content}</div>", unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Ask a question about the document..."):
    if not st.session_state.doc_processed:
        st.warning("⚠️ Please process a document in the sidebar first!")
        st.stop()
        
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving from VectorDB & Generating Answer..."):
            try:
                response = st.session_state.rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                source_documents = response["context"]
                
                st.markdown(answer)
                
                # Show sources explicitly for the assignment requirement "grounding"
                with st.expander("🔍 View Retrieved Context (VectorDB Results)"):
                    for i, doc in enumerate(source_documents):
                        st.markdown(f"<div class='source-box'><strong>Chunk {i+1} (Page {doc.metadata.get('page', 'N/A')}):</strong><br>{doc.page_content}</div>", unsafe_allow_html=True)
                        
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": source_documents  # Save sources to state so they persist on reload
                })
            except Exception as e:
                st.error(f"Error generating response: {e}")

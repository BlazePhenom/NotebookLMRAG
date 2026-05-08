import os
from typing import List, Any
import tempfile

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

def process_uploaded_file(uploaded_file) -> List[Document]:
    """
    Saves an uploaded file to a temporary location and loads it using LangChain loaders.
    """
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name

    documents = []
    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
        elif file_extension == '.txt':
            loader = TextLoader(temp_file_path, encoding='utf-8')
            documents = loader.load()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    return documents

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Chunks the documents using RecursiveCharacterTextSplitter.
    
    Strategy Documented:
    We use RecursiveCharacterTextSplitter because it tries to split on natural boundaries 
    (paragraphs, then sentences, then words) to keep semantically related pieces of text together.
    - chunk_size: 1000 characters. Large enough to capture context, small enough to fit in LLM prompts.
    - chunk_overlap: 200 characters. Ensures that concepts spanning across chunk boundaries are not lost.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def setup_vector_store(chunks: List[Document], api_key: str) -> Chroma:
    """
    Embeds the chunks and stores them in a local Chroma vector database.
    """
    # Use Google's embedding model for high-quality embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=api_key
    )
    
    # We use an ephemeral, in-memory Chroma instance for this session.
    # In a production app, we would persist this to disk.
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings
    )
    return vector_store

def create_rag_chain(vector_store: Chroma, api_key: str):
    """
    Creates a retrieval-augmented generation chain.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=api_key,
        temperature=0 # Low temperature for more factual, grounded answers
    )
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    # Define a strict system prompt to ensure grounding
    system_prompt = (
        "You are an expert Q&A assistant building a NotebookLM clone. "
        "Your task is to answer the user's question ONLY using the provided retrieved context chunks from their document. "
        "If the answer is NOT explicitly contained in the provided context, you must reply: "
        "'I am sorry, but the provided document does not contain information about this.' "
        "DO NOT use your outside knowledge, and DO NOT guess or hallucinate. "
        "Be detailed, but ensure every claim maps to the text.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
import tempfile
import os

load_dotenv()

st.set_page_config(
    page_title="Document Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 RAG Document Assistant")
st.markdown("Upload multiple PDFs and ask questions across all of them")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "uploaded_files_names" not in st.session_state:
    st.session_state.uploaded_files_names = []

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

def format_chat_history(chat_history):
    formatted = []
    for message in chat_history:
        if isinstance(message, HumanMessage):
            formatted.append(f"Human: {message.content}")
        else:
            formatted.append(f"Assistant: {message.content}")
    return "\n".join(formatted)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_answer(question, retriever, chat_history):
    llm = load_llm()

    # Step 1 — Rephrase question using chat history
    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the conversation history and the follow-up question, "
         "rephrase the follow-up question to be a standalone question. "
         "If no history exists, return the question as is. "
         "Return ONLY the rephrased question, nothing else.\n\n"
         "Chat History:\n{chat_history}"),
        ("human", "{question}")
    ])

    rephrase_chain = rephrase_prompt | llm | StrOutputParser()

    standalone_question = rephrase_chain.invoke({
        "question": question,
        "chat_history": format_chat_history(chat_history)
    })

    # Step 2 — Retrieve relevant documents
    docs = retriever.invoke(standalone_question)
    context = format_docs(docs)

    # Step 3 — Answer based on context
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful document assistant. "
         "Answer the question based ONLY on the context provided. "
         "If the answer is not in the context, say: "
         "'I could not find this information in the documents.' "
         "When relevant, mention which document the information "
         "comes from. Be clear and concise.\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])

    answer_chain = answer_prompt | llm | StrOutputParser()

    answer = answer_chain.invoke({
        "question": standalone_question,
        "context": context
    })

    return answer, docs

# Sidebar
with st.sidebar:
    st.header("📁 Upload Documents")

    # Multiple file uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            st.markdown(f"- 📄 {f.name}")

        if st.button("Process All Documents", type="primary"):
            with st.spinner(f"Processing {len(uploaded_files)} document(s)..."):

                all_chunks = []
                processed_names = []
                failed_names = []

                progress = st.progress(0)

                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".pdf"
                        ) as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name

                        loader = PyPDFLoader(tmp_path)
                        documents = loader.load()

                        # Add filename to each document's metadata
                        for doc in documents:
                            doc.metadata["source_file"] = uploaded_file.name

                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200
                        )
                        chunks = splitter.split_documents(documents)
                        all_chunks.extend(chunks)
                        processed_names.append(uploaded_file.name)
                        os.unlink(tmp_path)

                    except Exception as e:
                        failed_names.append(uploaded_file.name)

                    progress.progress((i + 1) / len(uploaded_files))

                if all_chunks:
                    # Build ONE vector store from ALL documents
                    embeddings = load_embeddings()
                    vectorstore = Chroma.from_documents(
                        documents=all_chunks,
                        embedding=embeddings
                    )

                    st.session_state.retriever = vectorstore.as_retriever(
                        search_kwargs={"k": 4}
                    )
                    st.session_state.document_processed = True
                    st.session_state.chat_history = []
                    st.session_state.uploaded_files_names = processed_names

                    st.success(
                        f"✅ {len(processed_names)} document(s) processed successfully"
                    )

                    if failed_names:
                        st.warning(
                            f"⚠️ Failed to process: {', '.join(failed_names)}"
                        )

    # Show currently loaded documents
    if st.session_state.document_processed:
        st.markdown("---")
        st.markdown("### 📂 Loaded Documents")
        for name in st.session_state.uploaded_files_names:
            st.markdown(f"✅ {name}")

        if st.button("🗑️ Clear All and Start Over"):
            st.session_state.document_processed = False
            st.session_state.chat_history = []
            st.session_state.retriever = None
            st.session_state.uploaded_files_names = []
            st.rerun()

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("""
    1. Upload one or more PDF files
    2. Click Process All Documents
    3. Ask questions across all files
    4. AI finds answers from any document
    """)

# Main chat interface
if st.session_state.document_processed:
    st.markdown("### 💬 Ask questions across all your documents")

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)

    user_question = st.chat_input(
        "Ask anything about your documents..."
    )

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching across all documents..."):
                answer, source_docs = get_answer(
                    user_question,
                    st.session_state.retriever,
                    st.session_state.chat_history
                )

                st.write(answer)

                if source_docs:
                    with st.expander("📚 Sources"):
                        for i, doc in enumerate(source_docs):
                            source_file = doc.metadata.get(
                                'source_file', 'Unknown'
                            )
                            page = doc.metadata.get('page', 'N/A')
                            if isinstance(page, int):
                                page = page + 1
                            st.markdown(
                                f"**Source {i+1}** — 📄 {source_file} "
                                f"(Page {page})"
                            )
                            st.markdown(doc.page_content[:300] + "...")
                            st.markdown("---")

                st.session_state.chat_history.append(
                    HumanMessage(content=user_question)
                )
                st.session_state.chat_history.append(
                    AIMessage(content=answer)
                )

else:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("###")
        st.info("👈 Upload your PDF files in the sidebar to get started")
        st.markdown("### What you can do")
        st.markdown("""
        - 📋 **Summarize** multiple documents at once
        - 🔍 **Search across** all uploaded files simultaneously
        - ❓ **Ask questions** in plain language
        - 📌 **See which document** each answer comes from
        """)



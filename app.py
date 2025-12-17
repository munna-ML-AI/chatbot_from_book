from pathlib import Path
import traceback
import streamlit as st
import time
import random

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

# -------------------------
# Config
# -------------------------
FAISS_DIR = "faiss_index"
FAISS_INDEX_PATH = "index.faiss"
FAISS_PKL_PATH = "index.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Initialize session state for vector stores
if "faiss_db" not in st.session_state:
    st.session_state.faiss_db = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embed" not in st.session_state:
    st.session_state.embed = None

# -------------------------
# Initialize embeddings
# -------------------------
def get_embeddings():
    """Initialize embeddings model"""
    try:
        return SentenceTransformerEmbeddings(model_name=EMBED_MODEL)
    except Exception as e:
        print(f"Error initializing embeddings: {e}")
        # Try alternative import
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        except ImportError:
            raise Exception(f"Could not load embeddings. Please install: pip install sentence-transformers")

# -------------------------
# Load pre-built FAISS index
# -------------------------
def load_faiss_index():
    """Load pre-existing FAISS index and pkl file"""
    try:
        # Check if index files exist
        index_path = Path(FAISS_DIR) / FAISS_INDEX_PATH if FAISS_DIR else Path(FAISS_INDEX_PATH)
        pkl_path = Path(FAISS_DIR) / FAISS_PKL_PATH if FAISS_DIR else Path(FAISS_PKL_PATH)
        
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")
        if not pkl_path.exists():
            raise FileNotFoundError(f"FAISS pkl file not found: {pkl_path}")
        
        print(f"Loading FAISS index from: {index_path}")
        print(f"Loading FAISS pkl from: {pkl_path}")
        
        # Initialize embeddings first
        embed = get_embeddings()
        st.session_state.embed = embed
        
        # Test embeddings
        test_text = "This is a test."
        test_embedding = embed.embed_query(test_text)
        print(f"Embeddings initialized. Test vector dimension: {len(test_embedding)}")
        
        # Load the FAISS index WITH embeddings
        faiss_db = FAISS.load_local(
            folder_path=FAISS_DIR if FAISS_DIR else ".",
            embeddings=embed,
            allow_dangerous_deserialization=True
        )
        
        print(f"FAISS index loaded successfully!")
        print(f"Number of documents in index: {faiss_db.index.ntotal if hasattr(faiss_db, 'index') else 'Unknown'}")
        
        return faiss_db
        
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        traceback.print_exc()
        raise

# -------------------------
# Query Functions (FAISS only)
# -------------------------
def search_faiss(vdb, query, k=5):
    """Search FAISS vector database"""
    return vdb.similarity_search(query, k=k)

# -------------------------
# Initialize the system
# -------------------------
def initialize_system():
    """Initialize the FAISS search system"""
    try:
        if st.session_state.faiss_db is None:
            with st.spinner("Loading FAISS search index..."):
                st.write("Loading pre-built FAISS index...")
                
                # Load FAISS index
                faiss_db = load_faiss_index()
                st.session_state.faiss_db = faiss_db
                
                st.success("FAISS search system initialized successfully!")
                st.info(f"Index contains approximately {faiss_db.index.ntotal if hasattr(faiss_db, 'index') else 'unknown'} documents")
                
        return True
    except Exception as e:
        st.error(f"Failed to initialize FAISS system: {e}")
        traceback.print_exc()
        return False

# -------------------------
# Streamlit App
# -------------------------
# Set page configuration
st.set_page_config(
    page_title="Document Q&A with FAISS",
    page_icon=":mag:",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title
st.title("FAISS Search created Mehedi Hasan Monna")
st.markdown("Search your documents using pre-built FAISS index!")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    # Initialize system button
    if st.button("Load FAISS Index", use_container_width=True):
        if initialize_system():
            st.success("FAISS index loaded successfully!")
        else:
            st.error("Failed to load FAISS index. Check console for details.")
    
    # Show system status
    st.divider()
    st.subheader("System Status")
    
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.metric("FAISS DB", "Ready" if st.session_state.faiss_db else "Not Loaded")
    with status_col2:
        st.metric("Embeddings", "Ready" if st.session_state.embed else "Not Loaded")
    
    # Display index info if loaded
    if st.session_state.faiss_db:
        try:
            if hasattr(st.session_state.faiss_db, 'index'):
                num_docs = st.session_state.faiss_db.index.ntotal
                st.metric("Documents in Index", num_docs)
        except:
            pass
    
    # Search settings
    st.divider()
    st.subheader("Search Settings")
    
    num_results = st.slider(
        "Number of results to show",
        min_value=1,
        max_value=20,
        value=5
    )
    
    # Streaming speed control
    streaming_speed = st.slider(
        "Streaming Speed (words per second)",
        min_value=1,
        max_value=20,
        value=10,
        help="Control how fast words appear"
    )
    
    # Search type
    search_type = st.selectbox(
        "Search Type",
        ["Similarity Search", "Similarity Search with Score"],
        index=0
    )
    
    # Clear chat button
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.markdown("""
    ### How it works:
    1. Load the pre-built FAISS index
    2. Ask questions about your documents
    3. Get responses with relevant document chunks
    4. Adjust search settings in sidebar
    
    ### Requirements:
    - `index.faiss` - FAISS vector index file
    - `index.pkl` - Document metadata file
    """)

# Display chat messages
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "chunk_results" in message:
                # Display the main response
                st.markdown(message["content"])
                
                # Display chunk results in expanders
                with st.expander(f"View {len(message['chunk_results'])} relevant document chunks"):
                    for i, chunk in enumerate(message["chunk_results"]):
                        metadata = chunk.metadata
                        page = metadata.get('page', 'N/A')
                        source = metadata.get('source', 'Unknown')
                        
                        st.markdown(f"**Chunk {i+1}** (Page {page}, Source: {source})")
                        if 'similarity_score' in metadata:
                            st.caption(f"Similarity Score: {metadata['similarity_score']}")
                        
                        # Display chunk content
                        chunk_text = chunk.page_content
                        if len(chunk_text) > 300:
                            st.text(chunk_text[:300] + "...")
                        else:
                            st.text(chunk_text)
                        st.divider()
            else:
                st.markdown(message["content"])

# Function to search documents using FAISS
def search_documents(query, k=5, search_type="Similarity Search"):
    """Search documents for relevant chunks using FAISS"""
    try:
        if st.session_state.faiss_db is None:
            st.warning("FAISS index not loaded. Please load it first.")
            return []
            
        if search_type == "Similarity Search with Score":
            results = st.session_state.faiss_db.similarity_search_with_score(query, k=k)
            # Convert to same format as similarity_search
            chunks = [doc for doc, score in results]
            # Add scores to metadata for display
            for i, (doc, score) in enumerate(results):
                chunks[i].metadata['similarity_score'] = f"{score:.4f}"
            return chunks
        else:
            return st.session_state.faiss_db.similarity_search(query, k=k)
            
    except Exception as e:
        st.error(f"Search error: {e}")
        traceback.print_exc()
        return []

# Function to generate AI response with chunk context
def generate_ai_response(user_input, chunk_results, search_type):
    """Generate AI response using chunk context"""
    if not chunk_results:
        return f"I received your question about '{user_input}', but I couldn't find relevant information in the document index."
    
    # Extract chunk content
    chunk_context = ""
    for i, chunk in enumerate(chunk_results):
        page = chunk.metadata.get('page', 'unknown')
        source = chunk.metadata.get('source', 'document')
        score = chunk.metadata.get('similarity_score', 'N/A')
        
        # Truncate chunk text for display
        chunk_text = chunk.page_content
        display_text = chunk_text[:400] + "..." if len(chunk_text) > 400 else chunk_text
        
        if search_type == "Similarity Search with Score" and score != 'N/A':
            chunk_context += f"\n\n**Chunk {i+1}** (Page {page}, Score: {score}):\n{display_text}"
        else:
            chunk_context += f"\n\n**Chunk {i+1}** (Page {page}):\n{display_text}"
    
    # Create response based on chunks
    base_responses = [
        f"Based on the document index, here's what I found regarding '{user_input}':",
        f"Regarding your question about '{user_input}', the document mentions:",
        f"Here's relevant information from the documents about '{user_input}':",
        f"The document index provides these insights about '{user_input}':",
        f"Based on the FAISS search, here's what the documents say about '{user_input}':"
    ]
    
    response = random.choice(base_responses) + chunk_context
    
    # Add a summary
    if len(chunk_results) > 1:
        response += f"\n\nFound {len(chunk_results)} relevant document chunks."
    
    return response

# Function to stream text word by word
def stream_text(text, words_per_second=10):
    """Stream text word by word with animation effect"""
    words = text.split()
    
    # Create a placeholder for the streaming text
    message_placeholder = st.empty()
    
    # Initialize empty string to build up
    displayed_text = ""
    
    # Stream words one by one
    for word in words:
        displayed_text += word + " "
        
        # Update the placeholder with current text
        message_placeholder.markdown(displayed_text + "▌")
        
        # Calculate delay based on words per second
        delay = 1.0 / words_per_second
        
        # Add slight randomness to make it feel more natural
        delay *= random.uniform(0.8, 1.2)
        
        # Wait before showing next word
        time.sleep(delay)
    
    # Remove the cursor after streaming is complete
    message_placeholder.markdown(displayed_text)
    
    return displayed_text.strip()

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Check if system is initialized
    if st.session_state.faiss_db is None:
        st.warning("Please load the FAISS index first using the button in the sidebar.")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Search for relevant chunks
    with st.spinner("Searching document index..."):
        chunk_results = search_documents(prompt, k=num_results, search_type=search_type)
    
    # Display assistant response with streaming effect
    with st.chat_message("assistant"):
        # Generate AI response with chunk context
        if chunk_results:
            ai_response = generate_ai_response(prompt, chunk_results, search_type)
            
            # Stream the response word by word
            full_response = stream_text(ai_response, words_per_second=streaming_speed)
        else:
            full_response = f"I couldn't find any relevant information about '{prompt}' in the document index. Try rephrasing your question or asking about a different topic."
            st.markdown(full_response)
    
    # Add assistant response to chat history with chunk results
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "chunk_results": chunk_results
    })

# Display index file information
with st.sidebar:
    st.divider()
    st.subheader("Index Files")
    
    # Check for index files
    index_files_exist = False
    if FAISS_DIR:
        index_path = Path(FAISS_DIR) / FAISS_INDEX_PATH
        pkl_path = Path(FAISS_DIR) / FAISS_PKL_PATH
    else:
        index_path = Path(FAISS_INDEX_PATH)
        pkl_path = Path(FAISS_PKL_PATH)
    
    if index_path.exists():
        index_size = index_path.stat().st_size / (1024 * 1024)  # Size in MB
        st.success(f"✓ {FAISS_INDEX_PATH} found ({index_size:.2f} MB)")
        index_files_exist = True
    else:
        st.error(f"✗ {FAISS_INDEX_PATH} not found")
    
    if pkl_path.exists():
        pkl_size = pkl_path.stat().st_size / (1024 * 1024)  # Size in MB
        st.success(f"✓ {FAISS_PKL_PATH} found ({pkl_size:.2f} MB)")
        index_files_exist = True
    else:
        st.error(f"✗ {FAISS_PKL_PATH} not found")
    
    if not index_files_exist:
        st.warning("""
        **Required files missing:**
        1. `index.faiss` - FAISS vector index
        2. `index.pkl` - Document metadata
        
        Place these files in the working directory or set `FAISS_DIR` in the code.
        """)

# Display system requirements
with st.sidebar:
    st.divider()
    st.subheader("Dependencies")
    
    st.code("""
# Required packages:
pip install streamlit
pip install langchain-community
pip install faiss-cpu  # or faiss-gpu
pip install sentence-transformers
""")

# Optional: Add some styling
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .stChatInput {
        position: fixed;
        bottom: 3rem;
        width: calc(100% - 3rem);
    }
    
    div[data-testid="stVerticalBlock"] > div:has(> .element-container > .stChatInput) {
        position: fixed;
        bottom: 0;
        width: calc(100% - 3rem);
        background-color: white;
        padding: 1rem 0;
        z-index: 999;
    }
    
    /* Style for chunk expanders */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Display message count in sidebar

st.sidebar.metric("Chat Messages", len([m for m in st.session_state.messages if m["role"] == "user"]))

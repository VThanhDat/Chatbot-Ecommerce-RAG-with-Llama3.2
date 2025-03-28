import streamlit as st
from streamlit import session_state
import time
import base64
import os
from vectors import EmbeddingsManager  # Import the EmbeddingsManager class
from chatbotRAG import ChatbotManager     # Import the ChatbotManager class
from chatbotEcommerce import EcommerceChatbotManager  # Import the EcommerceChatbotManager class
from dotenv import load_dotenv

load_dotenv()
FINE_TUNE_MODEL = os.getenv("FINE_TUNE_MODEL")

# Function to display the PDF of a given file
def displayPDF(file):
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Initialize session_state variables if not already present
if 'temp_pdf_path' not in st.session_state:
    st.session_state['temp_pdf_path'] = None

if 'chatbot_manager' not in st.session_state:
    st.session_state['chatbot_manager'] = None

if 'ecommerce_chatbot' not in st.session_state:
    st.session_state['ecommerce_chatbot'] = None

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'ecommerce_messages' not in st.session_state:
    st.session_state['ecommerce_messages'] = []

# Set the page configuration
st.set_page_config(
    page_title="Document Buddy App",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
with st.sidebar:
    st.image("llama3_2-blog-V2-groqMeta.webp", use_column_width=True)
    st.markdown("### Chatbot Document Assistant")
    st.markdown("---")
    menu = ["🏠 Home", "🤖 Chatbot RAG","🤖 Chatbot E-Commerce"]
    choice = st.selectbox("Navigate", menu)

# Home Page
if choice == "🏠 Home":
    st.title("📄 Question Answering from document")
    st.markdown("""
    **Technical use: Llama 3.2:3b, BGE Embeddings, and Qdrant.**
    - **Upload Documents**: Easily upload your PDF documents.
    - **Summarize**: Get concise summaries of your documents.
    - **Chat**: Interact with your documents through our intelligent chatbot.
    """)

# Chatbot RAG Page
elif choice == "🤖 Chatbot RAG":
    st.title("🤖 Chatbot Interface (Llama 3.2 RAG 🦙)")
    st.markdown("---")
    
    # Create three columns
    col1, col2, col3 = st.columns(3)

    # Column 1: File Uploader and Preview
    with col1:
        st.header("📂 Upload Document")
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded_file is not None:
            st.success("📄 File Uploaded Successfully!")
            # Display file name and size
            st.markdown(f"**Filename:** {uploaded_file.name}")
            st.markdown(f"**File Size:** {uploaded_file.size} bytes")
            
            # Display PDF preview using displayPDF function
            st.markdown("### 📖 PDF Preview")
            displayPDF(uploaded_file)
            
            # Save the uploaded file to a temporary location
            temp_pdf_path = "temp.pdf"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Store the temp_pdf_path in session_state
            st.session_state['temp_pdf_path'] = temp_pdf_path

    # Column 2: Create Embeddings
    with col2:
        st.header("🧠 Embeddings")
        create_embeddings = st.checkbox("✅ Create Embeddings")
        if create_embeddings:
            if st.session_state['temp_pdf_path'] is None:
                st.warning("⚠️ Please upload a PDF first.")
            else:
                try:
                    # Initialize the EmbeddingsManager
                    embeddings_manager = EmbeddingsManager(
                        model_name="BAAI/bge-small-en",
                        device="cpu",
                        encode_kwargs={"normalize_embeddings": True},
                        qdrant_url="http://localhost:6333",
                        collection_name="vector_db"
                    )
                    
                    with st.spinner("🔄 Embeddings are in process..."):
                        # Create embeddings
                        result = embeddings_manager.create_embeddings(st.session_state['temp_pdf_path'])
                        time.sleep(1)  # Optional: To show spinner for a bit longer
                    st.success(result)
                    
                    # Initialize the ChatbotManager after embeddings are created
                    if st.session_state['chatbot_manager'] is None:
                        st.session_state['chatbot_manager'] = ChatbotManager(
                            model_name="BAAI/bge-small-en",
                            device="cpu",
                            encode_kwargs={"normalize_embeddings": True},
                            llm_model="llama3.2:3b",
                            llm_temperature=0.7,
                            qdrant_url="http://localhost:6333",
                            collection_name="vector_db"
                        )
                    
                except FileNotFoundError as fnf_error:
                    st.error(fnf_error)
                except ValueError as val_error:
                    st.error(val_error)
                except ConnectionError as conn_error:
                    st.error(conn_error)
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")

    # Column 3: Chatbot Interface
    with col3:
        st.header("💬 Chat with Document")
        
        if st.session_state['chatbot_manager'] is None:
            st.info("🤖 Please upload a PDF and create embeddings to start chatting.")
        else:
            # Display existing messages
            for msg in st.session_state['messages']:
                st.chat_message(msg['role']).markdown(msg['content'])

            # User input
            if user_input := st.chat_input("Type your message here..."):
                # Display user message
                st.chat_message("user").markdown(user_input)
                st.session_state['messages'].append({"role": "user", "content": user_input})

                with st.spinner("🤖 Responding..."):
                    try:
                        # Get the chatbot response using the ChatbotManager
                        answer = st.session_state['chatbot_manager'].get_response(user_input)
                        time.sleep(1)  # Simulate processing time
                    except Exception as e:
                        answer = f"⚠️ An error occurred while processing your request: {e}"
                
                # Display chatbot message
                st.chat_message("assistant").markdown(answer)
                st.session_state['messages'].append({"role": "assistant", "content": answer})

# Chatbot E-Commerce Page
elif choice == "🤖 Chatbot E-Commerce":
    st.title("🤖 Chatbot Interface E-Commerce (Llama 3.2 🦙)")
    st.markdown("---")

    # Initialize the EcommerceChatbotManager if not already initialized
    if st.session_state['ecommerce_chatbot'] is None:
        with st.spinner("🔄 Loading E-commerce Chatbot..."):
            try:
                if not FINE_TUNE_MODEL:
                    raise ValueError("Biến môi trường FINE_TUNE_MODEL chưa được thiết lập!")
                else:
                    st.session_state['ecommerce_chatbot'] = EcommerceChatbotManager(
                        fine_tune_model=FINE_TUNE_MODEL
                    )
                    st.success("✅ E-commerce Chatbot initialized successfully!")
            except Exception as e:
                st.error(f"⚠️ An error occurred while initializing the E-commerce Chatbot: {e}")
                import traceback
                st.error(traceback.format_exc())

    # Create a columns layout
    col2 = st.columns([2])[0]  # Lấy đối tượng cột thay vì danh sách


    # Column 2: Chat Interface
    with col2:
        st.header("💬 Chat with E-commerce Assistant")

        if st.session_state['ecommerce_chatbot'] is None:
            st.info("🤖 Please wait while the E-commerce chatbot is being initialized.")
        else:
            # Display existing messages
            for msg in st.session_state['ecommerce_messages']:
                st.chat_message(msg['role']).markdown(msg['content'])

            # User input
            if user_input := st.chat_input("How can I help you with your shopping today?"):
                st.chat_message("user").markdown(user_input)
                st.session_state['ecommerce_messages'].append({"role": "user", "content": user_input})

                with st.spinner("🤖 Responding..."):
                    try:
                        answer = st.session_state['ecommerce_chatbot'].chat(user_input)
                        time.sleep(1)
                    except Exception as e:
                        answer = f"⚠️ An error occurred while processing your request: {e}"

                # Display chatbot response
                st.chat_message("assistant").markdown(answer)
                st.session_state['ecommerce_messages'].append({"role": "assistant", "content": answer})

# Footer
st.markdown("---")

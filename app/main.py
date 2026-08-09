# ============================================================================
# main.py — Flask application for the Knowledge Intelligence System (RAG)
# ============================================================================

# --- Standard library imports ------------------------------------------------
import os          # Provides functions for interacting with the operating system (file paths, env vars)
import tempfile    # Creates temporary files/directories that are automatically cleaned up
import logging     # Python's built-in logging framework for debug/info/error messages

# --- Third-party imports -----------------------------------------------------
from flask import Flask, request, render_template, jsonify  # Flask web framework essentials
#   Flask        → The main application class that handles routing and requests
#   request      → Gives access to incoming HTTP request data (form fields, files, JSON)
#   render_template → Renders an HTML template (Jinja2) and returns it as a response
#   jsonify      → Converts a Python dict into a proper JSON HTTP response

from langchain_community.document_loaders import PyPDFLoader, TextLoader  # Document loaders
#   PyPDFLoader  → Reads a PDF file and converts each page into a LangChain Document object
#   TextLoader   → Reads a plain text file and converts it into a LangChain Document object

from langchain_text_splitters import RecursiveCharacterTextSplitter  # Text chunking
#   RecursiveCharacterTextSplitter → Splits long documents into smaller overlapping chunks
#   so they fit within the LLM's context window and produce better embeddings

# --- Local project imports ---------------------------------------------------
from config import Config                        # Loads environment variables (API keys, bucket name, etc.)
from model.vectorStore import VectorStore        # Wrapper around ChromaDB for storing/retrieving embeddings
from service.llmService import LLMService        # RAG chain: reformulates questions + generates answers
from service.storageService import StorageService  # Uploads files to AWS S3 for backup/storage

# ============================================================================
# Logging Setup
# ============================================================================
logging.basicConfig(level=logging.INFO)  # Set the minimum log level to INFO (shows INFO, WARNING, ERROR)
logger = logging.getLogger(__name__)     # Create a logger named after this module ("main")

# ============================================================================
# Flask App Initialisation
# ============================================================================
app = Flask(                    # Create the Flask application instance
    __name__,                   # Tells Flask where to find resources relative to this file
    template_folder="template"  # Our HTML templates live in "template/" (not the default "templates/")
)

# ============================================================================
# Service Initialisation (runs once when the app starts)
# ============================================================================
vector_store = VectorStore(Config.VECTOR_DB)   # Initialise ChromaDB at the configured persist directory
storage_service = StorageService()             # Initialise the S3 client with AWS credentials from Config
llm_service = LLMService(vector_store)         # Initialise the RAG chain, wiring it to our vector store


# ============================================================================
# Helper Function — Document Processing
# ============================================================================
def process_document(file):
    """
    Takes an uploaded file, saves it to a temp directory, loads its content
    using the appropriate LangChain loader, splits it into chunks, and
    returns the list of text chunks.
    """
    temp_dir = tempfile.mkdtemp()                            # Create a temporary directory on disk
    temp_path = os.path.join(temp_dir, file.filename)        # Build full path: /tmp/xyz/myfile.pdf
    file.save(temp_path)                                     # Save the uploaded file to that path

    try:
        # Pick the right loader based on file extension
        if file.filename.lower().endswith('.pdf'):            # If it's a PDF...
            loader = PyPDFLoader(temp_path)                   #   → use PyPDFLoader to parse it
        else:                                                 # Otherwise (assume .txt)...
            loader = TextLoader(temp_path)                    #   → use TextLoader to read it

        documents = loader.load()                             # Load file contents into Document objects

        # Split the documents into smaller, overlapping chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,     # Each chunk will be at most 1000 characters
            chunk_overlap=200    # Adjacent chunks share 200 characters for context continuity
        )
        text_chunks = text_splitter.split_documents(documents)  # Perform the splitting

        return text_chunks                                    # Return the list of chunk Documents

    finally:
        # Clean up: remove the temp file and directory even if an error occurred
        if os.path.exists(temp_path):
            os.remove(temp_path)     # Delete the temporary file
        os.rmdir(temp_dir)           # Delete the temporary directory


# ============================================================================
# Route: Home Page
# ============================================================================
@app.route('/')                         # Register this function for GET requests to "/"
def index():
    """Serve the main HTML page."""
    return render_template('index.html')  # Render and return the index.html template


# ============================================================================
# Route: Upload a Document
# ============================================================================
@app.route('/upload', methods=['POST'])   # Register this function for POST requests to "/upload"
def upload_document():
    """
    Accepts a file upload via multipart form data.
    Processes the file → splits into chunks → stores in S3 → indexes in ChromaDB.
    """
    # --- Validate: check that a file was included in the request ---------------
    if 'file' not in request.files:                          # No 'file' field in the form
        logger.warning("No file in request")
        return jsonify({'error': 'No file provided'}), 400   # 400 = Bad Request

    file = request.files['file']                             # Extract the file object from the request

    if file.filename == '':                                  # User submitted the form without selecting a file
        logger.warning("Empty filename")
        return jsonify({'error': 'No file selected'}), 400

    # --- Validate: only accept .txt and .pdf -----------------------------------
    if not file.filename.lower().endswith(('.txt', '.pdf')):  # Check file extension
        logger.warning(f"Unsupported file type: {file.filename}")
        return jsonify({'error': 'Only .txt and .pdf files are supported'}), 400

    logger.info(f"Processing file: {file.filename}")

    # --- Step 1: Parse and chunk the document ----------------------------------
    try:
        text_chunks = process_document(file)                 # Call our helper function above
        logger.info(f"Document split into {len(text_chunks)} chunks")
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return jsonify({'error': f'Error processing document: {str(e)}'}), 500

    # --- Step 2: Backup the original file to S3 --------------------------------
    try:
        file.seek(0)                                         # Reset file pointer (process_document already read it)
        storage_service.upload_file(file, file.filename)     # Upload raw file to S3 bucket
        logger.info("File uploaded to S3")
    except Exception as e:
        logger.error(f"Error uploading to S3: {e}")
        return jsonify({'error': f'Error uploading to S3: {str(e)}'}), 500

    # --- Step 3: Index the text chunks into ChromaDB ---------------------------
    try:
        vector_store.add_documents(text_chunks)              # Embed chunks via OpenAI → store in ChromaDB
        logger.info("Documents indexed in vector store")
    except Exception as e:
        logger.error(f"Error indexing in vector store: {e}")
        return jsonify({'error': f'Error indexing in vector store: {str(e)}'}), 500

    # --- Success ---------------------------------------------------------------
    return jsonify({
        'message': 'File uploaded and processed successfully',
        'chunks_processed': len(text_chunks)                 # Tell the user how many chunks were created
    }), 200


# ============================================================================
# Route: Ask a Question (Query the Knowledge Base)
# ============================================================================
@app.route('/query', methods=['POST'])   # Register this function for POST requests to "/query"
def query():
    """
    Accepts a JSON body with a 'question' field.
    Runs the full RAG pipeline: retrieve relevant chunks → generate answer.
    Returns the AI-generated response.
    """
    data = request.get_json()                                # Parse the JSON body from the request

    # Validate that a question was provided
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400

    try:
        # Run the RAG chain:
        #   1. Reformulate the question using chat history (context-aware)
        #   2. Retrieve the most relevant document chunks from ChromaDB
        #   3. Generate an answer grounded in the retrieved context
        response = llm_service.get_response(data['question'])

        return jsonify({'response': response}), 200          # Return the answer as JSON
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Application Entry Point
# ============================================================================
if __name__ == '__main__':
    # Start the Flask development server
    # - host='0.0.0.0' → listen on all network interfaces (accessible from other machines)
    # - port=8080       → serve on port 8080
    # - debug=True      → auto-reload on code changes + detailed error pages
    #   ⚠️  Do NOT use debug=True in production — it exposes an interactive debugger
    app.run(host='0.0.0.0', port=8080, debug=True)
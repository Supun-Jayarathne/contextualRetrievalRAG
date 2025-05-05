import os
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from typing import List, Iterator, Optional, Dict, Any
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from docx import Document as DocxDocument
from io import BytesIO
import logging
import time
import hashlib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DocxLoader(BaseLoader):
    """Improved DOCX loader with better error handling"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load and return documents from the DOCX file"""
        try:
            doc = DocxDocument(self.file_path)

            # Extract paragraphs with basic formatting preservation
            paragraphs = []
            for para in doc.paragraphs:
                if para.text.strip():
                    # Include heading style information if available
                    if para.style and "Heading" in para.style.name:
                        paragraphs.append(f"{para.style.name}: {para.text}")
                    else:
                        paragraphs.append(para.text)

            # Join paragraphs with newlines
            text = "\n".join(paragraphs)

            # Extract basic metadata
            metadata = {
                "source": self.file_path,
                "filename": os.path.basename(self.file_path),
                "date_processed": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            return [Document(page_content=text, metadata=metadata)]
        except Exception as e:
            logger.error(f"Error loading {self.file_path}: {str(e)}")
            # Return empty document with error info in metadata
            return [Document(
                page_content="",
                metadata={
                    "source": self.file_path,
                    "error": str(e),
                    "status": "failed"
                }
            )]

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load documents"""
        yield from self.load()


class RAGApplication:
    """Main RAG application class to encapsulate functionality"""

    def __init__(self, persist_directory: str = "./chroma_db_docx"):
        """Initialize the RAG application"""
        self.persist_directory = persist_directory
        self.embeddings = None
        self.llm = None
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.memory = ConversationBufferWindowMemory(k=5, return_messages=True)

        # Load environment variables
        load_dotenv()

        # Initialize components
        self._initialize_azure_components()
        self._initialize_vectorstore()
        self._initialize_rag_pipeline()

        logger.info("RAG application initialized successfully")

    def _initialize_azure_components(self):
        """Initialize Azure OpenAI components"""
        try:
            # Get environment variables with fallbacks
            embedding_deployment = os.getenv(
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
            chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")

            # Check for required configuration
            if not all([embedding_deployment, chat_deployment, endpoint, api_key]):
                missing = []
                if not embedding_deployment:
                    missing.append("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
                if not chat_deployment:
                    missing.append("AZURE_OPENAI_CHAT_DEPLOYMENT")
                if not endpoint:
                    missing.append("AZURE_OPENAI_ENDPOINT")
                if not api_key:
                    missing.append("AZURE_OPENAI_API_KEY")

                raise ValueError(
                    f"Missing required Azure OpenAI configuration: {', '.join(missing)}")

            logger.info(
                f"Initializing Azure OpenAI with: Embedding deployment: {embedding_deployment}, Chat deployment: {chat_deployment}")

            # Initialize embeddings
            try:
                self.embeddings = AzureOpenAIEmbeddings(
                    azure_deployment=embedding_deployment,
                    openai_api_version=api_version,
                    azure_endpoint=endpoint,
                    api_key=api_key
                )
                # Test the embeddings with a simple query to catch errors early
                test_result = self.embeddings.embed_query("Test embedding")
                if not test_result or len(test_result) < 1:
                    raise ValueError(
                        "Embedding test failed - returned empty result")
                logger.info(
                    f"Embeddings successfully initialized and tested ({len(test_result)} dimensions)")
            except Exception as e:
                error_msg = str(e)
                if "OperationNotSupported" in error_msg:
                    raise ValueError(
                        f"The model in deployment '{embedding_deployment}' doesn't support embeddings. "
                        f"Please use a text-embedding model like 'text-embedding-ada-002'. Error: {error_msg}"
                    )
                else:
                    raise ValueError(
                        f"Failed to initialize embeddings: {error_msg}"
                        f"Failed to initialize embedding_deployment: {embedding_deployment}"
                        f"Failed to initialize endpoint: {endpoint}"
                        f"Failed to initialize api_key: {api_key}"
                    )

            # Initialize LLM
            try:
                self.llm = AzureChatOpenAI(
                    azure_deployment=chat_deployment,
                    openai_api_version=api_version,
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    temperature=0.1,  # Slightly increased for more natural responses
                    max_tokens=1000
                )
                logger.info(f"Chat model successfully initialized")
            except Exception as e:
                raise ValueError(f"Failed to initialize chat model: {str(e)}")

            logger.info("Azure OpenAI components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure components: {str(e)}")
            raise

    def _initialize_vectorstore(self):
        """Initialize or load the vector store"""
        try:
            # Check if persist directory exists
            if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
                # Load existing vector store
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                logger.info(
                    f"Loaded existing vector store from {self.persist_directory}")
            else:
                # Create directory if it doesn't exist
                os.makedirs(self.persist_directory, exist_ok=True)

                # Create new vector store
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                self.vectorstore.persist()
                logger.info(
                    f"Created new vector store at {self.persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise

    def _initialize_rag_pipeline(self):
        """Initialize the RAG pipeline"""
        try:
            # Create retriever with MMR search
            self.retriever = self.vectorstore.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance
                search_kwargs={
                    "k": 5,         # Number of documents to return
                    "fetch_k": 15,  # Number of documents to consider
                    "lambda_mult": 0.7  # Diversity of results (0-1)
                }
            )

            # Create QA chain with custom prompt
            prompt_template = """You are a professional document intelligence assistant. Answer the question based only on the following context from DOCX documents:
            
            CONTEXT:
            {context}
            
            CHAT HISTORY:
            {chat_history}
            
            USER QUESTION: {question}
            
            Answer in a professional and concise tone. Format your response for readability when appropriate.
            If the answer isn't in the documents, say "I don't have enough information in the documents to answer this question."
            """

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question", "chat_history"]
            )

            # Create the QA chain
            self.qa_chain = (
                {"context": self.retriever,
                 "question": RunnablePassthrough(),
                 "chat_history": lambda x: self._get_chat_history()}
                | PROMPT
                | self.llm
                | StrOutputParser()
            )

            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
            raise

    def _get_chat_history(self) -> str:
        """Get formatted chat history from memory"""
        messages = self.memory.load_memory_variables({}).get("history", [])
        formatted_messages = []

        for message in messages:
            if hasattr(message, "content"):
                role = "Human" if message.type == "human" else "Assistant"
                formatted_messages.append(f"{role}: {message.content}")

        return "\n".join(formatted_messages)

    def add_documents_from_directory(self, directory_path: str, glob_pattern: str = "**/*.docx") -> str:
        """Add documents from a directory to the vector store"""
        try:
            from langchain_community.document_loaders import DirectoryLoader

            # Create directory loader
            loader = DirectoryLoader(
                directory_path,
                glob=glob_pattern,
                loader_cls=DocxLoader,
                use_multithreading=True
            )

            # Load documents
            documents = loader.load()

            if not documents:
                return "No documents found in the specified directory"

            # Process and split documents
            processed_docs = self._process_documents(documents)

            # Add to vector store
            self.vectorstore.add_documents(processed_docs)
            self.vectorstore.persist()

            logger.info(
                f"Added {len(processed_docs)} documents from {directory_path}")
            return f"Successfully processed {len(processed_docs)} document chunks from {len(documents)} files"
        except Exception as e:
            logger.error(f"Error adding documents from directory: {str(e)}")
            return f"Error processing documents: {str(e)}"

    def add_uploaded_files(self, files) -> str:
        """Process and add uploaded files to the vector store"""
        try:
            documents = []

            for file in files:
                try:
                    # In Gradio, when file_count="multiple" and type="binary" is used,
                    # each file is a tuple of (filename, bytes_content)
                    if isinstance(file, tuple) and len(file) == 2:
                        filename, content = file
                    else:
                        # If it's not a tuple, try to handle it as a Gradio File object
                        try:
                            filename = file.name
                            content = file.read() if hasattr(file, 'read') else file
                        except AttributeError:
                            # If we can't get a name, create a generic one
                            filename = f"uploaded_file_{time.time()}.docx"
                            content = file

                    # Generate a unique file path for the uploaded file
                    file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
                    temp_path = f"./temp_{file_hash}_{os.path.basename(filename)}"

                    # Save the byte content to a temporary file
                    with open(temp_path, "wb") as f:
                        f.write(content)

                    # Use the DocxLoader to load the document
                    loader = DocxLoader(temp_path)
                    file_docs = loader.load()

                    # Add documents
                    documents.extend(file_docs)

                    # Remove temporary file
                    os.remove(temp_path)
                except Exception as e:
                    logger.error(f"Error processing file: {str(e)}")
                    return f"Error processing file: {str(e)}"

            # Process and split documents
            processed_docs = self._process_documents(documents)

            if processed_docs:
                # Add to vector store
                self.vectorstore.add_documents(processed_docs)
                self.vectorstore.persist()

                logger.info(
                    f"Added {len(processed_docs)} document chunks from {len(documents)} uploaded files")
                return f"✅ Successfully processed {len(processed_docs)} document chunks from {len(documents)} files"
            else:
                return "No valid content found in the uploaded files"
        except Exception as e:
            logger.error(f"Error processing uploaded files: {str(e)}")
            return f"Error processing files: {str(e)}"

    def _process_documents(self, documents: List[Document]) -> List[Document]:
        """Process and split documents"""
        # Filter out empty documents
        valid_docs = [doc for doc in documents if doc.page_content.strip()]

        if not valid_docs:
            return []

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True,
            separators=["\n\n", "\n", " ", ""]
        )

        return text_splitter.split_documents(valid_docs)

    def query(self, question: str, add_to_memory: bool = True) -> Dict[str, Any]:
        """Query the RAG pipeline and get results"""
        try:
            # Check if the vector store is empty
            if self.vectorstore._collection.count() == 0:
                return {
                    "result": "No documents have been added to the system. Please upload some documents first.",
                    "source_documents": []
                }

            # Get start time
            start_time = time.time()

            # Get the documents from the retriever for source tracking
            retrieved_docs = self.retriever.invoke(question)

            # Get the answer from the QA chain
            answer = self.qa_chain.invoke(question)

            # Calculate query time
            query_time = time.time() - start_time

            # Add to memory if requested
            if add_to_memory:
                # Add user question and assistant response to memory
                self.memory.save_context(
                    {"input": question}, {"output": answer})

            # Return results
            return {
                "result": answer,
                "source_documents": retrieved_docs,
                "query_time": f"{query_time:.2f} seconds"
            }
        except Exception as e:
            logger.error(f"Error querying RAG pipeline: {str(e)}")
            return {
                "result": f"Error processing your request: {str(e)}",
                "source_documents": [],
                "query_time": None
            }

    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the documents in the vector store"""
        try:
            # Get document count
            doc_count = self.vectorstore._collection.count()

            # Get unique sources
            if doc_count > 0:
                docs = self.vectorstore._collection.get()
                metadatas = docs["metadatas"]
                unique_sources = set()
                for metadata in metadatas:
                    if "source" in metadata:
                        unique_sources.add(metadata["source"])
            else:
                unique_sources = set()

            return {
                "document_chunks": doc_count,
                "unique_files": len(unique_sources),
                "files": list(unique_sources)
            }
        except Exception as e:
            logger.error(f"Error getting document stats: {str(e)}")
            return {
                "document_chunks": 0,
                "unique_files": 0,
                "files": []
            }

    def clear_memory(self) -> None:
        """Clear the conversation memory"""
        self.memory.clear()
        logger.info("Conversation memory cleared")

    def reset_vectorstore(self) -> str:
        """Reset the vector store, removing all documents"""
        try:
            # Delete the persist directory
            import shutil
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)

            # Reinitialize the vector store
            self._initialize_vectorstore()

            logger.info("Vector store reset successfully")
            return "✅ Vector store reset successfully. All documents have been removed."
        except Exception as e:
            logger.error(f"Error resetting vector store: {str(e)}")
            return f"Error resetting vector store: {str(e)}"


# Initialize the RAG application
rag_app = RAGApplication()

# Gradio Interface


def query_rag(question, history, show_sources, include_time):
    """Handle the question and return RAG response"""
    try:
        # Query the RAG pipeline
        result = rag_app.query(question)
        response = result["result"]

        # Add sources if requested
        if show_sources and "source_documents" in result:
            sources = "\n\n📄 **Document Sources:**\n" + "\n".join(
                [f"- {os.path.basename(doc.metadata.get('source', 'Unknown'))}"
                 for doc in result["source_documents"]]
            )
            response += sources

        # Add query time if requested
        if include_time and "query_time" in result:
            response += f"\n\n⏱️ Query time: {result['query_time']}"

        return response
    except Exception as e:
        logger.error(f"Error in query_rag: {str(e)}")
        return f"Error processing your request: {str(e)}"


def upload_files(files):
    """Handle DOCX file uploads"""
    if not files:
        return "No files uploaded"

    logger.info(f"Received {len(files)} files for upload")

    # Process files
    result = rag_app.add_uploaded_files(files)

    # Get updated stats
    stats = rag_app.get_document_stats()

    # Return status with stats
    return f"{result}\n\nCurrent document store contains {stats['document_chunks']} chunks from {stats['unique_files']} files."


def clear_conversation():
    """Clear the conversation memory"""
    rag_app.clear_memory()
    return "Conversation memory cleared"


def reset_documents():
    """Reset the document store"""
    result = rag_app.reset_vectorstore()

    # Clear memory as well
    rag_app.clear_memory()

    return result


def display_document_stats():
    """Display statistics about the documents"""
    stats = rag_app.get_document_stats()

    if stats["document_chunks"] == 0:
        return "No documents have been added to the system."

    response = f"📊 **Document Statistics**\n\n"
    response += f"- Total document chunks: {stats['document_chunks']}\n"
    response += f"- Unique files: {stats['unique_files']}\n\n"

    if stats["files"]:
        response += "**Files:**\n"
        for i, file in enumerate(stats["files"], 1):
            response += f"{i}. {os.path.basename(file)}\n"

    return response


# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📄 DOCX Document Intelligence
    **Azure OpenAI-powered RAG application for Word documents**
    """)

    with gr.Tab("💬 Chat with Documents"):
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.ChatInterface(
                    fn=query_rag,
                    additional_inputs=[
                        gr.Checkbox(label="Show document sources", value=True),
                        gr.Checkbox(label="Show query time", value=False)
                    ],
                    examples=[
                        ["What is the main purpose of these documents?", True, False],
                        ["Summarize the key points from these files", False, False],
                        ["What are the major sections in these documents?", True, True]
                    ]
                )

            with gr.Column(scale=1):
                gr.Markdown("### Options")
                clear_memory_btn = gr.Button("Clear Conversation Memory")
                clear_memory_btn.click(
                    clear_conversation,
                    inputs=[],
                    outputs=[gr.Textbox(label="Status")]
                )

                stats_btn = gr.Button("Show Document Statistics")
                stats_output = gr.Textbox(label="Document Stats", lines=10)
                stats_btn.click(
                    display_document_stats,
                    inputs=[],
                    outputs=stats_output
                )

    with gr.Tab("📤 Upload Documents"):
        with gr.Row():
            with gr.Column(scale=3):
                file_input = gr.File(
                    label="Upload DOCX Files",
                    file_types=[".docx"],
                    file_count="multiple",
                    type="binary"
                )
                upload_output = gr.Textbox(label="Upload Status", lines=5)
                file_input.upload(
                    upload_files,
                    inputs=file_input,
                    outputs=upload_output
                )

            with gr.Column(scale=2):
                gr.Markdown("""
                **Instructions:**
                1. Upload one or more DOCX files
                2. Switch to Chat tab to ask questions
                3. Toggle "Show document sources" to see references
                """)

                reset_btn = gr.Button("Reset Document Store", variant="stop")
                reset_output = gr.Textbox(label="Reset Status")
                reset_btn.click(
                    reset_documents,
                    inputs=[],
                    outputs=reset_output
                )

if __name__ == "__main__":
    # Display important setup information
    print("\n" + "="*60)
    print("DOCX Document Intelligence RAG Application")
    print("="*60)
    print("\nEnvironment Variable Check:")

    # Check environment variables
    env_vars = {
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        "AZURE_OPENAI_CHAT_DEPLOYMENT": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", "").replace("https://", "")
    }

    for var, value in env_vars.items():
        status = "✓" if value else "✗"
        if var == "AZURE_OPENAI_API_KEY":
            value = "********" if os.getenv(var) else None
        print(f" {status} {var}: {value or 'Not set'}")

    print("\nIMPORTANT NOTES:")
    print(" - Embedding deployment must use a text-embedding model (e.g., text-embedding-ada-002)")
    print(" - Chat deployment can use a chat model (e.g., gpt-4, gpt-35-turbo)")
    print(" - If you encounter errors, check the logs in 'rag_app.log'\n")
    print("="*60 + "\n")

    # Optional: Add documents from a directory on startup
    # rag_app.add_documents_from_directory("./docs/")

    try:
        # Launch the Gradio interface
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True
        )
    except Exception as e:
        print(f"Error launching application: {str(e)}")
        print("Check the logs in 'rag_app.log' for more details.")

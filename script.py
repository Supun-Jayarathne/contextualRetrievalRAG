import os
import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from typing import List, Iterator, Optional, Dict, Any, Tuple, Union
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from docx import Document as DocxDocument
from typing import Any, Dict
from io import BytesIO
import logging
import time
import hashlib
import json
from tqdm.auto import tqdm

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

# Custom function to safely filter complex metadata


def safe_filter_complex_metadata(doc: Union[Document, Tuple]) -> Document:
    """Safely filter complex metadata from a document or tuple"""
    try:
        # Handle both Document objects and (doc, score) tuples
        if isinstance(doc, tuple):
            doc = doc[0]  # Extract the Document from score tuple

        if not isinstance(doc, Document):
            raise ValueError("Input must be a Document object or tuple")

        # Ensure metadata exists
        metadata = getattr(doc, 'metadata', {})
        if not isinstance(metadata, dict):
            metadata = {}

        # Filter metadata
        filtered = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                filtered[k] = v
            elif isinstance(v, list):
                filtered[k] = ", ".join(
                    str(x) for x in v if isinstance(x, (str, int, float, bool)))

        return Document(
            page_content=doc.page_content,
            metadata=filtered
        )
    except Exception as e:
        logger.error(f"Error filtering metadata: {str(e)}")
        return Document(
            page_content=doc.page_content if hasattr(
                doc, 'page_content') else "",
            metadata={"source": "unknown", "error": str(e)}
        )


class DocxLoader(BaseLoader):
    """Loader for DOCX files that identifies speaker turns and groups content"""

    def __init__(self, file_path: str):
        """Initialize with file path"""
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load and process DOCX file with speaker identification"""
        try:
            doc = DocxDocument(self.file_path)
            full_text = []
            speaker_paragraphs = []
            current_speaker = None
            current_content = []

            def add_speaker_block():
                if current_speaker and current_content:
                    speaker_paragraphs.append({
                        'speaker': current_speaker,
                        'content': '\n'.join(current_content)
                    })
                    full_text.append(
                        f"{current_speaker}: {' '.join(current_content)}")

            for para in doc.paragraphs:
                if not para.text.strip():
                    continue

                # Check for speaker pattern with timestamp (e.g., "John Doe 0:06:")
                # This regex pattern matches:
                # 1. A speaker name (words and spaces)
                # 2. A timestamp (digits and colon)
                # 3. A colon
                import re
                speaker_match = re.match(r'^(.+?)\s\d+:\d+\s*:', para.text)
                if speaker_match:
                    possible_speaker = speaker_match.group(1).strip()
                    # We found a new speaker with timestamp
                    add_speaker_block()
                    current_speaker = possible_speaker
                    # Get content after the timestamp
                    content_start = para.text.find(':', speaker_match.end())
                    current_content = [
                        para.text[content_start+1:].strip()] if content_start != -1 else []
                    continue

                # Check for regular speaker pattern (e.g., "John Doe:")
                if ":" in para.text:
                    possible_speaker, possible_content = para.text.split(
                        ":", 1)
                    possible_speaker = possible_speaker.strip()
                    possible_content = possible_content.strip()

                    # Simple heuristic for speaker names (no spaces or 1-3 words)
                    if len(possible_speaker.split()) <= 3:
                        # We found a new speaker
                        add_speaker_block()
                        current_speaker = possible_speaker
                        current_content = [
                            possible_content] if possible_content else []
                        continue

                # If we get here, it's regular content
                if current_speaker:
                    current_content.append(para.text.strip())
                else:
                    # Content without a speaker
                    current_speaker = "Document"
                    current_content = [para.text.strip()]

            # Add the last speaker block
            add_speaker_block()

            # Create metadata with speaker information
            metadata = {
                "source": self.file_path,
                "filename": os.path.basename(self.file_path),
                "full_document": '\n'.join(full_text),
                "is_full_document": True,
                "speaker_blocks": speaker_paragraphs  # Store speaker-paragraph mapping
            }

            return [Document(page_content='\n'.join(full_text), metadata=metadata)]

        except Exception as e:
            logger.error(f"Error loading {self.file_path}: {str(e)}")
            return [Document(
                page_content="",
                metadata={
                    "source": self.file_path,
                    "error": str(e),
                    "status": "failed"
                }
            )]


class ContextualRetriever:
    """Class to handle contextual retrieval and enrichment of document chunks"""

    def __init__(self, llm, vectorstore):
        self.llm = llm
        self.vectorstore = vectorstore

        # Define prompts for both enrichment and retrieval
        self.enrichment_prompt = PromptTemplate(
            template="""<document>{document}</document>
            <chunk>{chunk}</chunk>
            Provide 1-2 sentences explaining this chunk's context within the document:""",
            input_variables=["document", "chunk"]
        )

        self.retrieval_prompt = PromptTemplate(
            template="""Document: {document}
            Question: {question}
            Chunk: {chunk}
            Score this chunk's relevance (0-1) and explain why:""",
            input_variables=["document", "question", "chunk"]
        )

    def enrich_chunk(self, doc: Document) -> Document:
        """Add contextual information to a document chunk"""
        try:
            if not isinstance(doc, Document):
                raise ValueError("Input must be a Document object")

            full_doc = doc.metadata.get("full_document", "")
            if not full_doc:
                return doc

            chain = self.enrichment_prompt | self.llm | StrOutputParser()
            context = chain.invoke({
                "document": full_doc,
                "chunk": doc.page_content
            })

            new_metadata = doc.metadata.copy()
            new_metadata["contextual_summary"] = context.strip()

            return Document(
                page_content=f"CONTEXT: {context}\nCONTENT: {doc.page_content}",
                metadata=new_metadata
            )

        except Exception as e:
            logger.error(f"Error enriching chunk: {str(e)}")
            return doc

    def contextual_search(self, question: str, k: int = 5) -> List[Document]:
        """Retrieve documents with contextual relevance scoring"""
        try:
            # First get standard similarity results
            docs = self.vectorstore.similarity_search(
                question, k=k*2)  # Get extra for filtering

            if not docs:
                return []

            # Process each document with contextual analysis
            processed_docs = []
            for doc in docs:
                try:
                    # Get full document context from metadata
                    full_context = doc.metadata.get(
                        "full_document", doc.page_content)

                    # Run contextual analysis with more specific prompt
                    analysis_prompt = """Given this document context and user question, please:
                                    1. Provide a relevance score between 0 and 1 (where 1 is most relevant)
                                    2. Explain in one sentence why this is relevant
                                    3. List key connections

                                    DOCUMENT CONTEXT:
                                    {document_context}

                                    USER QUESTION:
                                    {question}

                                    CHUNK CONTENT:
                                    {chunk_content}

                                    Respond in this exact format:
                                    Score: [0-1]
                                    Explanation: [your explanation]
                                    Connections: [comma-separated list]"""

                    analysis = self.llm.invoke(analysis_prompt.format(
                        document_context=full_context,
                        question=question,
                        chunk_content=doc.page_content
                    )).content

                    # Parse the analysis response more robustly
                    score = 0.5  # Default score if parsing fails
                    explanation = ""
                    connections = []

                    try:
                        # Extract score
                        score_line = next(line for line in analysis.split(
                            '\n') if line.startswith('Score:'))
                        score = float(score_line.split(':')[1].strip())
                        score = max(0, min(1, score))  # Clamp between 0-1

                        # Extract explanation
                        explanation_line = next(line for line in analysis.split(
                            '\n') if line.startswith('Explanation:'))
                        explanation = explanation_line.split(':', 1)[1].strip()

                        # Extract connections
                        connections_line = next(line for line in analysis.split(
                            '\n') if line.startswith('Connections:'))
                        connections = [c.strip() for c in connections_line.split(':', 1)[
                            1].split(',')]
                    except (StopIteration, ValueError, IndexError) as e:
                        logger.warning(
                            f"Couldn't fully parse analysis: {str(e)}. Using defaults.")

                    # Update document metadata with contextual analysis
                    enriched_metadata = doc.metadata.copy()
                    enriched_metadata.update({
                        "contextual_score": float(score),
                        "contextual_explanation": explanation,
                        "contextual_connections": connections
                    })

                    # Create enriched document
                    enriched_doc = Document(
                        page_content=doc.page_content,
                        metadata=enriched_metadata
                    )
                    processed_docs.append(enriched_doc)

                except Exception as e:
                    logger.error(f"Error processing document: {str(e)}")
                    continue

            # Sort by contextual score and return top k
            processed_docs.sort(
                key=lambda x: x.metadata.get("contextual_score", 0),
                reverse=True
            )

            return processed_docs[:k]

        except Exception as e:
            logger.error(f"Error in contextual search: {str(e)}")
            return []


class ContextualRAGApplication:
    """Enhanced RAG application with contextual embedding"""

    def __init__(self, persist_directory: str = "./chroma_db_docx"):
        """Initialize the Contextual RAG application"""
        self.persist_directory = persist_directory
        self.embeddings = None
        self.llm = None
        self.vectorstore = None
        self.retriever = None
        self.qa_chain = None
        self.contextual_enricher = None
        self.memory = ConversationBufferWindowMemory(k=5, return_messages=True)

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True
        )

        # Load environment variables
        load_dotenv()

        # Initialize components in proper order
        self._initialize_azure_components()
        self._initialize_vectorstore()

        # Initialize contextual components - this is correct
        self.contextual_enricher = ContextualRetriever(
            self.llm, self.vectorstore)
        self._initialize_rag_pipeline()

        logger.info("Contextual RAG application initialized successfully")

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
                    )

            # Initialize LLM
            try:
                self.llm = AzureChatOpenAI(
                    azure_deployment=chat_deployment,
                    openai_api_version=api_version,
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    temperature=0.1,  # Low temperature for factual responses
                    max_tokens=1000
                )
                logger.info(f"Chat model successfully initialized")
            except Exception as e:
                raise ValueError(f"Failed to initialize chat model: {str(e)}")

            logger.info("Azure OpenAI components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure components: {str(e)}")
            raise

    def _initialize_contextual_enricher(self):
        """Initialize the contextual enrichment component"""
        try:
            self.contextual_enricher = ContextualRetriever(
                self.llm, self.vectorstore)
            logger.info("Contextual enricher initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize contextual enricher: {str(e)}")
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
        """Initialize RAG pipeline with enhanced contextual retrieval"""
        try:
            # Define the prompt template with contextual awareness
            prompt_template = """You are a document intelligence assistant. Use the following context to answer:

            CONTEXTUAL ANALYSIS:
            {contextual_analysis}

            CHUNK CONTENT:
            {context}

            CHAT HISTORY:
            {chat_history}

            QUESTION: {question}

            Guidelines:
            1. First analyze the contextual relevance information
            2. Focus on answers that synthesize information across chunks
            3. For factual questions, verify against multiple chunks when possible
            4. If unsure, say "I couldn't find definitive information about this in the documents"
            """

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["contextual_analysis",
                                 "context", "question", "chat_history"]
            )

            # Create a runnable for contextual retrieval
            def contextual_retriever(input_dict: Dict[str, Any]) -> Dict[str, Any]:
                """Retrieve documents with contextual analysis"""
                try:
                    question = input_dict["question"]
                    docs = self.contextual_enricher.contextual_search(question)

                    contextual_analysis = "\n\n".join(
                        f"Document {i+1} (Relevance: {doc.metadata.get('contextual_score', 0):.0%}):\n"
                        f"Explanation: {doc.metadata.get('contextual_explanation', '')}\n"
                        f"Connections: {', '.join(doc.metadata.get('contextual_connections', []))}"
                        for i, doc in enumerate(docs)
                    )

                    context_content = "\n\n".join(
                        f"Document {i+1}:\n{doc.page_content}"
                        for i, doc in enumerate(docs))

                    return {
                        "contextual_analysis": contextual_analysis,
                        "context": context_content,
                        "question": question,
                        "chat_history": input_dict.get("chat_history", "")
                    }
                except Exception as e:
                    logger.error(f"Error in contextual retriever: {str(e)}")
                    return {
                        "contextual_analysis": "Error retrieving contextual information",
                        "context": "",
                        "question": question,
                        "chat_history": input_dict.get("chat_history", "")
                    }

            # Build the full chain
            self.qa_chain = (
                RunnablePassthrough()
                | contextual_retriever
                | PROMPT
                | self.llm
                | StrOutputParser()
            )

            logger.info("Contextual RAG pipeline initialized successfully")

        except ValueError as ve:
            logger.error(f"Configuration error in RAG pipeline: {str(ve)}")
            raise ValueError(
                f"Failed to initialize RAG pipeline due to configuration issue: {str(ve)}")

        except Exception as e:
            logger.error(
                f"Unexpected error initializing RAG pipeline: {str(e)}")
            raise RuntimeError(f"Failed to initialize RAG pipeline: {str(e)}")

    def _get_chat_history(self) -> str:
        """Get formatted chat history from memory"""
        messages = self.memory.load_memory_variables({}).get("history", [])
        formatted_messages = []

        for message in messages:
            if hasattr(message, "content"):
                role = "Human" if message.type == "human" else "Assistant"
                formatted_messages.append(f"{role}: {message.content}")

        return "\n".join(formatted_messages)

    def add_uploaded_files(self, files) -> str:
        """Process and add uploaded files to the vector store"""
        try:
            documents = []

            for file in files:
                try:
                    # Handle Gradio file input which can be either:
                    # 1. A tuple of (file_path, file_content)
                    # 2. A file-like object with .name and .read() attributes
                    # 3. Raw bytes

                    # Extract filename and content based on input type
                    if isinstance(file, tuple) and len(file) == 2:
                        # Case 1: Gradio tuple input
                        filename, file_content = file
                        file_bytes = file_content
                    elif hasattr(file, 'name') and hasattr(file, 'read'):
                        # Case 2: File-like object
                        filename = file.name
                        file.seek(0)  # Rewind in case it was already read
                        file_bytes = file.read()
                    else:
                        # Case 3: Assume raw bytes
                        filename = f"uploaded_file_{time.time()}.docx"
                        file_bytes = file

                    # Validate we have bytes content
                    if not isinstance(file_bytes, bytes):
                        raise ValueError("File content must be bytes")

                    # Create a temporary file path
                    file_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
                    temp_path = f"./temp_{file_hash}_{os.path.basename(filename)}"

                    # Write to temporary file
                    with open(temp_path, "wb") as f:
                        f.write(file_bytes)

                    # Load document using DocxLoader
                    loader = DocxLoader(temp_path)
                    loaded_docs = loader.load()

                    # Only keep documents with actual content
                    for doc in loaded_docs:
                        if doc.page_content.strip():
                            # Ensure metadata exists and is a dictionary
                            if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
                                doc.metadata = {}

                            # Add original filename to metadata
                            doc.metadata['original_filename'] = filename
                            documents.append(doc)

                    # Clean up temporary file
                    os.remove(temp_path)

                except Exception as e:
                    logger.error(
                        f"Error processing file {filename if 'filename' in locals() else 'unknown'}: {str(e)}")
                    continue

            if not documents:
                return "No valid documents were extracted from the uploaded files"

            if documents:
                # Process documents with safe metadata handling
                processed_docs = self._process_documents(documents)

                if processed_docs:
                    # Add to vector store with additional safety check
                    safe_docs = [safe_filter_complex_metadata(
                        doc) for doc in processed_docs]
                    self.vectorstore.add_documents(safe_docs)
                    self.vectorstore.persist()
                    return f"Processed {len(safe_docs)} chunks"

            return "No valid content found"

        except Exception as e:
            logger.error(
                f"Error in add_uploaded_files: {str(e)}", exc_info=True)
            return f"Error processing files: {str(e)}"

    def _process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents while preserving speaker information and adding contextual enrichment"""
        # First filter out empty documents and non-Document objects
        valid_docs = []
        for doc in documents:
            try:
                if not isinstance(doc, Document):
                    logger.warning(
                        f"Skipping non-Document object: {type(doc)}")
                    continue
                if not doc.page_content.strip():
                    continue
                valid_docs.append(doc)
            except Exception as e:
                logger.error(f"Error validating document: {str(e)}")
                continue

        if not valid_docs:
            return []

        processed_chunks = []
        for doc in valid_docs:
            try:
                # Safely get metadata with defaults
                metadata = getattr(doc, 'metadata', {})
                if not isinstance(metadata, dict):
                    metadata = {}

                full_document = metadata.get("full_document", doc.page_content)
                speaker_blocks = metadata.get("speaker_blocks", [])

                # Split document into chunks
                chunks = self.text_splitter.split_documents([doc])

                for chunk in chunks:
                    try:
                        if not isinstance(chunk, Document) or not chunk.page_content.strip():
                            continue

                        # Preserve speaker information
                        chunk_speaker_blocks = []
                        for block in speaker_blocks:
                            if isinstance(block, dict) and block.get('content', '') in chunk.page_content:
                                chunk_speaker_blocks.append(block)

                        # Prepare chunk metadata
                        chunk_metadata = {
                            "original_content": chunk.page_content,
                            "full_document": full_document,
                            "is_full_document": False,
                            "speaker_blocks": chunk_speaker_blocks,
                            "source": metadata.get("source", "unknown"),
                            "filename": metadata.get("filename", "unknown")
                        }

                        # Update with existing metadata (filtered)
                        chunk_metadata.update(
                            safe_filter_complex_metadata(chunk).metadata)

                        # Create filtered chunk
                        filtered_chunk = Document(
                            page_content=chunk.page_content,
                            metadata=chunk_metadata
                        )

                        # Enrich the chunk with contextual information
                        enriched_chunk = self.contextual_enricher.enrich_chunk(
                            filtered_chunk)
                        if enriched_chunk and enriched_chunk.page_content.strip():
                            processed_chunks.append(enriched_chunk)

                    except Exception as chunk_error:
                        logger.error(
                            f"Error processing chunk: {str(chunk_error)}")
                        continue

            except Exception as doc_error:
                logger.error(f"Error processing document: {str(doc_error)}")
                continue

        return processed_chunks

    def query(self, question: str, add_to_memory: bool = True) -> Dict[str, Any]:
        """Query the contextual RAG pipeline"""
        try:
            # Check if the vector store is empty
            if self.vectorstore._collection.count() == 0:
                return {
                    "result": "No documents have been added to the system. Please upload some documents first.",
                    "source_documents": []
                }

            # Get start time
            start_time = time.time()

            # Get the answer from the QA chain
            answer = self.qa_chain.invoke({
                "question": question,
                "chat_history": self._get_chat_history()
            })

            # Get the source documents with contextual information
            # Changed from contextual_retriever to contextual_enricher
            source_docs = self.contextual_enricher.contextual_search(question)

            # Calculate query time
            query_time = time.time() - start_time

            # Add to memory if requested
            if add_to_memory:
                self.memory.save_context(
                    {"input": question},
                    {"output": answer}
                )

            return {
                "result": answer,
                "source_documents": source_docs,
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

                # Collect topic statistics
                all_topics = []
                all_entities = []

                for metadata in metadatas:
                    if "source" in metadata:
                        unique_sources.add(metadata["source"])

                    # Collect topics and entities for statistics
                    if "topics" in metadata and isinstance(metadata["topics"], list):
                        all_topics.extend(metadata["topics"])
                    if "entities" in metadata and isinstance(metadata["entities"], list):
                        all_entities.extend(metadata["entities"])

                # Get top topics and entities
                top_topics = self._get_top_items(all_topics, 10)
                top_entities = self._get_top_items(all_entities, 10)

            else:
                unique_sources = set()
                top_topics = []
                top_entities = []

            return {
                "document_chunks": doc_count,
                "unique_files": len(unique_sources),
                "files": list(unique_sources),
                "top_topics": top_topics,
                "top_entities": top_entities
            }
        except Exception as e:
            logger.error(f"Error getting document stats: {str(e)}")
            return {
                "document_chunks": 0,
                "unique_files": 0,
                "files": [],
                "top_topics": [],
                "top_entities": []
            }

    def _get_top_items(self, items: List[str], limit: int = 10) -> List[Tuple[str, int]]:
        """Get top items by frequency"""
        from collections import Counter
        if not items:
            return []

        counter = Counter(items)
        return counter.most_common(limit)

    def clear_memory(self) -> None:
        """Clear the conversation memory"""
        self.memory.clear()
        logger.info("Conversation memory cleared")

    def reset_vectorstore(self) -> str:
        """Reset the vector store, removing all documents"""
        try:
            # Close existing connections first
            if self.vectorstore:
                try:
                    self.vectorstore._client = None
                    self.vectorstore._collection = None
                except Exception as e:
                    logger.warning(f"Error cleaning up vectorstore: {str(e)}")

            # Delete the persist directory
            import shutil
            if os.path.exists(self.persist_directory):
                for _ in range(3):  # Retry up to 3 times
                    try:
                        shutil.rmtree(self.persist_directory)
                        break
                    except PermissionError as e:
                        logger.warning(f"Retrying delete: {str(e)}")
                        time.sleep(0.5)  # Wait briefly before retrying
                    except Exception as e:
                        logger.error(f"Error deleting directory: {str(e)}")
                        raise

            # Reinitialize the vector store
            self._initialize_vectorstore()

            logger.info("Vector store reset successfully")
            return "✅ Vector store reset successfully. All documents have been removed."
        except Exception as e:
            logger.error(f"Error resetting vector store: {str(e)}")
            return f"❌ Error resetting vector store: {str(e)}"

    def search_by_metadata(self, field: str, query: str) -> List[Document]:
        """Search for documents with specific metadata field values"""
        try:
            # Get all documents
            all_docs = self.vectorstore._collection.get()
            metadatas = all_docs["metadatas"]
            documents = all_docs["documents"]

            # Filter documents by metadata field
            results = []
            for i, metadata in enumerate(metadatas):
                # Check if the field exists and contains the query
                if field in metadata:
                    field_value = metadata[field]

                    # Handle different types of metadata fields
                    if isinstance(field_value, list):
                        # For lists (like topics, entities)
                        if any(query.lower() in item.lower() for item in field_value):
                            results.append(Document(
                                page_content=documents[i],
                                metadata=metadata
                            ))
                    elif isinstance(field_value, str):
                        # For string fields
                        if query.lower() in field_value.lower():
                            results.append(Document(
                                page_content=documents[i],
                                metadata=metadata
                            ))

            return results
        except Exception as e:
            logger.error(f"Error searching by metadata: {str(e)}")
            return []


# Initialize the RAG application
rag_app = ContextualRAGApplication()

# Gradio Interface


def query_rag(question, history, show_sources, include_time, show_context):
    """Handle the question and return RAG response with contextual sources"""
    try:
        # Query the RAG pipeline
        result = rag_app.query(question)
        response = result["result"]

        # Add sources if requested
        if show_sources and "source_documents" in result:
            sources = "\n\n📄 **Document Sources (with Contextual Relevance):**\n"

            for i, doc in enumerate(result["source_documents"], 1):
                source_name = os.path.basename(
                    doc.metadata.get('source', 'Unknown'))
                score = doc.metadata.get("contextual_score", 0)
                score_percent = int(score * 100)

                sources += f"\n**Source {i}: {source_name} (Relevance: {score_percent}%)**\n"

                if show_context:
                    explanation = doc.metadata.get(
                        "contextual_explanation", "")
                    connections = doc.metadata.get(
                        "contextual_connections", [])

                    sources += f"*Why relevant:* {explanation}\n"
                    if connections:
                        sources += f"*Key connections:* {', '.join(connections)}\n"

                # Show the original content
                chunk_content = doc.metadata.get(
                    "original_content", doc.page_content)
                sources += f'"{chunk_content}"\n'

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

    if stats["top_topics"]:
        response += "**Top Topics:**\n"
        for topic, count in stats["top_topics"]:
            response += f"- {topic} ({count})\n"
        response += "\n"

    if stats["top_entities"]:
        response += "**Top Entities:**\n"
        for entity, count in stats["top_entities"]:
            response += f"- {entity} ({count})\n"
        response += "\n"

    if stats["files"]:
        response += "**Files:**\n"
        for i, file in enumerate(stats["files"], 1):
            response += f"{i}. {os.path.basename(file)}\n"

    return response


# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 📄 Contextual Document Intelligence
    **Azure OpenAI-powered Contextual RAG application**
    
    This application uses contextual retrieval to enrich document chunks with additional information before indexing them.
    """)

    with gr.Tab("💬 Chat with Documents"):
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.ChatInterface(
                    fn=query_rag,
                    additional_inputs=[
                        gr.Checkbox(
                            label="Show document sources with content", value=True),
                        gr.Checkbox(label="Show query time", value=False),
                        gr.Checkbox(
                            label="Show contextual summaries", value=True)
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
                3. Toggle "Show document sources" to see references with content and relevance scores
                """)

                reset_btn = gr.Button("Reset Document Store", variant="stop")
                reset_output = gr.Textbox(label="Reset Status")
                reset_btn.click(
                    reset_documents,
                    inputs=[],
                    outputs=reset_output
                )

if __name__ == "__main__":
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

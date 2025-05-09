import numpy as np
import logging
import nltk
from rank_bm25 import BM25Okapi
from typing import List
from langchain_core.documents import Document
from processors.keyword_processor import KeywordProcessor
from langchain_community.vectorstores import Chroma


class BM25Retriever:
    """Enhanced BM25 retriever with proper empty collection handling"""

    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        self.bm25 = None
        self.documents = []
        self.keyword_processor = KeywordProcessor()
        self._initialize_bm25()

    def _initialize_bm25(self):
        """Initialize BM25 with proper error handling"""
        try:
            collection = self.vectorstore._collection.get()
            self.documents = []

            # Check if we have any documents
            if not collection.get('documents'):
                logging.warning(
                    "Vector store contains no documents - BM25 will be disabled")
                self.bm25 = None
                return

            for content, meta in zip(collection['documents'], collection['metadatas']):
                # Skip empty documents
                if not content.strip():
                    continue

                # Combine keywords and preprocessed text
                keywords = meta.get('keywords', [])
                preprocessed = meta.get('preprocessed_text',
                                        self.keyword_processor.preprocess_text(content))

                # Create hybrid document representation
                doc_representation = f"{' '.join(keywords)} {preprocessed}"
                self.documents.append(doc_representation)

            # Check if we ended up with any valid documents
            if not self.documents:
                logging.warning(
                    "No valid documents found for BM25 initialization")
                self.bm25 = None
                return

            tokenized_docs = [doc.split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
            logging.info(
                f"BM25 retriever initialized with {len(self.documents)} documents")

        except Exception as e:
            logging.error(f"Error initializing BM25: {str(e)}")
            self.bm25 = None  # Ensure it's None if initialization fails
            raise

    def get_hybrid_query(self, query: str) -> str:
        """Enhance the query with extracted keywords"""
        if not query.strip():
            return ""

        query_keywords = self.keyword_processor.extract_keywords(query)
        query_preprocessed = self.keyword_processor.preprocess_text(query)
        return f"{' '.join(query_keywords)} {query_preprocessed}"

    def retrieve(self, query: str, k: int = 10) -> List[Document]:
        """Retrieve documents with proper empty collection handling"""
        try:
            if not self.bm25 or not self.documents:
                return []

            if not query.strip():
                return []

            # Enhance the query
            enhanced_query = self.get_hybrid_query(query)

            # Get top documents with BM25
            tokenized_query = enhanced_query.split()
            doc_scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(doc_scores)[::-1][:k]

            # Get corresponding documents from vector store
            collection = self.vectorstore._collection.get()
            retrieved_docs = []

            for idx in top_indices:
                if idx < len(collection['ids']):
                    doc_id = collection['ids'][idx]
                    doc_content = collection['documents'][idx]
                    doc_metadata = collection['metadatas'][idx] if 'metadatas' in collection else {
                    }

                    # Add BM25 score to metadata
                    metadata = doc_metadata.copy()
                    metadata['bm25_score'] = float(doc_scores[idx])
                    metadata['bm25_query'] = enhanced_query

                    retrieved_docs.append(Document(
                        page_content=doc_content,
                        metadata=metadata
                    ))

            return retrieved_docs

        except Exception as e:
            logging.error(f"Error in BM25 retrieval: {str(e)}")
            return []

    # Download NLTK resources if not already present
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')

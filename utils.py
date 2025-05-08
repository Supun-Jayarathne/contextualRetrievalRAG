import os
import time
import logging
import hashlib
import re
from typing import Union
from langchain_core.documents import Document


def safe_filter_metadata(doc: Union[Document, dict]) -> Union[Document, dict]:
    """Safely filter metadata to only include simple types that ChromaDB can handle.
    Handles both Document objects and raw dictionaries."""
    try:
        if isinstance(doc, Document):
            # Handle Document objects
            metadata = getattr(doc, 'metadata', {})
            if not isinstance(metadata, dict):
                metadata = {}

            filtered_metadata = {}
            for key, value in metadata.items():
                # Skip None values
                if value is None:
                    continue

                # Handle allowed simple types
                if isinstance(value, (str, int, float, bool)):
                    filtered_metadata[key] = value
                # Convert lists to comma-separated strings
                elif isinstance(value, list):
                    filtered_metadata[key] = ", ".join(
                        str(item) for item in value
                        if isinstance(item, (str, int, float, bool))
                    )
                # Convert other types to strings
                else:
                    try:
                        filtered_metadata[key] = str(value)
                    except:
                        continue

            return Document(
                page_content=doc.page_content,
                metadata=filtered_metadata
            )

        elif isinstance(doc, dict):
            # Handle dictionary inputs
            filtered = {}
            for key, value in doc.items():
                if value is None:
                    continue
                if isinstance(value, (str, int, float, bool)):
                    filtered[key] = value
                elif isinstance(value, list):
                    filtered[key] = ", ".join(
                        str(item) for item in value
                        if isinstance(item, (str, int, float, bool))
                    )
                else:
                    try:
                        filtered[key] = str(value)
                    except:
                        continue
            return filtered

        raise ValueError("Input must be a Document object or dictionary")

    except Exception as e:
        logging.error(f"Error filtering metadata: {str(e)}")
        # Fallback to minimal metadata
        if isinstance(doc, Document):
            return Document(
                page_content=doc.page_content,
                metadata={"source": "unknown",
                          "error": "metadata_filter_failed"}
            )
        return {"source": "unknown", "error": "metadata_filter_failed"}


def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("rag_app.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

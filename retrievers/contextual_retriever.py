import logging
import time
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from processors.keyword_processor import KeywordProcessor
from langchain_community.vectorstores import Chroma


class ContextualRetriever:
    """Class to handle contextual retrieval and enrichment of document chunks"""

    def __init__(self, llm, vectorstore, bm25_retriever):
        self.llm = llm
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        self.keyword_processor = KeywordProcessor()

        # Define prompts for both enrichment and retrieval
        self.enrichment_prompt = PromptTemplate(
            template="""<document>{document}</document>
            <chunk>{chunk}</chunk>
            Provide 1-4 sentences explaining this chunk's context within the document:""",
            input_variables=["document", "chunk"]
        )

        self.retrieval_prompt = PromptTemplate(
            template="""Document: {document}
            Question: {question}
            Chunk: {chunk}
            Score this chunk's relevance (0-1) and explain why:""",
            input_variables=["document", "question", "chunk"]
        )

    def _analyze_query(self, question: str) -> dict:
        """Determine query characteristics to guide retrieval"""
        analysis = {
            'type': 'fact',  # fact, opinion, summary, etc.
            'entities': [],
            'keywords': []
        }

        # Simple heuristic - could be enhanced with LLM
        question_lower = question.lower()
        if 'what is' in question_lower or 'who is' in question_lower:
            analysis['type'] = 'definition'
        elif 'how to' in question_lower:
            analysis['type'] = 'process'
        elif 'why' in question_lower:
            analysis['type'] = 'explanation'

        # Extract entities and keywords (simple version)
        analysis['keywords'] = [word for word in question.split()
                                if word.lower() not in ['what', 'how', 'why', 'when']]

        return analysis

    def hybrid_search(self, question: str, k: int = 10) -> List[Document]:
        """Improved hybrid search with consistent scoring and deduplication"""
        try:
            # 1. Retrieve documents from both methods
            vector_docs = self.vectorstore.similarity_search(question, k=k*3)
            bm25_docs = self.bm25_retriever.retrieve(question, k=k*3)

            # 2. Create unified scoring dictionary with content as key
            doc_dict = {}

            # Process vector docs first
            for doc in vector_docs:
                if not isinstance(doc, Document):
                    continue

                content = doc.page_content
                if content not in doc_dict:
                    doc_dict[content] = {
                        'doc': doc,
                        'vector_score': doc.metadata.get('similarity_score', 0),
                        'bm25_score': 0  # Initialize BM25 score
                    }
                else:
                    # Keep the highest vector score if we see duplicates
                    doc_dict[content]['vector_score'] = max(
                        doc_dict[content]['vector_score'],
                        doc.metadata.get('similarity_score', 0)
                    )

            # Process BM25 docs
            for doc in bm25_docs:
                if not isinstance(doc, Document):
                    continue

                content = doc.page_content
                if content not in doc_dict:
                    doc_dict[content] = {
                        'doc': doc,
                        'vector_score': 0,  # Initialize vector score
                        'bm25_score': doc.metadata.get('bm25_score', 0)
                    }
                else:
                    # Keep the highest BM25 score if we see duplicates
                    doc_dict[content]['bm25_score'] = max(
                        doc_dict[content]['bm25_score'],
                        doc.metadata.get('bm25_score', 0)
                    )

            # 3. Normalize and combine scores
            if not doc_dict:
                return []

            # Get max scores for normalization
            max_vector = max(info['vector_score']
                             for info in doc_dict.values()) or 1
            max_bm25 = max(info['bm25_score']
                           for info in doc_dict.values()) or 1

            # Calculate combined scores
            scored_docs = []
            for content, info in doc_dict.items():
                # Normalize scores
                norm_vector = info['vector_score'] / max_vector
                norm_bm25 = info['bm25_score'] / max_bm25

                # Weighted combination (adjust weights as needed)
                combined_score = (norm_vector * 0.4) + (norm_bm25 * 0.6)

                # Create enriched document
                metadata = info['doc'].metadata.copy()
                metadata.update({
                    'vector_score': norm_vector,
                    'bm25_score': norm_bm25,
                    'combined_score': combined_score,
                    'relevance_percentage': int(combined_score * 100)
                })

                scored_docs.append((
                    Document(page_content=content, metadata=metadata),
                    combined_score
                ))

            # 4. Sort by combined score (descending)
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # 5. Apply contextual analysis to top k unique documents
            top_docs = [doc for doc, score in scored_docs[:k]]
            return self._apply_contextual_analysis(question, top_docs)

        except Exception as e:
            logging.error(f"Error in hybrid search: {str(e)}")
            return []

    def _apply_contextual_analysis(self, question: str, docs: List[Document]) -> List[Document]:
        """Apply contextual analysis to retrieved documents"""
        processed_docs = []

        for doc in docs:
            try:
                # Get full document context from metadata
                full_context = doc.metadata.get(
                    "full_document", doc.page_content)

                # Run contextual analysis
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

                # Parse the analysis response
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
                    logging.warning(
                        f"Couldn't fully parse analysis: {str(e)}. Using defaults.")

                # Update document metadata with contextual analysis
                enriched_metadata = doc.metadata.copy()
                enriched_metadata.update({
                    "contextual_score": float(score),
                    "contextual_explanation": explanation,
                    "contextual_connections": connections,
                    "retrieval_method": "hybrid"
                })

                # Create enriched document
                enriched_doc = Document(
                    page_content=doc.page_content,
                    metadata=enriched_metadata
                )
                processed_docs.append(enriched_doc)

            except Exception as e:
                logging.error(f"Error processing document: {str(e)}")
                continue

        return processed_docs

    def contextual_search(self, question: str, k: int = 10) -> List[Document]:
        """Use hybrid search by default"""
        return self.hybrid_search(question, k)

    def enrich_chunk(self, doc: Document) -> Document:
        """Enhanced contextual enrichment with better error handling"""
        try:
            if not isinstance(doc, Document):
                raise ValueError("Input must be a Document object")

            # Get full document context
            full_doc = doc.metadata.get("full_document", "")
            if not full_doc:
                logging.warning(
                    "No full document context available for enrichment")
                return doc

            # Skip enrichment if chunk is too small (may not have enough context)
            if len(doc.page_content) < 50:
                logging.debug("Skipping enrichment for very small chunk")
                return doc

            try:
                # Run enrichment chain
                chain = self.enrichment_prompt | self.llm | StrOutputParser()
                context = chain.invoke({
                    "document": full_doc,
                    "chunk": doc.page_content
                }).strip()

                if not context:
                    raise ValueError("Empty context returned from enrichment")

                # Create new metadata
                new_metadata = doc.metadata.copy()
                new_metadata.update({
                    "contextual_summary": context,
                    "enrichment_time": time.time()
                })

                return Document(
                    page_content=f"CONTEXT: {context}\nCONTENT: {doc.page_content}",
                    metadata=new_metadata
                )

            except Exception as e:
                logging.error(f"Error in enrichment chain: {str(e)}")
                return doc  # Return original doc if enrichment fails

        except Exception as e:
            logging.error(f"Error in enrich_chunk: {str(e)}")
            return doc  # Fallback to original document

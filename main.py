import os
import gradio as gr
from models.rag_app import ContextualRAGApplication
from utils import setup_logging

# Set up logging
logger = setup_logging()

# Initialize the RAG application
rag_app = ContextualRAGApplication()


def query_rag(question, history, show_sources, include_time, show_context):
    """Handle the question and return RAG response with contextual sources"""
    try:
        # Query the RAG pipeline
        result = rag_app.query(question)
        response = result["result"]

        # Add sources if requested
        if show_sources and "source_documents" in result:
            sources = "\n\n📄 **Document Sources (with Contextual Relevance):**\n"

            for i, doc in enumerate(result["source_documents"][:10], 1):
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

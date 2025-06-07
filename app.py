import streamlit as st
import fitz  # PyMuPDF
import openai
# from pinecone import Pinecone
import pinecone
from uuid import uuid4
from datetime import datetime
import hashlib
import re
from typing import List, Dict, Tuple

from dotenv import load_dotenv
load_dotenv()
import os

# Load environment variables

# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.getenv("PORT", 8000))
#     uvicorn.run("main:app", host="0.0.0.0", port=port)

# Initialize APIs with error handling
try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        st.error("OpenAI API key not found. Please add OPENAI_API_KEY to your .env file.")
        st.stop()
    
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        st.error("Pinecone API key not found. Please add PINECONE_API_KEY to your .env file.")
        st.stop()
        

    pinecone.init(api_key=pinecone_api_key)
    index = pinecone.Index("n8npdffiles")
    
except Exception as e:
    st.error(f"Failed to initialize APIs: {str(e)}")
    st.stop()

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = {}

def get_file_hash(file_bytes: bytes) -> str:
    """Generate a unique hash for the file to prevent duplicates."""
    return hashlib.md5(file_bytes).hexdigest()

def extract_text_from_pdf(file_bytes: bytes) -> Tuple[str, int]:
    """Extract text from PDF and return text with page count."""
    doc = None
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        pages_text = []
        
        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():  # Only add non-empty pages
                pages_text.append(f"[Page {page_num}]\n{text}")
        
        return "\n\n".join(pages_text), len(doc)
    
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return "", 0
    finally:
        if doc:
            doc.close()

def chunk_text_smart(text: str, max_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """Smart chunking that preserves sentence boundaries."""
    # Split by sentences using regex
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    current_words = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        # If adding this sentence would exceed max_size, save current chunk
        if current_words + sentence_words > max_size and current_chunk:
            # Extract page number from chunk if available
            page_match = re.search(r'\[Page (\d+)\]', current_chunk)
            page_num = int(page_match.group(1)) if page_match else None
            
            chunks.append({
                "text": current_chunk.strip(),
                "word_count": current_words,
                "page": page_num
            })
            
            # Start new chunk with overlap
            overlap_words = current_chunk.split()[-overlap:] if overlap > 0 else []
            current_chunk = " ".join(overlap_words) + " " + sentence
            current_words = len(overlap_words) + sentence_words
        else:
            current_chunk += " " + sentence if current_chunk else sentence
            current_words += sentence_words
    
    # Add the last chunk
    if current_chunk.strip():
        page_match = re.search(r'\[Page (\d+)\]', current_chunk)
        page_num = int(page_match.group(1)) if page_match else None
        
        chunks.append({
            "text": current_chunk.strip(),
            "word_count": current_words,
            "page": page_num
        })
    
    return chunks

def embed_and_store(chunks: List[Dict], filename: str, file_hash: str) -> bool:
    """Embed chunks and store in Pinecone with progress tracking."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for i, chunk_data in enumerate(chunks):
            status_text.text(f"Processing chunk {i+1}/{len(chunks)}...")
            
            # Create embedding
            embedding_response = openai.embeddings.create(
                input=[chunk_data["text"]], 
                model="text-embedding-3-small"
            )
            embedding = embedding_response.data[0].embedding
            
            # Prepare metadata
            metadata = {
                "chunk": chunk_data["text"],
                "filename": filename,
                "file_hash": file_hash,
                "page": chunk_data.get("page"),
                "word_count": chunk_data["word_count"],
                "upload_date": datetime.now().isoformat()
            }
            
            # Upsert to Pinecone
            index.upsert([{
                "id": f"{file_hash}_{i}",
                "values": embedding,
                "metadata": metadata
            }])
            
            # Update progress
            progress_bar.progress((i + 1) / len(chunks))
        
        progress_bar.empty()
        status_text.empty()
        return True
        
    except Exception as e:
        st.error(f"Error during embedding and storage: {str(e)}")
        progress_bar.empty()
        status_text.empty()
        return False

def search_documents(question: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
    """Search for relevant chunks and return context with source info."""
    try:
        # Create question embedding
        embedding_response = openai.embeddings.create(
            input=[question], 
            model="text-embedding-3-small"
        )
        question_embedding = embedding_response.data[0].embedding
        
        # Query Pinecone
        results = index.query(
            vector=question_embedding, 
            top_k=top_k, 
            include_metadata=True
        )
        
        # Process results
        context_parts = []
        sources = []
        
        for match in results["matches"]:
            if "metadata" in match and "chunk" in match["metadata"]:
                metadata = match["metadata"]
                similarity_score = match["score"]
                
                context_parts.append(metadata["chunk"])
                sources.append({
                    "filename": metadata.get("filename", "Unknown"),
                    "page": metadata.get("page"),
                    "score": similarity_score,
                    "chunk_preview": metadata["chunk"][:100] + "..." if len(metadata["chunk"]) > 100 else metadata["chunk"]
                })
        
        return "\n\n".join(context_parts), sources
        
    except Exception as e:
        st.error(f"Error during search: {str(e)}")
        return "", []

def generate_answer(context: str, question: str) -> str:
    """Generate answer using OpenAI with improved prompting."""
    try:
        prompt = f"""Answer the question based on the provided context. If the context doesn't contain enough information to fully answer the question, say so clearly.

Context:
{context}

Question: {question}

Instructions:
- Be specific and cite relevant details from the context
- If information is missing, acknowledge what cannot be answered
- Keep the answer concise but comprehensive
- Use plain text formatting only - no markdown, italics, or special formatting
- Ensure numbers and currency values are clearly readable
- Use proper spacing between words and sentences"""

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
        
        # Clean up any potential formatting issues
        answer = response.choices[0].message.content
        # Remove any stray formatting characters that might cause display issues
        answer = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\$\%\n]', ' ', answer)
        # Fix multiple spaces
        answer = re.sub(r'\s+', ' ', answer)
        
        return answer.strip()
        
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "Sorry, I couldn't generate an answer due to an error."

# Streamlit UI
st.set_page_config(page_title="PDF Search with LLM", page_icon="🔍", layout="wide")

st.title("🔍 PDF Search with LLM")
st.markdown("Upload PDFs and ask questions about their content using AI-powered search.")

# Sidebar with file management
with st.sidebar:
    st.header("📁 Uploaded Files")
    if st.session_state.uploaded_files:
        for filename, info in st.session_state.uploaded_files.items():
            st.write(f"📄 {filename}")
            st.caption(f"Pages: {info['pages']} | Chunks: {info['chunks']}")
    else:
        st.write("No files uploaded yet.")
    
    if st.button("🗑️ Clear All Files", type="secondary"):
        st.session_state.uploaded_files.clear()
        st.rerun()

# Main tabs
tab1, tab2 = st.tabs(["📄 Upload PDF", "🔎 Ask Questions"])

with tab1:
    st.header("Upload PDF Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF documents to index their content for searching."
    )
    
    if uploaded_files:
        # Display file summary
        total_size = sum(file.size for file in uploaded_files)
        st.info(f"Selected {len(uploaded_files)} file(s) - Total size: {total_size / (1024*1024):.1f} MB")
        
        # Check individual file sizes
        oversized_files = [f.name for f in uploaded_files if f.size > 10 * 1024 * 1024]
        if oversized_files:
            st.error(f"The following files are too large (>10MB): {', '.join(oversized_files)}")
            st.stop()
        
        # Check for duplicates
        files_to_process = []
        existing_files = []
        
        for uploaded_file in uploaded_files:
            file_bytes = uploaded_file.read()
            file_hash = get_file_hash(file_bytes)
            
            if file_hash in [info['hash'] for info in st.session_state.uploaded_files.values()]:
                existing_files.append(uploaded_file.name)
            else:
                files_to_process.append((uploaded_file.name, file_bytes, file_hash))
        
        if existing_files:
            st.warning(f"Already uploaded: {', '.join(existing_files)}")
        
        if files_to_process:
            st.success(f"Ready to process {len(files_to_process)} new file(s)")
            
            if st.button("Process All PDFs", type="primary"):
                # Create overall progress tracking
                overall_progress = st.progress(0)
                overall_status = st.empty()
                
                successful_uploads = 0
                failed_uploads = []
                
                for idx, (filename, file_bytes, file_hash) in enumerate(files_to_process):
                    overall_status.text(f"Processing file {idx + 1}/{len(files_to_process)}: {filename}")
                    
                    with st.expander(f"Processing: {filename}", expanded=True):
                        # Extract text
                        text, page_count = extract_text_from_pdf(file_bytes)
                        
                        if not text:
                            st.error(f"Could not extract text from {filename}. Please ensure it's not a scanned document.")
                            failed_uploads.append(filename)
                        else:
                            # Chunk text
                            chunks = chunk_text_smart(text)
                            
                            st.info(f"Extracted {len(text.split())} words from {page_count} pages, created {len(chunks)} chunks.")
                            
                            # Embed and store
                            if embed_and_store(chunks, filename, file_hash):
                                # Save to session state
                                st.session_state.uploaded_files[filename] = {
                                    'hash': file_hash,
                                    'pages': page_count,
                                    'chunks': len(chunks),
                                    'upload_date': datetime.now().strftime("%Y-%m-%d %H:%M")
                                }
                                
                                st.success(f"✅ Successfully processed '{filename}'!")
                                successful_uploads += 1
                            else:
                                st.error(f"Failed to process {filename}.")
                                failed_uploads.append(filename)
                    
                    # Update overall progress
                    overall_progress.progress((idx + 1) / len(files_to_process))
                
                # Final summary
                overall_progress.empty()
                overall_status.empty()
                
                if successful_uploads > 0:
                    st.success(f"🎉 Successfully processed {successful_uploads} file(s)!")
                    if successful_uploads == len(files_to_process):
                        st.balloons()
                
                if failed_uploads:
                    st.error(f"Failed to process: {', '.join(failed_uploads)}")
        else:
            st.info("No new files to process.")

with tab2:
    st.header("Ask Questions About Your Documents")
    
    if not st.session_state.uploaded_files:
        st.info("Please upload some PDF documents first to start asking questions.")
    else:
        question = st.text_input(
            "Enter your question:", 
            placeholder="e.g., What are the main findings in the document?",
            help="Ask specific questions about the content in your uploaded PDFs."
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            search_button = st.button("🔍 Search", type="primary", disabled=not question)
        with col2:
            top_k = st.selectbox("Results", [3, 5, 10], index=1, help="Number of relevant chunks to retrieve")
        
        if search_button and question:
            with st.spinner("Searching documents..."):
                context, sources = search_documents(question, top_k)
                
                if context:
                    # Generate answer
                    answer = generate_answer(context, question)
                    
                    # Display results
                    st.markdown("### 💡 Answer")
                    st.write(answer)
                    
                    # Show sources
                    st.markdown("### 📚 Sources")
                    for i, source in enumerate(sources, 1):
                        page_info = f"(Page {source['page']})" if source['page'] else ""
                        with st.expander(f"Source {i}: {source['filename']} {page_info} - Relevance: {source['score']:.3f}"):
                            st.write(source['chunk_preview'])
                else:
                    st.warning("No relevant information found in the uploaded documents.")

# Footer
st.markdown("---")
st.markdown("💡 **Tip**: Upload multiple PDFs to search across all your documents at once!")

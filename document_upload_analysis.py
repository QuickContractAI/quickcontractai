import streamlit as st
import io
import re
import PyPDF2
import docx2txt
import logging
from typing import Optional, List, Dict
import tempfile
import uuid
import os
from datetime import datetime
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def process_uploaded_document(uploaded_file) -> Optional[str]:
    """
    Extract text from an uploaded document file (PDF, DOCX, or TXT)
    
    Args:
        uploaded_file: The uploaded file object from st.file_uploader
        
    Returns:
        str: Extracted text from the document or None if extraction failed
    """
    try:
        # Get file extension
        file_name = uploaded_file.name
        file_extension = file_name.split(".")[-1].lower()
        
        # Extract text based on file type
        if file_extension == "pdf":
            return extract_text_from_pdf(uploaded_file)
        elif file_extension == "docx":
            return extract_text_from_docx(uploaded_file)
        elif file_extension == "txt":
            return extract_text_from_txt(uploaded_file)
        else:
            logger.error(f"Unsupported file type: {file_extension}")
            return None
    except Exception as e:
        logger.error(f"Error processing document: {e}", exc_info=True)
        return None

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from a PDF file"""
    text = ""
    try:
        pdf_bytes = io.BytesIO(pdf_file.getvalue())
        pdf_reader = PyPDF2.PdfReader(pdf_bytes)
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n\n"
            
        # Clean up the text
        text = clean_extracted_text(text)
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}", exc_info=True)
        return ""

def extract_text_from_docx(docx_file) -> str:
    """Extract text from a DOCX file"""
    try:
        # Read bytes from uploaded file
        docx_bytes = io.BytesIO(docx_file.getvalue())
        
        # Extract text using docx2txt
        text = docx2txt.process(docx_bytes)
        
        # Clean up the text
        text = clean_extracted_text(text)
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}", exc_info=True)
        return ""

def extract_text_from_txt(txt_file) -> str:
    """Extract text from a TXT file"""
    try:
        # Decode the uploaded file
        text = txt_file.getvalue().decode("utf-8")
        
        # Clean up the text
        text = clean_extracted_text(text)
        return text
    except Exception as e:
        logger.error(f"Error extracting text from TXT: {e}", exc_info=True)
        return ""

def clean_extracted_text(text: str) -> str:
    """Clean up the extracted text for better processing"""
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove unnecessary control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    
    return text.strip()

def chunk_document_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split document text into overlapping chunks for better analysis
    
    Args:
        text: The document text to chunk
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    chunks = []
    
    # First try to split by double newlines (paragraphs)
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If adding this paragraph would exceed chunk size
        if len(current_chunk) + len(paragraph) > chunk_size:
            # If current_chunk is already too big, add it to chunks
            if current_chunk:
                chunks.append(current_chunk)
                
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_words = words[-int(chunk_overlap/5):] if len(words) > int(chunk_overlap/5) else words
                current_chunk = ' '.join(overlap_words) + "\n\n" + paragraph
            else:
                # If paragraph alone is bigger than chunk_size, split it
                chunks.extend(split_large_paragraph(paragraph, chunk_size, chunk_overlap))
                current_chunk = ""
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def split_large_paragraph(paragraph: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split a large paragraph into smaller chunks"""
    chunks = []
    words = paragraph.split()
    
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1  # +1 for the space
        
        if current_size + word_size > chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                
                # Create overlap
                overlap_start = max(0, len(current_chunk) - int(chunk_overlap/5))
                current_chunk = current_chunk[overlap_start:]
                current_size = sum(len(word) + 1 for word in current_chunk)
                
            current_chunk.append(word)
            current_size += word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def store_document_in_temp_db(doc_text: str, filename: str, contract_type: str) -> Dict:
    """
    Store document in a temporary database structure for analysis
    This simulates storing it in Snowflake if direct upload is not implemented
    
    Returns:
        Document metadata
    """
    # Generate a unique document ID
    doc_id = str(uuid.uuid4())
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Create chunks
    chunks = chunk_document_text(doc_text)
    
    # Store document metadata in session state
    if "temp_document_db" not in st.session_state:
        st.session_state.temp_document_db = {}
    
    doc_metadata = {
        "doc_id": doc_id,
        "filename": filename,
        "contract_type": contract_type,
        "timestamp": timestamp,
        "chunk_count": len(chunks),
        "text_length": len(doc_text),
        "chunks": chunks
    }
    
    # Store in session state
    st.session_state.temp_document_db[doc_id] = doc_metadata
    
    return doc_metadata

def render_document_upload_tab(us_states):
    """Render the document upload and analysis tab"""
    st.header("Document Analysis")
    st.markdown("Upload a contract document for comprehensive analysis.")
    
    # File uploader for document
    uploaded_file = st.file_uploader("Upload a contract document (.pdf, .docx, or .txt)", 
                                     type=["pdf", "docx", "txt"])
    
    if uploaded_file is None:
        st.info("Please upload a document to begin analysis.")
        return
    
    # Contract type selection for analysis context
    contract_types = ["Service Agreement", "Employment Agreement", "Residential Lease Agreement", "Other Contract"]
    selected_contract_type = st.selectbox(
        "Select document type for better analysis:",
        options=contract_types
    )
    
    # Jurisdiction selection for legal context
    jurisdiction = st.selectbox(
        "Select governing law jurisdiction for risk analysis:",
        options=us_states,
        index=us_states.index("Massachusetts") if "Massachusetts" in us_states else 0
    )
    
    # Process the uploaded file
    doc_text = process_uploaded_document(uploaded_file)
    
    if not doc_text:
        st.error("Failed to extract text from the uploaded document.")
        return
    
    # Store document info in session state for future reference
    if "uploaded_doc_info" not in st.session_state or st.session_state.uploaded_doc_info.get("filename") != uploaded_file.name:
        # Store document in temp DB and get metadata
        doc_metadata = store_document_in_temp_db(doc_text, uploaded_file.name, selected_contract_type)
        
        st.session_state.uploaded_doc_info = {
            "filename": uploaded_file.name,
            "contract_type": selected_contract_type,
            "jurisdiction": jurisdiction,
            "doc_text": doc_text,
            "metadata": doc_metadata
        }
        
        # Clear previous analysis results if this is a new document
        if "doc_analysis_results" in st.session_state:
            del st.session_state.doc_analysis_results
    
    # Display basic document info
    word_count = len(doc_text.split())
    st.success(f"Document processed: {uploaded_file.name} ({word_count} words)")
    
    # Run analysis button
    if st.button("Analyze Document", type="primary"):
        with st.spinner("Analyzing document... This may take a moment."):
            # Use the existing analyze_contract function from contract_analysis.py
            from contract_analysis import analyze_contract
            
            # Perform the analysis
            extracted_clauses, risk_flags, summary = analyze_contract(
                contract_text=doc_text,
                contract_type=selected_contract_type,
                jurisdiction=jurisdiction
            )
            
            # Store results in session state
            st.session_state.doc_analysis_results = {
                "extracted_clauses": extracted_clauses,
                "risk_flags": risk_flags,
                "summary": summary
            }
        
        st.success(f"Analysis complete for {uploaded_file.name}")
    
    # Display results if available
    if "doc_analysis_results" in st.session_state:
        # Create three tabs for the different analysis features
        analysis_tab, risks_tab, summary_tab = st.tabs(["Clause Extraction", "Risk Analysis", "Plain Language Summary"])
        
        results = st.session_state.doc_analysis_results
        extracted_clauses = results["extracted_clauses"]
        risk_flags = results["risk_flags"]
        summary = results["summary"]
        
        # Clause Extraction Tab
        with analysis_tab:
            st.header("Key Clauses & Terms")
            
            if "error" in extracted_clauses:
                st.error(f"Error in clause extraction: {extracted_clauses['error']}")
            else:
                # Parties
                if "parties" in extracted_clauses and extracted_clauses["parties"]:
                    st.subheader("Parties")
                    for party in extracted_clauses["parties"]:
                        with st.expander(f"{party.get('name', 'Unnamed Party')} - {party.get('role', 'Role not specified')}"):
                            st.write(party.get("description", "No description provided"))
                
                # Key Dates
                if "key_dates" in extracted_clauses and extracted_clauses["key_dates"]:
                    st.subheader("Key Dates")
                    date_data = []
                    for date_item in extracted_clauses["key_dates"]:
                        date_data.append({
                            "Title": date_item.get("title", "Unnamed Date"),
                            "Date": date_item.get("date", "Not specified"),
                            "Section": date_item.get("section", "Not specified")
                        })
                    st.table(date_data)
                
                # Display other extracted elements...
                # (This follows the same pattern as in contract_analysis.py)
                # Payment Terms
                if "payment_terms" in extracted_clauses and extracted_clauses["payment_terms"]:
                    st.subheader("Payment Terms")
                    for term in extracted_clauses["payment_terms"]:
                        with st.expander(f"{term.get('description', 'Payment Term')} (Section: {term.get('section', 'N/A')})"):
                            st.write(term.get("details", "No details provided"))
                
                # And so on for other elements...
        
        # Risk Analysis Tab
        with risks_tab:
            st.header("Risk Assessment")
            
            if isinstance(risk_flags, list):
                if not risk_flags:
                    st.success("No significant risks identified in this document.")
                else:
                    # Count risks by level
                    risk_counts = {"High": 0, "Medium": 0, "Low": 0}
                    for risk in risk_flags:
                        level = risk.get("risk_level", "Unknown")
                        if level in risk_counts:
                            risk_counts[level] += 1
                    
                    # Display risk summary
                    st.markdown(f"""
                    ### Risk Summary
                    - **High Risk Items:** {risk_counts["High"]}
                    - **Medium Risk Items:** {risk_counts["Medium"]}
                    - **Low Risk Items:** {risk_counts["Low"]}
                    """)
                    
                    # Display risks grouped by level
                    for level in ["High", "Medium", "Low"]:
                        level_risks = [r for r in risk_flags if r.get("risk_level") == level]
                        if level_risks:
                            if level == "High":
                                st.markdown(f"### ⚠️ High Risk Items ({len(level_risks)})")
                            elif level == "Medium":
                                st.markdown(f"### ⚠ Medium Risk Items ({len(level_risks)})")
                            else:
                                st.markdown(f"### ℹ️ Low Risk Items ({len(level_risks)})")
                            
                            for i, risk in enumerate(level_risks):
                                with st.expander(f"{i+1}. {risk.get('category', 'Risk')} - {risk.get('section', 'Section')}"):
                                    st.markdown(f"**Description:** {risk.get('description', 'No description provided')}")
                                    st.markdown(f"**Recommendation:** {risk.get('recommendation', 'No recommendation provided')}")
            else:
                st.error("Error processing risk analysis results.")
        
        # Summary Tab
        with summary_tab:
            st.header("Plain Language Summary")
            
            if summary and isinstance(summary, str) and len(summary) > 10:
                st.markdown(summary)
                
                # Add a download button for the summary
                st.download_button(
                    label="Download Summary",
                    data=summary,
                    file_name=f"Document_Summary.md",
                    mime="text/markdown"
                )
            else:
                st.error("Error generating plain language summary.")
        
        # Button to reset analysis
        if st.button("Reset Analysis"):
            # Clear the analysis results from session state
            if "doc_analysis_results" in st.session_state:
                del st.session_state.doc_analysis_results
            st.rerun()
            
def render_document_qa_with_uploaded():
    """
    Render a document Q&A tab for the currently uploaded document
    This is a variant of the document Q&A that works specifically with 
    the document that's been uploaded rather than the entire database
    """
    st.header("Document Q&A")
    st.markdown("Ask specific questions about your uploaded document.")
    
    # Check if a document has been uploaded
    if "uploaded_doc_info" not in st.session_state or not st.session_state.uploaded_doc_info.get("doc_text"):
        st.warning("Please upload a document in the 'Document Analysis' tab first.")
        return
    
    # Get the uploaded document info
    doc_info = st.session_state.uploaded_doc_info
    filename = doc_info.get("filename", "Unnamed document")
    doc_text = doc_info.get("doc_text", "")
    
    # Display document info
    st.success(f"Ready to answer questions about: {filename}")
    
    # Initialize chat history if not exists
    if "uploaded_qa_history" not in st.session_state:
        st.session_state.uploaded_qa_history = []
    
    # Display chat history
    for message in st.session_state.uploaded_qa_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    user_question = st.chat_input("Ask a question about this document...")
    
    if user_question:
        # Add user message to chat history
        st.session_state.uploaded_qa_history.append({"role": "user", "content": user_question})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document..."):
                response = generate_uploaded_document_qa_response(user_question, doc_text, filename)
                st.markdown(response)
                # Add assistant message to chat history
                st.session_state.uploaded_qa_history.append({"role": "assistant", "content": response})
    
    # Reset button to clear the chat history
    if st.session_state.uploaded_qa_history and st.button("Clear Chat History"):
        st.session_state.uploaded_qa_history = []
        st.rerun()

def generate_uploaded_document_qa_response(user_question, doc_text, filename):
    """
    Generate a response to a user question about the uploaded document
    using the document text directly
    """
    try:
        # Break the document into chunks for more relevant context
        chunks = chunk_document_text(doc_text)
        
        # Simple keyword-based retrieval for relevant chunks
        # In a real implementation, this would use better semantic search
        query_terms = set(user_question.lower().split())
        chunk_scores = []
        
        for i, chunk in enumerate(chunks):
            # Simple scoring: count how many query terms appear in the chunk
            chunk_words = set(chunk.lower().split())
            score = len(query_terms.intersection(chunk_words))
            chunk_scores.append((i, score, chunk))
        
        # Sort chunks by relevance score
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 3 most relevant chunks
        top_chunks = [chunk for _, score, chunk in chunk_scores[:3] if score > 0]
        
        if not top_chunks:
            return "I couldn't find information in the document to answer that question. Please try a different question."
        
        # Format the context from retrieved chunks
        context = "\n\n".join(top_chunks)
        
        # Use the Gemini model for answer generation
        llm = get_draft_model()  # Reuse the Gemini model from the main application
        
        if not llm:
            return "Sorry, the language model is not available. Please check your API keys and try again."
        
        # Create a custom prompt for Q&A
        template = """
        You are a legal assistant analyzing a contract document to answer a specific question.
        
        Document Name: {document_name}
        
        Use only the following context from the document to answer the question.
        If the context doesn't contain the information needed to answer the question, say so clearly.
        Don't make up information that's not in the provided context.
        
        Context from document:
        {context}
        
        Question: {question}
        
        Answer:
        """
        
        prompt = PromptTemplate(
            input_variables=["document_name", "context", "question"],
            template=template,
        )
        
        # Create and run the chain
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(document_name=filename, context=context, question=user_question)
        
        if not response:
            return "I couldn't generate a response. Please try a different question."
        
        return response
    except Exception as e:
        logger.error(f"Error generating Q&A response: {e}", exc_info=True)
        return f"Sorry, an error occurred while processing your question: {str(e)}"
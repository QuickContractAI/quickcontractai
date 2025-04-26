import streamlit as st
import os
import io
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import logging
from datetime import datetime, date
import re
import time
import json
from typing import List, Dict, Any, Optional, Tuple
import difflib


# LangChain imports
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.schema import Document

# Ollama models
from langchain_community.llms import Ollama
from langchain.chat_models import ChatOllama

# Gemini model
from langchain_google_genai import ChatGoogleGenerativeAI

# Snowflake integration
import snowflake.connector

from snowflake.snowpark import Session
from langchain.vectorstores.base import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
# Load environment variables
import dotenv
dotenv.load_dotenv()

# --- Configuration & Initialization ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
APP_NAME = "QuickContractAI"

# --- Snowflake Cortex RAG Integration ---
class SnowflakeCortexRetriever:
    """Custom retriever for Snowflake Cortex Search Service"""
    
    def __init__(self, session=None):
        self.session = session or self._create_snowflake_session()
        
    def _create_snowflake_session(self):
        """Create Snowflake session using environment variables or hardcoded values"""
        # Use connection parameters with defaults
        connection_parameters = {
            "account": os.getenv("SNOWFLAKE_ACCOUNT", "SFEDU02-PDB57018"),
            "user": os.getenv("SNOWFLAKE_USER", "CAT"),
            "password": os.getenv("SNOWFLAKE_PASSWORD", ""),  
            "role": os.getenv("SNOWFLAKE_ROLE", "TRAINING_ROLE"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "DAMG7374"),
            "database": os.getenv("SNOWFLAKE_DATABASE", "LAWS_CONTRACTS"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA", "TEXT")
        }
        
        logger.info(f"Connection parameters: account={connection_parameters['account']}, user={connection_parameters['user']}, role={connection_parameters['role']}, warehouse={connection_parameters['warehouse']}, database={connection_parameters['database']}, schema={connection_parameters['schema']}")
        
        try:
            # Using snowflake.connector instead of Snowpark Session
            import snowflake.connector
            logger.info("Attempting to connect to Snowflake...")
            conn = snowflake.connector.connect(**connection_parameters)
            logger.info("Connected to Snowflake!")
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {e}")
            return None
    
    def get_full_chunk_text(self, cursor, doc_index, chunk_index):
        """
        Retrieve the full text of a chunk based on its document and chunk indices
        """
        query = f"""
        SELECT CHUNK 
        FROM LAWS_CONTRACTS.TEXT.DOCS_CHUNKS_TABLE
        WHERE DOC_INDEX = {doc_index} AND CHUNK_INDEX = {chunk_index}
        """
        
        logger.info(f"Executing full chunk query for doc_index={doc_index}, chunk_index={chunk_index}")
        cursor.execute(query)
        result = cursor.fetchone()
        
        if result:
            logger.info(f"Found full chunk text with length: {len(result[0]) if result[0] else 0}")
            return result[0]
        else:
            logger.warning(f"Full text not found for doc_index={doc_index}, chunk_index={chunk_index}")
            return "Full text not found"
            
    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """Retrieve relevant document snippets using Cortex Search Service"""
        if not self.session:
            logger.warning("No Snowflake session available. Returning empty results.")
            return []
            
        try:
            # Create cursor
            logger.info("Creating Snowflake cursor...")
            cursor = self.session.cursor()
            
            # Set limit for results
            limit = kwargs.get('k', 5)
            logger.info(f"Search limit set to: {limit}")

            cleaned_query = query.replace("'", "''").replace("\n", " ").replace("\r", " ")
            logger.info(f"Cleaned search query: {cleaned_query}")

            
            # Using the exact format from the successful implementation
            search_query = f"""
            SELECT PARSE_JSON(
            SNOWFLAKE.CORTEX.SEARCH_PREVIEW(
                'LAWS_CONTRACTS.TEXT.laws_search_service',
                '{{
                "query": "{cleaned_query}", 
                "columns": ["SOURCE", "CHUNK", "DOC_INDEX", "CHUNK_INDEX"],
                "limit": {limit}
                }}'
            )
            )['results'] AS results;
            """
            
            logger.info(f"Executing Snowflake Cortex search query...")
            cursor.execute(search_query)
            
            # Fetch results
            logger.info("Fetching search results...")
            results = cursor.fetchall()
            logger.info(f"Search returned {len(results)} result rows")
            
            # Try to log the raw results for debugging
            try:
                for i, row in enumerate(results):
                    result_str = str(row[0])
                    logger.info(f"Raw result {i+1}: {result_str[:200]}...")  # Log first 200 chars
            except Exception as e:
                logger.error(f"Error logging raw results: {e}")
            
            # Convert to Documents
            documents = []
            
            for row in results:
                # The row contains a JSON string that we need to parse
                result_json = row[0]
                logger.info(f"Processing result JSON type: {type(result_json)}")
                
                if isinstance(result_json, str):
                    logger.info("Parsing result JSON from string...")
                    result_data = json.loads(result_json)
                else:
                    logger.info("Using result JSON directly...")
                    result_data = result_json
                
                # Try to log the parsed data structure
                try:
                    if isinstance(result_data, list):
                        logger.info(f"Result data is a list with {len(result_data)} items")
                    elif isinstance(result_data, dict):
                        logger.info(f"Result data is a dict with keys: {', '.join(result_data.keys())}")
                    else:
                        logger.info(f"Result data is type: {type(result_data)}")
                except Exception as e:
                    logger.error(f"Error logging result data structure: {e}")
                    
                # Process results
                if isinstance(result_data, list):
                    for item in result_data:
                        # Store doc_index and chunk_index
                        doc_index = item.get("DOC_INDEX")
                        chunk_index = item.get("CHUNK_INDEX")
                        source = item.get("SOURCE", "")
                        
                        logger.info(f"Processing list item: source={source}, doc_index={doc_index}, chunk_index={chunk_index}")
                        
                        # Get full chunk text if possible
                        if doc_index is not None and chunk_index is not None:
                            try:
                                chunk_text = self.get_full_chunk_text(cursor, doc_index, chunk_index)
                            except Exception as e:
                                logger.error(f"Error retrieving full chunk: {str(e)}")
                                chunk_text = item.get("CHUNK", "")
                        else:
                            chunk_text = item.get("CHUNK", "")
                            
                        logger.info(f"Chunk text length: {len(chunk_text) if chunk_text else 0}")
                            
                        # Create Document object
                        doc = Document(
                            page_content=chunk_text,
                            metadata={
                                "source": source,
                                "doc_index": doc_index,
                                "chunk_index": chunk_index
                            }
                        )
                        documents.append(doc)
                elif isinstance(result_data, dict):
                    # This might be the case where a single result is returned as a dict
                    # or the results are nested under a key
                    
                    # Check if there's a nested results array
                    if "results" in result_data and isinstance(result_data["results"], list):
                        logger.info(f"Found nested results array with {len(result_data['results'])} items")
                        nested_results = result_data["results"]
                        
                        for item in nested_results:
                            doc_index = item.get("DOC_INDEX")
                            chunk_index = item.get("CHUNK_INDEX")
                            source = item.get("SOURCE", "")
                            
                            logger.info(f"Processing nested item: source={source}, doc_index={doc_index}, chunk_index={chunk_index}")
                            
                            doc = Document(
                                page_content=item.get("CHUNK", ""),
                                metadata={
                                    "source": source,
                                    "doc_index": doc_index,
                                    "chunk_index": chunk_index
                                }
                            )
                            documents.append(doc)
                    else:
                        # Handle single result case
                        doc_index = result_data.get("DOC_INDEX")
                        chunk_index = result_data.get("CHUNK_INDEX")
                        source = result_data.get("SOURCE", "")
                        
                        logger.info(f"Processing single dict result: source={source}, doc_index={doc_index}, chunk_index={chunk_index}")
                        
                        doc = Document(
                            page_content=result_data.get("CHUNK", ""),
                            metadata={
                                "source": source,
                                "doc_index": doc_index,
                                "chunk_index": chunk_index
                            }
                        )
                        documents.append(doc)
                else:
                    logger.warning(f"Unhandled result_data type: {type(result_data)}")
            
            # Close cursor
            logger.info(f"Search complete, found {len(documents)} documents")
            cursor.close()
            
            return documents
        except Exception as e:
            logger.error(f"Failed to retrieve documents from Snowflake Cortex: {e}")
            logger.error(f"Error details: {str(e)}")
            return []
        
# --- Model Initialization ---
def get_refiner_model():
    """Get the gemma3 model for input refinement"""
    return Ollama(
        model="gemma3:12b-it-qat",
        callbacks=[StreamingStdOutCallbackHandler()],
        temperature=0.1,
    )

def get_validation_model():
    """Get the ShieldGemma model for validation"""
    return Ollama(
        model="shieldgemma",
        callbacks=[StreamingStdOutCallbackHandler()],
        temperature=0.1,
    )

def get_consistency_model():
    """Get the Phi4 model for consistency validation"""
    return Ollama(
        model="phi4",
        callbacks=[StreamingStdOutCallbackHandler()],
        temperature=0.1,
    )

def get_draft_model():
    """Get the Gemini model for main draft generation"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Google API key not found in environment variables.")
        return None
        
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-pro-exp-03-25",
        google_api_key=api_key,
        temperature=0.5,
    )

# --- LangChain Chain Definitions ---
def create_refinement_chain():
    """Create a chain for refining user inputs"""
    llm = get_refiner_model()
    
    template = """
    You are a helpful assistant tasked with refining and improving legal contract language.
    Review the following input for a {field_name} in a {contract_type} contract.
    Your task is to check for:
    1. Grammar and clarity issues
    2. Placeholder text that needs to be replaced (like [X], $[Value], etc.)
    3. Incomplete sentences or unclear language
    
    CONTEXT:
    {context}

    Original input:
    {original_text}
    
    If you find issues, improve the text while preserving the original intent.
    
    Return the improved text only, strictly WITHOUT explanations or additional formatting. Also, don't generate alternatives to suggested reviews.
    """
    
    prompt = PromptTemplate(
        input_variables=["field_name", "contract_type", "original_text"],
        template=template,
    )
    
    return LLMChain(llm=llm, prompt=prompt, output_key="refined_text")

def create_placeholder_detection_chain():
    """Create a chain for detecting leftover placeholders"""
    llm = get_validation_model()
    
    template = """
    You are a validation assistant. Your task is to identify any placeholders or template language in the given contract text.
    Placeholders might look like:
    - [Insert X]
    - [Company Name]
    - <PLACEHOLDER>
    - $VARIABLE
    - TBD
    - To be determined
    
    Contract text:
    {contract_text}
    
    Return a JSON list of any placeholders you find with their location and context. Example format:
    [
      {{"type":"placeholder","field":"Section 2.1","text":"[Insert payment terms]","message":"Found template placeholder"}},
      {{"type":"placeholder","field":"Section 5","text":"TBD","message":"Found incomplete section"}}
    ]
    
    If no placeholders are found, return an empty list: []
    """
    
    prompt = PromptTemplate(
        input_variables=["contract_text"],
        template=template,
    )
    
    return LLMChain(llm=llm, prompt=prompt, output_key="placeholder_issues")

def create_consistency_validation_chain():
    """Create a chain for validating contract consistency"""
    llm = get_consistency_model()
    
    template = """
    You are a contract validation assistant. Your task is to check the consistency and completeness of the given contract.
    
    Contract type: {contract_type}
    User inputs: {user_inputs}
    
    Contract text:
    {contract_text}
    
    Check for the following issues:
    1. All sections are numbered correctly and in sequence
    2. All required user inputs are properly reflected in the document
    3. No contradictions between different sections
    4. All cross-references are valid and correct
    
    Return a JSON list of any issues you find. Example format:
    [
      {{"type":"consistency","field":"Numbering","message":"Section 3 is followed by Section 5, missing Section 4"}},
      {{"type":"consistency","field":"UserInput","message":"Company name 'Acme Inc' is not consistently used throughout"}}
    ]
    
    If no issues are found, return an empty list: []
    """
    
    prompt = PromptTemplate(
        input_variables=["contract_type", "user_inputs", "contract_text"],
        template=template,
    )
    
    return LLMChain(llm=llm, prompt=prompt, output_key="consistency_issues")

def create_draft_generation_chain():
    """Create a chain for generating the main contract draft"""
    llm = get_draft_model()
    
    template = """
    You are a professional legal document drafting assistant. Create a complete and balanced {contract_type} based on the following details. 
    The document should be thorough yet concise, with appropriate legal language and formatting.
    
    Contract Type: {contract_type}
    Jurisdiction: {jurisdiction}
    
    User Inputs:
    {user_inputs}
    
    {precedent_extracts}
    
    Format the document with:
    1. Clear section numbering (e.g., "SECTION 1. DEFINITIONS")
    2. Subsections with decimal numbering (e.g., "1.1. Term")
    3. Appropriate indentation for lists and clauses
    4. Standard legal formatting conventions
    
    Include these standard sections (as appropriate for this contract type):
    - Parties and Recitals
    - Definitions
    - Term and Termination
    - Rights and Obligations
    - Payment Terms (if applicable)
    - Representations and Warranties
    - Limitation of Liability
    - Indemnification
    - Confidentiality
    - Governing Law and Dispute Resolution
    - General Provisions (including Notice, Assignment, Severability, Entire Agreement)
    
    STOP before creating any signature block.
    """
    
    prompt = PromptTemplate(
        input_variables=["contract_type", "jurisdiction", "user_inputs", "precedent_extracts"],
        template=template,
    )
    
    return LLMChain(llm=llm, prompt=prompt, output_key="generated_draft")

def create_clause_refinement_chain():
    """Create a chain for refining specific clauses with issues"""
    llm = get_refiner_model()
    
    template = """
    You are a contract clause refinement specialist. Fix the following issue in a {contract_type} contract:
    
    Issue: {issue_description}
    
    Original Clause:
    {original_clause}
    
    User Requirements:
    {user_inputs}
    
    Rewrite only this specific clause to fix the issue while maintaining legal accuracy and clarity.
    Return only the corrected clause text without explanations or comments.
    """
    
    prompt = PromptTemplate(
        input_variables=["contract_type", "issue_description", "original_clause", "user_inputs"],
        template=template,
    )
    
    return LLMChain(llm=llm, prompt=prompt, output_key="refined_clause")

# --- PDF Generation Class ---
class ContractPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 10)
        doc_title = self.title if hasattr(self, 'title') else 'Generated Contract'
        safe_title = doc_title.encode('latin-1', 'replace').decode('latin-1')
        self.cell(0, 10, safe_title, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', border=0, new_x=XPos.RIGHT, new_y=YPos.TOP, align='C')

    def chapter_title(self, title):
        self.set_font('helvetica', 'B', 12)
        safe_title = title.encode('latin-1', 'replace').decode('latin-1')
        self.multi_cell(0, 6, safe_title, border=0, align='L')
        self.ln(4)

    def chapter_body(self, body_text):
        self.set_font('helvetica', '', 10)
        try:
            encoded_text = body_text.encode('latin-1', 'replace').decode('latin-1')
            self.multi_cell(0, 5, encoded_text, border=0, align='L')
        except Exception as e:
            logger.warning(f"PDF encoding/rendering issue for text segment: {e}")
            safe_text = body_text.encode('latin-1', 'replace').decode('latin-1')
            self.multi_cell(0, 5, safe_text, border=0, align='L')

    def add_signature_lines(self, party1_label, party2_label):
        self.ln(15)
        self.set_font('helvetica', '', 10)
        line_len = 80
        sig_block_width = 100
        
        safe_party1 = party1_label.encode('latin-1', 'replace').decode('latin-1')
        safe_party2 = party2_label.encode('latin-1', 'replace').decode('latin-1')
        
        def draw_party_sig(label):
            self.cell(line_len, 10, "_" * 40, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
            self.cell(sig_block_width, 6, f"By: {label}", border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
            self.cell(sig_block_width, 6, "Name:", border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
            self.cell(sig_block_width, 6, "Title:", border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
            self.cell(sig_block_width, 6, "Date:", border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
            self.ln(8)
            
        draw_party_sig(safe_party1)
        draw_party_sig(safe_party2)

def create_pdf_from_generated_text(generated_text, title, input_data):
    """Creates a PDF document in memory from the generated text."""
    pdf = ContractPDF(orientation='P', unit='mm', format='A4')
    pdf.set_title(title)
    pdf.set_author(APP_NAME)
    pdf.set_margins(left=20, top=15, right=20)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.alias_nb_pages()
    pdf.set_font('helvetica', '', 10)
    
    # Determine party names for signature block
    party1_keys = ['company_name', 'party1_name', 'client_name', 'employer_name', 'landlord_name']
    party2_keys = ['distributor_name', 'party2_name', 'contractor_name', 'employee_name', 'tenant_name']
    
    party1_raw = input_data.get(next((k for k in party1_keys if k in input_data and input_data.get(k)), None), "Party 1")
    party2_raw = input_data.get(next((k for k in party2_keys if k in input_data and input_data.get(k)), None), "Party 2")
    
    party1 = str(party1_raw) if party1_raw else "Party 1"
    party2 = str(party2_raw) if party2_raw else "Party 2"

    lines = generated_text.split('\n')
    for line in lines:
        line_stripped = line.strip()
        
        # Skip signature-related lines that might be in the text
        sig_keywords = ("in witness whereof", "agreed and accepted by:", "by:", "name:", "title:", 
                      "date:", "signature:", "party 1:", "party 2:", "landlord:", "tenant:", 
                      "company:", "distributor:", "client:", "contractor:", "employer:", "employee:")
        if line_stripped.lower().startswith(sig_keywords) and len(line_stripped) < 50:
            logger.info(f"Filtering likely sig line: '{line_stripped}'")
            continue
        
        # Skip review flags
        if line_stripped.startswith("[Review Recommended:"):
            logger.info(f"Filtering review flag: '{line_stripped}'")
            continue
            
        # Handle empty lines
        if not line_stripped:
            pdf.ln(3)
            continue

        # PDF Formatting Logic
        if line_stripped.startswith("**") and line_stripped.endswith("**") and len(line_stripped) > 4:
            # Main Section Titles (like **ARTICLE X: TITLE**)
            pdf.ln(4)
            pdf.set_font('helvetica', 'B', 12)
            pdf.chapter_title(line_stripped.strip('* '))
            pdf.ln(2)
        elif (line_stripped.startswith("ARTICLE ") or line_stripped.startswith("SECTION ") or 
             line_stripped.startswith("Section ") or re.match(r"^[IVXLCDM]+\.\s+", line_stripped)) and ':' in line_stripped:
            # Numbered Section Titles
            pdf.ln(4)
            pdf.set_font('helvetica', 'B', 11)
            pdf.chapter_body(line_stripped)
            pdf.ln(2)
        elif line_stripped.isupper() and len(line_stripped) > 5 and len(line_stripped) < 50 and not re.search(r'[\.\(\)]', line_stripped):
            # Simple ALL CAPS headings
            pdf.ln(3)
            pdf.set_font('helvetica', 'B', 10)
            pdf.chapter_body(line_stripped)
            pdf.ln(2)
        elif line_stripped.endswith(":") and len(line_stripped) < 80 and not re.match(r"^\s*(\d+(\.\d+)*\.?|\*|-|[a-z]\)|\(i\)|\(ii\)|\(iii\)|\(iv\)|\(v\))\s+", line_stripped, re.IGNORECASE):
            # Sub-headings ending in a colon
            pdf.ln(2)
            pdf.set_font('helvetica', 'B', 10)
            pdf.chapter_body(line_stripped)
            pdf.ln(1)
        elif re.match(r"^\s*(\d+(\.\d+)*\.?|\([a-z]\)|[a-z]\.|\([ivxlcdm]+\)|[ivxlcdm]+\.|\*|-)\s+", line_stripped, re.IGNORECASE):
            # Numbered/Bulleted List Items
            pdf.set_font('helvetica', '', 10)
            match = re.match(r"^(\s*)(\d+(\.\d+)*\.?|\([a-z]\)|[a-z]\.|\([ivxlcdm]+\)|[ivxlcdm]+\.|\*|-)\s+", line_stripped, re.IGNORECASE)
            if match:
                prefix_len = len(match.group(0))
                # Calculate basic indent based on whitespace before the list marker
                indent = len(match.group(1)) * 2  # Simple indent factor
                base_indent = 5 + indent

                # Additional indent for deeper levels
                list_marker = match.group(2)
                if re.match(r"^\s*\([a-z]\)|\([ivxlcdm]+\)", list_marker, re.IGNORECASE):
                    base_indent += 5  # Additional indent for (a), (i) etc.
                elif re.match(r"^\s*[a-z]\.", list_marker, re.IGNORECASE):
                    base_indent += 5  # Additional indent for a., b. etc.
                elif '.' in list_marker and list_marker.count('.') > 1:
                    base_indent += 5  # Additional indent for multi-level numbers like 1.1.1

                pdf.set_left_margin(20 + base_indent)  # Set temporary margin for this item
                pdf.ln(0.1)  # Tiny line break needed sometimes before changing margin
                pdf.multi_cell(0, 5, line_stripped[prefix_len:], align='L')
                pdf.set_left_margin(20)  # Reset margin
                pdf.ln(1)
            else:
                # Fallback for any issues
                pdf.chapter_body(line_stripped)
                pdf.ln(1)
        else:
            # Regular paragraph text
            pdf.set_font('helvetica', '', 10)
            pdf.chapter_body(line_stripped)
            pdf.ln(1)

    # Add signature lines at the end
    pdf.add_signature_lines(party1, party2)
    
    buffer = io.BytesIO()
    try:
        # Get PDF output bytes
        pdf_output_bytes = pdf.output()
        buffer.write(pdf_output_bytes)
        buffer.seek(0)
        return buffer
    except Exception as e:
        logger.error(f"PDF generation output error: {e}", exc_info=True)
        st.error(f"Failed to generate PDF bytes: {e}")
        return None

# --- Contract Type Definitions ---
SERVICE_AGREEMENT_FIELDS = [
    {"key": "client_name", "label": "Client Name", "type": "text", "refine": False, "required": True},
    {"key": "client_address", "label": "Client Address", "type": "text_area", "refine": True, "required": True},
    {"key": "provider_name", "label": "Service Provider Name", "type": "text", "refine": False, "required": True},
    {"key": "provider_address", "label": "Service Provider Address", "type": "text_area", "refine": True, "required": True},
    {"key": "services_description", "label": "Description of Services", "type": "text_area", "refine": True, "required": True, "height": 150},
    {"key": "fees", "label": "Fees and Payment Terms", "type": "text_area", "refine": True, "required": True},
    {"key": "start_date", "label": "Start Date", "type": "date_input", "refine": False, "required": True},
    {"key": "end_date", "label": "End Date", "type": "date_input", "refine": False, "required": False},
    {"key": "termination", "label": "Termination Provisions", "type": "text_area", "refine": True, "required": False},
    {"key": "intellectual_property", "label": "Intellectual Property Rights", "type": "text_area", "refine": True, "required": False},
    {"key": "confidentiality", "label": "Confidentiality Terms", "type": "text_area", "refine": True, "required": False},
    {"key": "dispute_resolution", "label": "Dispute Resolution", "type": "text_area", "refine": True, "required": False},
]

EMPLOYMENT_AGREEMENT_FIELDS = [
    {"key": "employer_name", "label": "Employer Name", "type": "text", "refine": False, "required": True},
    {"key": "employer_address", "label": "Employer Address", "type": "text_area", "refine": True, "required": True},
    {"key": "employee_name", "label": "Employee Name", "type": "text", "refine": False, "required": True},
    {"key": "employee_address", "label": "Employee Address", "type": "text_area", "refine": True, "required": True},
    {"key": "position", "label": "Position/Title", "type": "text", "refine": False, "required": True},
    {"key": "duties", "label": "Duties and Responsibilities", "type": "text_area", "refine": True, "required": True, "height": 150},
    {"key": "salary", "label": "Salary", "type": "text", "refine": True, "required": True},
    {"key": "bonus", "label": "Bonus/Commission Structure", "type": "text_area", "refine": True, "required": False},
    {"key": "start_date", "label": "Start Date", "type": "date_input", "refine": False, "required": True},
    {"key": "benefits", "label": "Benefits", "type": "text_area", "refine": True, "required": False},
    {"key": "termination", "label": "Termination Provisions", "type": "text_area", "refine": True, "required": True},
    {"key": "confidentiality", "label": "Confidentiality Terms", "type": "text_area", "refine": True, "required": False},
    {"key": "non_compete", "label": "Non-Compete (Optional)", "type": "text_area", "refine": True, "required": False},
]

RESIDENTIAL_LEASE_FIELDS = [
    {"key": "landlord_name", "label": "Landlord Name", "type": "text", "refine": False, "required": True},
    {"key": "landlord_address", "label": "Landlord Address", "type": "text_area", "refine": True, "required": True},
    {"key": "tenant_name", "label": "Tenant Name", "type": "text", "refine": False, "required": True},
    {"key": "tenant_address", "label": "Tenant Current Address", "type": "text_area", "refine": True, "required": False},
    {"key": "property_address", "label": "Property Address", "type": "text_area", "refine": True, "required": True},
    {"key": "property_description", "label": "Property Description", "type": "text_area", "refine": True, "required": True},
    {"key": "start_date", "label": "Lease Start Date", "type": "date_input", "refine": False, "required": True},
    {"key": "end_date", "label": "Lease End Date", "type": "date_input", "refine": False, "required": True},
    {"key": "rent_amount", "label": "Monthly Rent", "type": "text", "refine": False, "required": True},
    {"key": "rent_due_date", "label": "Rent Due Date", "type": "text", "refine": False, "required": True},
    {"key": "late_fees", "label": "Late Fee Policy", "type": "text_area", "refine": True, "required": False},
    {"key": "security_deposit", "label": "Security Deposit", "type": "text", "refine": False, "required": True},
    {"key": "utilities", "label": "Utilities Responsibility", "type": "text_area", "refine": True, "required": True},
    {"key": "pets_policy", "label": "Pets Policy", "type": "text_area", "refine": True, "required": False},
    {"key": "smoking_policy", "label": "Smoking Policy", "type": "text", "refine": False, "required": False},
]

AGREEMENT_QUESTIONS = {
    "Service Agreement": SERVICE_AGREEMENT_FIELDS,
    "Employment Agreement": EMPLOYMENT_AGREEMENT_FIELDS,
    "Residential Lease Agreement": RESIDENTIAL_LEASE_FIELDS,
}

US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
    "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
    "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico",
    "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
    "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
    "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
]

# --- Contract Generation Functions --
def refine_text_field(original_text, field_context_label, contract_type, all_inputs=None):
    """Refine text field using Llama 3.2 with context from other relevant fields"""
    try:
        # Skip refinement for certain fields
        if field_context_label.lower().endswith("address") and len(original_text.split(",")) <= 1:
            return f"{original_text}, [City], [State] [ZIP]"  
                    
        llm = get_refiner_model()
        
        # Build context dictionary of relevant fields based on contract type
        context_fields = {}
        if all_inputs:
            # Common fields to always include
            for key in ['client_name', 'provider_name', 'employer_name', 'employee_name', 
                        'landlord_name', 'tenant_name']:
                if key in all_inputs and all_inputs[key]:
                    context_fields[key] = all_inputs[key]
            
            # Add relevant fields based on contract type
            if contract_type == "Service Agreement":
                relevant_keys = ['client_name', 'provider_name', 'start_date', 'end_date']
            elif contract_type == "Employment Agreement":
                relevant_keys = ['employer_name', 'employee_name', 'position', 'salary', 'start_date']
            elif contract_type == "Residential Lease Agreement":
                relevant_keys = ['landlord_name', 'tenant_name', 'property_address', 
                                 'rent_amount', 'rent_due_date', 'start_date', 'end_date']
            
            for key in relevant_keys:
                if key in all_inputs and all_inputs[key]:
                    # Convert dates to string format
                    if isinstance(all_inputs[key], date):
                        context_fields[key] = all_inputs[key].strftime('%Y-%m-%d')
                    else:
                        context_fields[key] = all_inputs[key]
        
        # Build context string
        context_str = "\n".join([f"{key.replace('_', ' ').title()}: {value}" 
                               for key, value in context_fields.items()])
        
        template = """
        You are refining text for a {field_name} in a {contract_type} contract.

        CONTEXT:
        {context}

        ORIGINAL TEXT:
        {original_text}

        Your task:
        1. Fix grammar and clarity issues
        2. Replace placeholders with actual values from context
        3. Make language more complete and professional

        IMPORTANT FORMATTING RULES:
        - DO NOT include any prefix like "[Review Recommended:]" 
        - DO NOT include any "Alternatively:" sections
        - DO NOT include "Note:" or explanations of your changes
        - DO NOT include instructions or comments to the user
        - Return ONLY the improved text without any additional formatting or commentary
        - If the original text is already good, return exactly the same text

        Output should be ONLY the plain text with NO prefixes or tags.
        """
        
        # Run the LLM directly
        input_text = template.format(
            field_name=field_context_label,
            contract_type=contract_type,
            context=context_str,
            original_text=original_text
        )
        
        result = llm.invoke(input_text)
        
        # Clean up the result
        result_text = clean_llm_output(result.strip(), original_text)
        
        # If result is very similar to original, just return original
        if is_essentially_same(result_text, original_text):
            return original_text
            
        return result_text
    except Exception as e:
        logger.error(f"Error during refinement for '{field_context_label}': {e}", exc_info=True)
        return original_text

def clean_llm_output(output_text, original_text):
    """Clean LLM output to remove common instruction artifacts"""
    
    # Remove common instruction patterns
    patterns_to_remove = [
        r"ORIGINAL TEXT:.*?(?=\n\n)",
        r"IMPROVED TEXT:.*?\n",
        r"CONTEXT:.*?(?=\n\n)",
        r"Your task is to.*?\n",
        r"Return ONLY the improved.*?\n",
        r"If you find issues.*?\n",
        r"If the original text is already.*?\n"
    ]
    
    cleaned_text = output_text
    for pattern in patterns_to_remove:
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    
    # Check for review needed (for UI warning flag)
    needs_review = False
    placeholder_pattern = r"\[review recommended:.*?\]"
    if re.search(placeholder_pattern, cleaned_text.lower()):
        needs_review = True
    
    # Further cleaning - split into lines and remove instruction-like lines
    lines = cleaned_text.split('\n')
    cleaned_lines = []
    
    # Remove "Alternatively:" sections, notes, and ALL review recommendations
    in_alternative_section = False
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Skip instruction-like lines
        if any(keyword in line_lower for keyword in [
            "original text:", "improved text:", "return only", "task is to", 
            "grammar and clarity", "incomplete sentences"
        ]):
            continue
            
        # Skip "Alternatively:" sections
        if line_lower.startswith("alternatively:"):
            in_alternative_section = True
            continue
            
        if in_alternative_section and line_lower.strip() == "":
            in_alternative_section = False
            
        if in_alternative_section:
            continue
            
        # Skip notes at the end
        if line_lower.startswith("note:"):
            continue
            
        # Remove ALL review recommendation prefixes
        if "[review recommended:" in line_lower:
            # Remove the prefix and keep only the content after "]"
            content = re.sub(r"\[review recommended:.*?\]", "", line, flags=re.IGNORECASE).strip()
            if content:  # Only add if there's content left
                cleaned_lines.append(content)
        else:
            cleaned_lines.append(line)
    
    # Join and strip
    final_text = "\n".join(cleaned_lines).strip()
    
    # If we've lost too much, return original
    if len(final_text) < len(original_text) * 0.5 and len(original_text) > 20:
        return original_text
        
    # Store the needs_review flag in the context where appropriate
    # This can be done via return value or other mechanism
    return final_text

def is_essentially_same(text1, text2):
    """Check if two texts are essentially the same, ignoring whitespace and case"""
    if text1 is None or text2 is None:
        return False
        
    # Normalize both texts
    t1 = re.sub(r'\s+', ' ', text1.lower()).strip()
    t2 = re.sub(r'\s+', ' ', text2.lower()).strip()
    
    # If they're exactly the same after normalization
    if t1 == t2:
        return True
        
    # Calculate similarity ratio
    similarity = difflib.SequenceMatcher(None, t1, t2).ratio()
    return similarity > 0.9  # 90% similar is "essentially the same"

def get_precedent_extracts(input_data, contract_type, jurisdiction):
    """Get relevant precedent extracts from Snowflake Cortex Search Service"""
    try:
        # Initialize the retriever
        logger.info("Initializing SnowflakeCortexRetriever...")
        retriever = SnowflakeCortexRetriever()
        
        # Create a query from the input data
        query_parts = []
        for key, value in input_data.items():
            if isinstance(value, str) and value.strip():
                # Add key terms from input data, limit to keep query concise
                if len(value) < 100:  # Only use short fields to avoid overly complex queries
                    query_parts.append(value)
        
        # Add contract type and jurisdiction to the query parts
        if contract_type:
            query_parts.append(contract_type)
        if jurisdiction:
            query_parts.append(jurisdiction)
        
        # Combine all parts with spaces, limit total length
        query = " ".join(query_parts)
        if len(query) > 500:
            query = query[:500]  # Truncate to reasonable length
        
        logger.info(f"Searching for precedents with query: {query}")
        
        # Get documents using the retriever
        logger.info("Calling get_relevant_documents...")
        docs = retriever.get_relevant_documents(query)
        
        # Log the results
        if not docs:
            logger.info("No relevant precedents found in Snowflake Cortex.")
            return "No relevant precedent extracts found."
            
        logger.info(f"Found {len(docs)} documents from Snowflake Cortex search.")
        
        # Format the precedent extracts
        extracts = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'Unknown Source')
            doc_index = doc.metadata.get('doc_index', -1)
            chunk_index = doc.metadata.get('chunk_index', -1)
            
            # Log each document's metadata
            logger.info(f"Document {i+1}: Source={source}, Doc Index={doc_index}, Chunk Index={chunk_index}")
            
            # Format the precedent with metadata
            extract = f"Precedent {i+1} (Source: {source}, Doc: {doc_index}, Chunk: {chunk_index}):\n{doc.page_content}\n"
            extracts.append(extract)
        
        result = "Precedent Extracts:\n" + "\n".join(extracts)
        logger.info(f"Returning precedent extracts with total length: {len(result)} characters")
        return result
    except Exception as e:
        logger.error(f"Error retrieving precedents: {e}", exc_info=True)
        return "No precedent extracts available due to an error."

def validate_draft(draft_text, contract_type, user_inputs):
    """Validate the generated draft for placeholders and consistency"""
    issues = []
    
    try:
        # Ensure user_inputs is serializable (convert dates to strings)
        serializable_inputs = {}
        for key, value in user_inputs.items():
            if isinstance(value, date):
                serializable_inputs[key] = value.strftime('%Y-%m-%d')
            else:
                serializable_inputs[key] = value
        
        # Placeholder detection
        placeholder_chain = create_placeholder_detection_chain()
        placeholder_result = placeholder_chain.run(contract_text=draft_text)
        
        try:
            placeholder_issues = json.loads(placeholder_result)
            issues.extend(placeholder_issues)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from placeholder detection: {placeholder_result}")
            issues.append({
                "type": "error",
                "field": "Validation",
                "message": "Failed to parse placeholder detection results"
            })
            
        # Consistency validation
        consistency_chain = create_consistency_validation_chain()
        consistency_result = consistency_chain.run(
            contract_type=contract_type,
            user_inputs=json.dumps(serializable_inputs),
            contract_text=draft_text
        )
        
        try:
            consistency_issues = json.loads(consistency_result)
            issues.extend(consistency_issues)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from consistency validation: {consistency_result}")
            issues.append({
                "type": "error",
                "field": "Validation",
                "message": "Failed to parse consistency validation results"
            })
            
        return issues
    except Exception as e:
        logger.error(f"Error during validation: {e}", exc_info=True)
        return [{
            "type": "error",
            "field": "Validation",
            "message": f"Validation failed: {str(e)}"
        }]

def refine_clause_with_issue(issue, draft_text, contract_type, user_inputs):
    """Refine a specific clause based on validation issue"""
    try:
        # Make sure user_inputs is serializable
        serializable_inputs = {}
        for key, value in user_inputs.items():
            if isinstance(value, date):
                serializable_inputs[key] = value.strftime('%Y-%m-%d')
            else:
                serializable_inputs[key] = value
                
        # Extract the clause with the issue
        issue_field = issue.get("field", "")
        issue_text = issue.get("text", "")
        issue_message = issue.get("message", "")
        
        # Find the relevant section in the draft
        section_pattern = re.compile(r"(SECTION|Section|ARTICLE)\s+[\d\.]+\s*[:\.]\s*" + re.escape(issue_field), re.IGNORECASE)
        section_match = section_pattern.search(draft_text)
        
        if not section_match:
            # Try to find by just the field name
            field_pattern = re.compile(r"(\n|^)([^\n]*" + re.escape(issue_field) + "[^\n]*\n)", re.IGNORECASE)
            field_match = field_pattern.search(draft_text)
            
            if not field_match:
                logger.warning(f"Could not locate section for issue: {issue}")
                return None
            
            # Extract a reasonable chunk around the match
            start_pos = max(0, field_match.start() - 200)
            end_pos = min(len(draft_text), field_match.end() + 500)
            original_clause = draft_text[start_pos:end_pos]
        else:
            # Find the end of the section (the next section heading)
            section_start = section_match.start()
            next_section_match = re.search(r"(\n|^)(SECTION|Section|ARTICLE)\s+[\d\.]+\s*[:\.]", draft_text[section_match.end():], re.IGNORECASE)
            
            if next_section_match:
                section_end = section_match.end() + next_section_match.start()
            else:
                section_end = min(section_match.end() + 1000, len(draft_text))
                
            original_clause = draft_text[section_start:section_end]
        
        # Run the refinement chain
        refinement_chain = create_clause_refinement_chain()
        result = refinement_chain.run(
            contract_type=contract_type,
            issue_description=f"{issue_field}: {issue_message}",
            original_clause=original_clause,
            user_inputs=json.dumps(serializable_inputs)
        )
        
        if not result or len(result) < 10:  # Sanity check for reasonable response length
            logger.warning(f"Refinement failed or produced too short result for issue: {issue}")
            return None
            
        return {
            "original": original_clause,
            "refined": result,
            "issue": issue
        }
    except Exception as e:
        logger.error(f"Error refining clause for issue {issue}: {e}", exc_info=True)
        return None

def regenerate_draft_with_fixes(draft_text, fixes, contract_type, jurisdiction, user_inputs, precedent_extracts):
    """Regenerate the draft with specific fixes applied"""
    if not fixes:
        return draft_text
        
    try:
        # Make sure user_inputs is serializable
        serializable_inputs = {}
        for key, value in user_inputs.items():
            if isinstance(value, date):
                serializable_inputs[key] = value.strftime('%Y-%m-%d')
            else:
                serializable_inputs[key] = value
                
        # Apply each fix to the draft text
        updated_text = draft_text
        for fix in fixes:
            if not fix:
                continue
                
            original = fix.get("original")
            refined = fix.get("refined")
            
            if original and refined and original in updated_text:
                updated_text = updated_text.replace(original, refined)
            else:
                logger.warning(f"Could not apply fix: {fix.get('issue')}")
        
        # If we've made substantial changes, regenerate the entire draft
        if len(fixes) > 2:
            # Create a modified prompt highlighting the issues fixed
            issues_desc = ", ".join([f"{fix.get('issue', {}).get('field', 'unknown')}" for fix in fixes if fix])
            
            draft_chain = create_draft_generation_chain()
            # Add special instructions about the issues
            modified_prompt = f"""
            You are regenerating a {contract_type} contract that had the following issues: {issues_desc}.
            Make sure these issues are properly addressed in your new draft.
            """
            
            # Combine user inputs with the modified prompt
            user_inputs_dict = serializable_inputs.copy()
            user_inputs_dict["_special_instructions"] = modified_prompt
            
            # Regenerate
            result = draft_chain.run(
                contract_type=contract_type,
                jurisdiction=jurisdiction,
                user_inputs=json.dumps(user_inputs_dict),
                precedent_extracts=precedent_extracts
            )
            
            if result and len(result) > 100:
                return result
                
        # If regeneration failed or we made only minor changes, return the updated text
        return updated_text
    except Exception as e:
        logger.error(f"Error regenerating draft with fixes: {e}", exc_info=True)
        # Return the original with fixes applied, as a fallback
        return updated_text

def generate_contract(contract_type, jurisdiction, input_data):
    """End-to-end contract generation pipeline"""
    try:
        # 1. Get precedent extracts
        st.info("Retrieving relevant precedents...")
        precedent_extracts = get_precedent_extracts(input_data, contract_type, jurisdiction)
        
        # 2. Create the main draft generation chain
        st.info("Generating initial draft...")
        draft_chain = create_draft_generation_chain()
        
        # Fix for JSON serialization of date objects
        # Create a copy of input_data with dates converted to strings
        serializable_input = {}
        for key, value in input_data.items():
            if isinstance(value, date):
                serializable_input[key] = value.strftime('%Y-%m-%d')
            else:
                serializable_input[key] = value
        
        # 3. Run the draft generation chain
        initial_draft = draft_chain.run(
            contract_type=contract_type,
            jurisdiction=jurisdiction,
            user_inputs=json.dumps(serializable_input),
            precedent_extracts=precedent_extracts
        )
        
        if not initial_draft or len(initial_draft) < 100:
            st.error("Draft generation failed to produce a valid document")
            return None
            
        # 4. Validate the draft
        st.info("Validating draft for consistency and completeness...")
        issues = validate_draft(initial_draft, contract_type, serializable_input)
        
        # 5. If there are issues, attempt to fix them
        if issues:
            st.warning(f"Found {len(issues)} issues to fix in the initial draft")
            
            # Limit to max 3 iterations for fixes
            for iteration in range(3):
                if not issues:
                    break
                    
                st.info(f"Refinement iteration {iteration+1}...")
                
                # Refine clauses with issues
                fixes = []
                for issue in issues:
                    fix = refine_clause_with_issue(issue, initial_draft, contract_type, serializable_input)
                    if fix:
                        fixes.append(fix)
                
                if not fixes:
                    st.warning("Could not automatically fix the identified issues")
                    break
                    
                # Regenerate the draft with fixes
                initial_draft = regenerate_draft_with_fixes(
                    initial_draft, fixes, contract_type, jurisdiction, serializable_input, precedent_extracts
                )
                
                # Validate again
                issues = validate_draft(initial_draft, contract_type, serializable_input)
                
                if not issues:
                    st.success("All issues resolved automatically!")
                    break
                    
            # If we still have issues after iterations, inform the user
            if issues:
                issue_descriptions = "\n".join([f"- {issue.get('field', '')}: {issue.get('message', '')}" for issue in issues])
                st.warning(f"Some issues could not be automatically resolved and may require review:\n{issue_descriptions}")
        
        # 6. Return the final draft
        return initial_draft
    except Exception as e:
        logger.error(f"Error in contract generation pipeline: {e}", exc_info=True)
        st.error(f"An error occurred during contract generation: {str(e)}")
        return None

# --- Streamlit App UI ---
def main():
    st.title(APP_NAME)
    st.caption("Generate legal agreement drafts with AI assistance. Review is essential.")
    
    # Initialize session state
    if 'app_stage' not in st.session_state:
        st.session_state.app_stage = 'input'
    if 'initial_input_data' not in st.session_state:
        st.session_state.initial_input_data = {}
    if 'review_data' not in st.session_state:
        st.session_state.review_data = {}
    if 'final_input_data' not in st.session_state:
        st.session_state.final_input_data = {}
    if 'agreement_type' not in st.session_state:
        st.session_state.agreement_type = list(AGREEMENT_QUESTIONS.keys())[0]
    if 'jurisdiction' not in st.session_state:
        try:
            default_jurisdiction_index = US_STATES.index("Massachusetts")
        except ValueError:
            default_jurisdiction_index = 0
        st.session_state.jurisdiction = US_STATES[default_jurisdiction_index]
    if 'generated_text' not in st.session_state:
        st.session_state.generated_text = None
    
    # Sidebar selectors
    st.sidebar.header("Contract Options")
    
    agreement_type_options = list(AGREEMENT_QUESTIONS.keys())
    try:
        default_agreement_type_index = agreement_type_options.index(st.session_state.agreement_type)
    except ValueError:
        default_agreement_type_index = 0
        
    try:
        default_jurisdiction_index = US_STATES.index(st.session_state.jurisdiction)
    except ValueError:
        try:
            default_jurisdiction_index = US_STATES.index("Massachusetts")
        except ValueError:
            default_jurisdiction_index = 0
            
    st.session_state.agreement_type = st.sidebar.selectbox(
        "Select Agreement Type:",
        options=agreement_type_options,
        index=default_agreement_type_index,
        key="agreement_type_selector",
        disabled=(st.session_state.app_stage != 'input')
    )
    
    st.session_state.jurisdiction = st.sidebar.selectbox(
        "Select Governing Law State:",
        options=US_STATES,
        index=default_jurisdiction_index,
        key="jurisdiction_selector",
        disabled=(st.session_state.app_stage != 'input')
    )
    
    # Stage 1: Input Form
    if st.session_state.app_stage == 'input':
        st.header(f"Generate: {st.session_state.agreement_type}")
        st.subheader(f"Governing Law: {st.session_state.jurisdiction}")
        st.markdown("---")
        
        st.markdown("""
        **Instructions:** Fill in the details below. Click 'Review Inputs' for refinement and review. 
        *Use clear sentences.* Fill required fields. **Legal review is essential.**
        """)
        
        current_input_data = {}
        questions = AGREEMENT_QUESTIONS.get(st.session_state.agreement_type, [])
        
        if not questions:
            st.error("Configuration missing.")
        else:
            col1, col2 = st.columns(2)
            for i, q in enumerate(questions):
                target_col = col1 if i % 2 == 0 else col2
                with target_col:
                    q_widget_key = f"input_{st.session_state.agreement_type}_{q['key']}"
                    label = q["label"]
                    help_text = q.get("help", None)
                    
                    # Use existing initial data if available
                    current_default = st.session_state.initial_input_data.get(q['key'], q.get("default"))
                    
                    if q["type"] == "text":
                        current_input_data[q["key"]] = st.text_input(
                            label, 
                            key=q_widget_key, 
                            value=str(current_default or ""), 
                            help=help_text
                        )
                    elif q["type"] == "text_area":
                        current_input_data[q["key"]] = st.text_area(
                            label, 
                            key=q_widget_key, 
                            value=str(current_default or ""), 
                            height=q.get("height", 100), 
                            help=help_text
                        )
                    elif q["type"] == "date_input":
                        date_val = None
                        if isinstance(current_default, date):
                            date_val = current_default
                        elif isinstance(current_default, str):
                            try:
                                date_val = datetime.strptime(current_default, '%Y-%m-%d').date()
                            except (ValueError, TypeError):
                                date_val = None
                        
                        current_input_data[q["key"]] = st.date_input(
                            label, 
                            key=q_widget_key, 
                            value=date_val, 
                            help=help_text
                        )
            
            st.markdown("---")
            if st.button("Review Inputs", type="primary"):
                validation_errors = []
                
                # Check required fields
                missing_fields = []
                for q in questions:
                    if q.get('required', False):
                        value = current_input_data.get(q['key'])
                        if value is None or (isinstance(value, str) and not value.strip()):
                            missing_fields.append(q['label'])
                
                if missing_fields:
                    validation_errors.append(f"Please fill in the required fields: {', '.join(missing_fields)}")
                
                # Date validation
                start_date_key, end_date_key = None, None
                agreement_type = st.session_state.agreement_type
                q_list = AGREEMENT_QUESTIONS.get(agreement_type, [])
                
                start_q = next((q for q in q_list if q['key'] == 'start_date' and q['type'] == 'date_input'), None)
                end_q = next((q for q in q_list if q['key'] == 'end_date' and q['type'] == 'date_input'), None)
                
                if start_q and end_q:
                    start_date_val = current_input_data.get('start_date')
                    end_date_val = current_input_data.get('end_date')
                    
                    if isinstance(start_date_val, date) and isinstance(end_date_val, date):
                        if end_date_val < start_date_val:
                            validation_errors.append(f"'{end_q['label']}' cannot be before '{start_q['label']}'.")
                
                if validation_errors:
                    for error in validation_errors:
                        st.error(error)
                else:
                    # Store current inputs before refinement
                    st.session_state.initial_input_data = current_input_data.copy()
                    st.session_state.review_data = {}
                    st.session_state.final_input_data = {}
                    
                    # In the review stage, when processing inputs
                    with st.spinner("Analyzing inputs for refinement..."):
                        questions_for_type = AGREEMENT_QUESTIONS.get(st.session_state.agreement_type, [])

                        # Process refinable fields
                        for q in questions_for_type:
                            field_key = q['key']
                            original_value = st.session_state.initial_input_data.get(field_key)
                            
                            # Check if refinement is enabled for this field
                            if q.get("refine", False) and isinstance(original_value, str) and original_value.strip():
                                # Determine if this field should be skipped based on type
                                should_skip = False
                                
                                # Skip address fields that already have city/state/zip
                                if "address" in field_key.lower():
                                    parts = original_value.split(',')
                                    if len(parts) >= 3 and any(p.strip().isdigit() for p in parts):
                                        # Already has city, state, zip format
                                        should_skip = True
                                
                                if not should_skip:
                                    # Pass all input data as context
                                    suggested_value = refine_text_field(
                                        str(original_value), 
                                        q['label'], 
                                        st.session_state.agreement_type,
                                        st.session_state.initial_input_data  # Pass all inputs as context
                                    )
                                    
                                    # Only store for review if actually different
                                    if suggested_value != original_value:
                                        st.session_state.review_data[field_key] = {
                                            'original': str(original_value), 
                                            'suggested': suggested_value
                                        }
                                        st.session_state.final_input_data[field_key] = suggested_value
                                    else:
                                        # If no changes, use original directly
                                        st.session_state.final_input_data[field_key] = original_value
                                else:
                                    # Skip refinement for this field
                                    st.session_state.final_input_data[field_key] = original_value
                            else:
                                # No refinement for this field
                                st.session_state.final_input_data[field_key] = original_value

                    st.session_state.app_stage = 'review'
                    st.rerun()
    
    # Stage 2: Review and Edit
    elif st.session_state.app_stage == 'review':
        st.header(f"Review & Edit: {st.session_state.agreement_type}")
        st.markdown("---")
        
        st.info("Review inputs below. Edit fields marked 'Suggested Refinement'. Text starting '[Review Recommended...]' indicates potential issues (placeholders/unclear). Edit to fix or confirm.")
        
        questions = AGREEMENT_QUESTIONS.get(st.session_state.agreement_type, [])
        if not questions:
            st.error("Configuration error: No questions found for this agreement type.")
        else:
            col1, col2 = st.columns(2)
            current_review_edits = {}
            
            for i, q in enumerate(questions):
                target_col = col1 if i % 2 == 0 else col2
                with target_col:
                    field_key = q['key']
                    label = q['label']
                    
                    # Check if this field had a refinement suggestion
                    is_refined = field_key in st.session_state.review_data
                    
                    # Get the current value for editing
                    value_for_editing = st.session_state.final_input_data.get(field_key, "")
                    
                    if is_refined:
                        st.markdown(f"**{label}** (Suggested Refinement)")
                        st.caption(f"Original: {st.session_state.review_data[field_key]['original']}")
                        
                        review_widget_key = f"review_{st.session_state.agreement_type}_{field_key}"
                        
                        # Check for the review warning prefix
                        if isinstance(value_for_editing, str) and value_for_editing.startswith("[Review Recommended:"):
                            st.warning("Input may need revision (placeholder/unclear). Please edit.")
                            
                        # Use text_area for potentially long refined fields
                        edited_value = st.text_area(
                            "Edit suggestion if needed:", 
                            value=str(value_for_editing), 
                            key=review_widget_key, 
                            height=q.get("height", 100) + 20, 
                            help=q.get("help")
                        )
                        
                        current_review_edits[field_key] = edited_value
                    else:
                        # Display non-refined fields (read-only view)
                        st.markdown(f"**{label}**")
                        value_to_display = value_for_editing
                        display_text = ""
                        
                        if isinstance(value_to_display, date):
                            display_text = value_to_display.strftime('%Y-%m-%d')
                        elif isinstance(value_to_display, list):
                            display_text = ", ".join(map(str, value_to_display))
                        elif isinstance(value_to_display, (int, float)):
                            display_text = str(value_to_display)
                        else:
                            display_text = str(value_to_display or " ")
                            
                        # Use text_input for read-only display with a proper label
                        st.text_input(
                            f"Display {label}", 
                            value=display_text, 
                            key=f"display_{field_key}", 
                            disabled=True, 
                            label_visibility="collapsed"
                        )
                        
                        # Store this value even though it wasn't edited
                        current_review_edits[field_key] = value_to_display
            
            st.markdown("---")
            review_col1, review_col2 = st.columns(2)
            
            with review_col1:
                if st.button("Confirm and Generate Document", type="primary"):
                    # Update the final input data with any edits
                    st.session_state.final_input_data.update(current_review_edits)
                    logger.info("Final input confirmed after review.")
                    
                    # Clear review suggestions as they've been processed
                    st.session_state.review_data = {}
                    
                    st.session_state.app_stage = 'generating'
                    st.rerun()
                    
            with review_col2:
                if st.button("Back to Edit Inputs"):
                    # Update initial_input_data with the reviewed/edited values
                    st.session_state.initial_input_data.update(current_review_edits)
                    
                    # Clear review/final data as we are going back to the start
                    st.session_state.review_data = {}
                    st.session_state.final_input_data = {}
                    
                    st.session_state.app_stage = 'input'
                    st.rerun()
    
    # Stage 3: Generating / Displaying Results
    elif st.session_state.app_stage in ['generating', 'done']:
        final_data_for_prompt = st.session_state.final_input_data
        
        if st.session_state.app_stage == 'generating':
            st.info("Preparing final request...")
            
            agreement_type = st.session_state.agreement_type
            jurisdiction = st.session_state.jurisdiction
            
            with st.spinner("Generating contract draft... This may take a moment."):
                # Call the end-to-end generation pipeline
                generated_text = generate_contract(agreement_type, jurisdiction, final_data_for_prompt)
                
                st.session_state.generated_text = generated_text
                st.session_state.app_stage = 'done'
                st.rerun()
                
        if st.session_state.app_stage == 'done':
            st.header(f"Generated Draft: {st.session_state.agreement_type}")
            st.subheader(f"Governing Law: {st.session_state.jurisdiction}")
            st.markdown("---")
            
            generated_text = st.session_state.get('generated_text', None)
            
            if not generated_text:
                st.error("Document generation failed. No text was produced.")
            elif isinstance(generated_text, str):
                # Generation succeeded, proceed to PDF creation
                st.success("Generation complete. Preparing PDF...")
                
                pdf_title = f"{st.session_state.agreement_type} - Draft"
                try:
                    pdf_buffer = create_pdf_from_generated_text(
                        generated_text, 
                        pdf_title, 
                        st.session_state.final_input_data
                    )
                    
                    if pdf_buffer:
                        # Create a readable filename
                        party_name_raw = st.session_state.final_input_data.get(
                            'client_name',
                            st.session_state.final_input_data.get(
                                'employer_name',
                                st.session_state.final_input_data.get(
                                    'landlord_name',
                                    'Generated'
                                )
                            )
                        )
                        
                        party_name_clean = re.sub(r'[^\w\-]+', '_', str(party_name_raw))[:20]
                        jurisdiction_clean = st.session_state.jurisdiction.replace(' ', '')
                        agreement_clean = st.session_state.agreement_type.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')
                        timestamp = datetime.now().strftime('%Y%m%d')
                        
                        file_name = f"{agreement_clean}_Draft_{party_name_clean}_{jurisdiction_clean}_{timestamp}.pdf"
                        
                        st.success("PDF Generated Successfully!")
                        st.download_button(
                            label="Download PDF Draft",
                            data=pdf_buffer,
                            file_name=file_name,
                            mime="application/pdf"
                        )
                        
                        st.subheader("Generated Text (for review):")
                        st.text_area(
                            "Review the text used for the PDF:",
                            generated_text,
                            height=300,
                            key="generated_text_review",
                            help="This is the raw text generated by the AI before PDF formatting."
                        )
                    else:
                        st.error("PDF generation failed after text was created. See logs for details.")
                        st.subheader("Raw Output (PDF Generation Failed):")
                        st.text_area("You can copy the text below:", generated_text, height=400)
                except Exception as pdf_e:
                    st.error(f"An unexpected error occurred during PDF creation: {pdf_e}")
                    logger.error(f"PDF creation raised exception: {pdf_e}", exc_info=True)
                    st.subheader("Raw Output (PDF Creation Error):")
                    st.text_area("You can copy the text below:", generated_text, height=400)
            else:
                # Handle cases where generate_contract returned unexpected type
                st.warning("Generation produced unexpected output format.")
                st.text_area("Raw Output:", str(generated_text), height=300)
                
            # Button to reset the process
            if st.button("Start New Document"):
                # Clear relevant session state variables
                st.session_state.app_stage = 'input'
                st.session_state.initial_input_data = {}
                st.session_state.review_data = {}
                st.session_state.final_input_data = {}
                st.session_state.generated_text = None
                st.rerun()
    
    # Footer
    st.sidebar.divider()
    st.sidebar.markdown("""
    **QwikContractAI**
    
    Generate contract drafts with AI assistance:
    - Service Agreements
    - Employment Agreements
    - Residential Leases
    
    *All drafts require legal review.*
    """)

if __name__ == "__main__":
    main()
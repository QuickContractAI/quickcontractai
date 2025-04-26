import streamlit as st
import json
import re
from typing import List, Dict, Any, Optional, Tuple
import logging

# Import from your existing modules
from langchain.chat_models import ChatOllama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os

logger = logging.getLogger(__name__)

# --- Contract Analysis Models ---
def get_analysis_model():
    """Get the Gemini model for contract analysis"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Google API key not found in environment variables.")
        return None
        
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-pro-exp-03-25",
        google_api_key=api_key,
        temperature=0.2,
    )

def get_summary_model():
    """Get the Gemini model for contract summarization"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Google API key not found in environment variables.")
        return None
        
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-pro-exp-03-25",  # Using the same model as analysis
        google_api_key=api_key,
        temperature=0.3,  # Slightly higher temperature for more creative summaries
    )

# --- Contract Analysis Chains ---
def create_clause_extraction_chain():
    """Create a chain for extracting key clauses and terms from contracts"""
    llm = get_analysis_model()
    
    template = """
    You are a legal analyst specialized in contract review. Extract the key clauses, terms, and obligations from the following contract.
    
    Contract Type: {contract_type}
    
    Contract Text:
    {contract_text}
    
    Extract and organize the following information:
    1. Parties: Who are the parties involved in this contract? Be very specific and look for FULL NAMES, not just roles. 
       Check both the main contract text AND the signature block for names of individuals or companies.
    2. Key Dates: Extract all important dates (effective date, termination date, renewal dates, etc.)
    3. Payment Terms: Extract all payment-related information (amounts, schedules, penalties)
    4. Obligations: List the primary obligations of each party
    5. Termination Clauses: Extract conditions for termination
    6. Liability Clauses: Extract any limitation of liability or indemnification provisions
    7. Confidentiality: Extract confidentiality or non-disclosure provisions
    8. Dispute Resolution: Extract dispute resolution mechanisms
    9. Governing Law: Extract the governing law jurisdiction
    10. Other Critical Terms: Any other critical terms specific to this contract type
    
    For each clause or term, include:
    - A brief description of the term
    - The relevant section number/title
    - The actual text of the clause (condensed if very long)
    
    Return the extracted information as a JSON object. Use this exact format:
    {{
      "parties": [
        {{"name": "Party name", "role": "Role in contract", "description": "Brief description"}}
      ],
      "key_dates": [
        {{"title": "Date description", "date": "The actual date", "section": "Section reference"}}
      ],
      "payment_terms": [
        {{"description": "Term description", "details": "Specific details", "section": "Section reference"}}
      ],
      "obligations": [
        {{"party": "Obligated party", "obligation": "Description of obligation", "section": "Section reference"}}
      ],
      "termination_clauses": [
        {{"description": "Description of clause", "text": "Actual text (condensed)", "section": "Section reference"}}
      ],
      "liability_clauses": [
        {{"description": "Description of clause", "text": "Actual text (condensed)", "section": "Section reference"}}
      ],
      "confidentiality": [
        {{"description": "Description of clause", "text": "Actual text (condensed)", "section": "Section reference"}}
      ],
      "dispute_resolution": [
        {{"description": "Description of mechanism", "text": "Actual text (condensed)", "section": "Section reference"}}
      ],
      "governing_law": {{"jurisdiction": "Governing jurisdiction", "section": "Section reference"}},
      "other_critical_terms": [
        {{"title": "Term title", "description": "Description", "text": "Actual text (condensed)", "section": "Section reference"}}
      ]
    }}
    
    Only include sections if they exist in the contract. If a category is empty, include it as an empty array [].
    Ensure your JSON is properly formatted and parseable.
    """
    
    prompt = PromptTemplate(
        input_variables=["contract_type", "contract_text"],
        template=template,
    )
    
    return LLMChain(llm=llm, prompt=prompt, output_key="extracted_clauses")

def create_risk_flagging_chain():
    """Create a chain for identifying and flagging risky or uncommon clauses"""
    llm = get_analysis_model()
    
    template = """
    You are a risk analysis expert specializing in contract review. Analyze the following contract to identify any clauses that:
    1. Are uncommon or unusual for this type of contract
    2. May present significant risks to one of the parties
    3. Contain vague or ambiguous language
    4. Are heavily one-sided or potentially unenforceable
    5. Contain hidden obligations or liabilities
    6. May conflict with other parts of the contract
    7. May have regulatory compliance issues
    
    Contract Type: {contract_type}
    Jurisdiction: {jurisdiction}
    
    Contract Text:
    {contract_text}
    
    For each identified risk, provide:
    - Risk Level: High, Medium, or Low
    - Category: Type of risk (e.g., Financial, Liability, Compliance, Operational)
    - Section Reference: The section number or title
    - Description: A clear explanation of the risk
    - Recommendation: Specific suggestion on how to address or mitigate the risk
    
    Return the identified risks as a JSON array using this exact format:
    [
      {{
        "risk_level": "High/Medium/Low",
        "category": "Risk category",
        "section": "Section reference",
        "description": "Clear explanation of the risk",
        "recommendation": "Suggestion to address the risk"
      }}
    ]
    
    If no significant risks are identified, return an empty array.
    Ensure your JSON is properly formatted and parseable.
    """
    
    prompt = PromptTemplate(
        input_variables=["contract_type", "jurisdiction", "contract_text"],
        template=template,
    )
    
    return LLMChain(llm=llm, prompt=prompt, output_key="risk_flags")

def create_summarization_chain():
    """Create a chain for generating plain-language summaries of contracts"""
    llm = get_summary_model()
    
    template = """
    You are a contract simplification specialist. Create a plain-language summary of the following contract that would be understandable to someone without legal training.
    
    Contract Type: {contract_type}
    
    Contract Text:
    {contract_text}
    
    Extracted Key Clauses:
    {extracted_clauses}
    
    Provide a summary with the following sections:
    1. Overview: A 2-3 sentence general description of what this contract is for
    2. Key Terms: The most important terms in simple language (bullet points)
    3. Party Obligations: What each party must do (bullet points)
    4. Important Dates: When things happen (bullet points)
    5. Important Considerations: Things to be aware of (bullet points)
    
    Use simple, direct language. Avoid legalese and jargon. When technical terms must be used, explain them.
    The entire summary should be concise, no more than 500 words.
    
    Format the summary with appropriate Markdown formatting.
    """
    
    prompt = PromptTemplate(
        input_variables=["contract_type", "contract_text", "extracted_clauses"],
        template=template,
    )
    
    return LLMChain(llm=llm, prompt=prompt, output_key="plain_language_summary")

# --- Contract Analysis Functions ---
def analyze_contract(contract_text, contract_type, jurisdiction):
    """
    Performs complete contract analysis
    
    Returns:
        tuple: (extracted_clauses, risk_flags, plain_language_summary)
    """
    try:
        # 1. Extract clauses
        st.info("Extracting key clauses and terms...")
        extraction_chain = create_clause_extraction_chain()
        extraction_result = extraction_chain.run(
            contract_type=contract_type,
            contract_text=contract_text
        )
        
        # Parse the JSON result
        try:
            extracted_clauses = json.loads(extraction_result)
            logger.info("Successfully extracted clauses")
        except json.JSONDecodeError:
            logger.error(f"Failed to parse extracted clauses JSON: {extraction_result[:100]}...")
            # Try to extract JSON from text if direct parsing fails
            json_pattern = r"```(?:json)?\s*([\[\{].*?[\]\}])\s*```"
            json_match = re.search(json_pattern, extraction_result, re.DOTALL)
            
            if json_match:
                try:
                    extracted_clauses = json.loads(json_match.group(1).strip())
                    logger.info("Successfully extracted clauses from markdown block")
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON from markdown block")
                    extracted_clauses = {"error": "Failed to parse clause extraction results"}
            else:
                extracted_clauses = {"error": "Failed to parse clause extraction results"}
        
        # 2. Flag risks
        st.info("Analyzing for potential risks...")
        risk_chain = create_risk_flagging_chain()
        risk_result = risk_chain.run(
            contract_type=contract_type,
            jurisdiction=jurisdiction,
            contract_text=contract_text
        )
        
        # Parse the JSON result
        try:
            risk_flags = json.loads(risk_result)
            logger.info(f"Identified {len(risk_flags)} potential risks")
        except json.JSONDecodeError:
            logger.error(f"Failed to parse risk flags JSON: {risk_result[:100]}...")
            # Try to extract JSON from text
            json_pattern = r"```(?:json)?\s*([\[\{].*?[\]\}])\s*```"
            json_match = re.search(json_pattern, risk_result, re.DOTALL)
            
            if json_match:
                try:
                    risk_flags = json.loads(json_match.group(1).strip())
                    logger.info(f"Successfully extracted {len(risk_flags)} risks from markdown block")
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON from markdown block")
                    risk_flags = []
            else:
                risk_flags = []
        
        # 3. Generate plain language summary
        st.info("Generating plain language summary...")
        
        # Convert extracted_clauses to string for the summarization prompt
        extracted_clauses_str = json.dumps(extracted_clauses, indent=2)
        
        summary_chain = create_summarization_chain()
        summary_result = summary_chain.run(
            contract_type=contract_type,
            contract_text=contract_text,
            extracted_clauses=extracted_clauses_str
        )
        
        logger.info("Successfully generated plain language summary")
        
        return extracted_clauses, risk_flags, summary_result
    
    except Exception as e:
        logger.error(f"Error in contract analysis pipeline: {e}", exc_info=True)
        st.error(f"An error occurred during contract analysis: {str(e)}")
        return (
            {"error": f"Clause extraction failed: {str(e)}"},
            [],
            f"Summary generation failed: {str(e)}"
        )

# --- UI Components for Analysis ---
def render_contract_analysis_tab(contract_text, contract_type, jurisdiction):
    """Render the contract analysis tab in the UI"""
    
    if not contract_text or len(contract_text) < 100:
        st.warning("Please generate a contract first before analyzing it.")
        return
        
    # Create three tabs for the different analysis features
    analysis_tab, risks_tab, summary_tab = st.tabs(["Clause Extraction", "Risk Analysis", "Plain Language Summary"])
    
    # Check if analysis has already been done
    if "analysis_results" not in st.session_state:
        with st.spinner("Analyzing contract. This may take a moment..."):
            extracted_clauses, risk_flags, summary = analyze_contract(
                contract_text=contract_text,
                contract_type=contract_type,
                jurisdiction=jurisdiction
            )
            # Store the results in session state
            st.session_state.analysis_results = {
                "extracted_clauses": extracted_clauses,
                "risk_flags": risk_flags,
                "summary": summary
            }
    
    # Get results from session state
    extracted_clauses = st.session_state.analysis_results["extracted_clauses"]
    risk_flags = st.session_state.analysis_results["risk_flags"]
    summary = st.session_state.analysis_results["summary"]
    
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
            
            # Payment Terms
            if "payment_terms" in extracted_clauses and extracted_clauses["payment_terms"]:
                st.subheader("Payment Terms")
                for term in extracted_clauses["payment_terms"]:
                    with st.expander(f"{term.get('description', 'Payment Term')} (Section: {term.get('section', 'N/A')})"):
                        st.write(term.get("details", "No details provided"))
            
            # Obligations
            if "obligations" in extracted_clauses and extracted_clauses["obligations"]:
                st.subheader("Party Obligations")
                obligations_by_party = {}
                for obligation in extracted_clauses["obligations"]:
                    party = obligation.get("party", "Unspecified Party")
                    if party not in obligations_by_party:
                        obligations_by_party[party] = []
                    obligations_by_party[party].append({
                        "obligation": obligation.get("obligation", "Not specified"),
                        "section": obligation.get("section", "Not specified")
                    })
                
                for party, obligations in obligations_by_party.items():
                    with st.expander(f"Obligations of {party}"):
                        for obligation in obligations:
                            st.markdown(f"- **{obligation['obligation']}** (Section: {obligation['section']})")
            
            # Termination Clauses
            if "termination_clauses" in extracted_clauses and extracted_clauses["termination_clauses"]:
                st.subheader("Termination Clauses")
                for clause in extracted_clauses["termination_clauses"]:
                    with st.expander(f"{clause.get('description', 'Termination Clause')} (Section: {clause.get('section', 'N/A')})"):
                        st.write(clause.get("text", "No text provided"))
            
            # Liability Clauses
            if "liability_clauses" in extracted_clauses and extracted_clauses["liability_clauses"]:
                st.subheader("Liability Clauses")
                for clause in extracted_clauses["liability_clauses"]:
                    with st.expander(f"{clause.get('description', 'Liability Clause')} (Section: {clause.get('section', 'N/A')})"):
                        st.write(clause.get("text", "No text provided"))
            
            # Confidentiality
            if "confidentiality" in extracted_clauses and extracted_clauses["confidentiality"]:
                st.subheader("Confidentiality")
                for clause in extracted_clauses["confidentiality"]:
                    with st.expander(f"{clause.get('description', 'Confidentiality Clause')} (Section: {clause.get('section', 'N/A')})"):
                        st.write(clause.get("text", "No text provided"))
            
            # Dispute Resolution
            if "dispute_resolution" in extracted_clauses and extracted_clauses["dispute_resolution"]:
                st.subheader("Dispute Resolution")
                for mechanism in extracted_clauses["dispute_resolution"]:
                    with st.expander(f"{mechanism.get('description', 'Dispute Resolution')} (Section: {mechanism.get('section', 'N/A')})"):
                        st.write(mechanism.get("text", "No text provided"))
            
            # Governing Law
            if "governing_law" in extracted_clauses and extracted_clauses["governing_law"]:
                st.subheader("Governing Law")
                st.write(f"**Jurisdiction:** {extracted_clauses['governing_law'].get('jurisdiction', 'Not specified')}")
                st.write(f"**Section:** {extracted_clauses['governing_law'].get('section', 'Not specified')}")
            
            # Other Critical Terms
            if "other_critical_terms" in extracted_clauses and extracted_clauses["other_critical_terms"]:
                st.subheader("Other Critical Terms")
                for term in extracted_clauses["other_critical_terms"]:
                    with st.expander(f"{term.get('title', 'Term')} (Section: {term.get('section', 'N/A')})"):
                        st.write(f"**Description:** {term.get('description', 'No description provided')}")
                        st.write(f"**Text:** {term.get('text', 'No text provided')}")
    
    # Risk Analysis Tab
    with risks_tab:
        st.header("Risk Assessment")
        
        if isinstance(risk_flags, list):
            if not risk_flags:
                st.success("No significant risks identified in this contract.")
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
                file_name=f"{contract_type}_Summary.md",
                mime="text/markdown"
            )
        else:
            st.error("Error generating plain language summary.")
    
    # Button to reset analysis
    if st.button("Refresh Analysis"):
        # Clear the analysis results from session state
        if "analysis_results" in st.session_state:
            del st.session_state.analysis_results
        st.rerun()
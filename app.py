
import streamlit as st
import os
from dotenv import load_dotenv
from io import BytesIO
import threading
import time
from functools import lru_cache

# Import modules only when needed
@st.cache_resource
def load_modules():
    from modules.file_parser import extract_text
    from modules.eligibility_analyzer import run_eligibility_check
    from modules.checklist_generator import generate_submission_checklist
    from modules.risk_analyzer import analyze_contract_risks
    from modules.report_generator import generate_summary_pdf
    from modules.utils import ask_query_from_context
    from modules.llm import call_llm  # Add this import
    from modules.cache_utils import process_document_in_chunks  # Add this import
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    import textwrap
    
    return {
        "extract_text": extract_text,
        "run_eligibility_check": run_eligibility_check,
        "generate_submission_checklist": generate_submission_checklist,
        "analyze_contract_risks": analyze_contract_risks,
        "generate_summary_pdf": generate_summary_pdf, 
        "ask_query_from_context": ask_query_from_context,
        "call_llm": call_llm,  # Add this
        "process_document_in_chunks": process_document_in_chunks,  # Add this
        "letter": letter,
        "canvas": canvas,
        "textwrap": textwrap
    }

# Load environment variables
load_dotenv()

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.processing = False
    st.session_state.chat_history = []
    st.session_state.loaded_files = False
    st.session_state.current_rfp_file = None
    st.session_state.current_profile_file = None

# Streamlit config
st.set_page_config(page_title="RFP Analyzer", layout="wide")
st.markdown("<h1 style='text-align: center;'>📄 RFP Analyzer – AI-Powered Eligibility Checker</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: right;'>by @team-6", unsafe_allow_html=True)
st.markdown("---")

# Custom CSS for scrollable containers
st.markdown("""
      <style>
        .scrollable-box {
            border: 1px solid #ddd;
            padding: 10px;
            border-radius: 5px;
            max-height: 500px;
            overflow-y: auto;
            background-color: rgba(255, 255, 255, 0.05);
            color: rgba(250, 250, 250, 0.95);  /* Light text for dark theme */
        }
        /* Add media query for light theme if needed */
        @media (prefers-color-scheme: light) {
            .scrollable-box {
                background-color: white;
                color: rgba(0, 0, 0, 0.87);  /* Dark text for light theme */
            }
        }
        </style>
""", unsafe_allow_html=True)

# Cache file processing
@st.cache_data
def process_uploaded_file(uploaded_file):
    modules = load_modules()
    file_path = save_uploaded_file(uploaded_file, f"temp_{uploaded_file.name}")
    text = modules["extract_text"](file_path)
    return text

# Save utility
def save_uploaded_file(uploaded_file, filename):
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# PDF generator with wrapping (optimized)
@st.cache_data
def generate_pdf(content, filename="result.pdf"):
    modules = load_modules()
    buffer = BytesIO()
    pdf = modules["canvas"].Canvas(buffer, pagesize=modules["letter"])
    pdf.setFont("Helvetica", 12)
    text_object = pdf.beginText(40, 750)
    for line in content.split("\n"):
        for sub_line in modules["textwrap"].wrap(line, width=90):
            text_object.textLine(sub_line)
    pdf.drawText(text_object)
    pdf.save()
    buffer.seek(0)
    return buffer

# Function to reset state when files change
def reset_state_for_new_files():
    # Keep these keys
    keep_keys = ["initialized", "processing", "current_rfp_file", "current_profile_file"]
    
    # Clear specific analysis results
    analysis_keys = [
        "eligibility_report", "eligibility_running",
        "submission_checklist", "checklist_running",
        "contract_risks", "risks_running",
        "summary_text", "summary_pdf", "summary_running",
        "query_running", "loaded_files", "rfp_text", "profile_text"
    ]
    
    # Remove analysis results from session state
    for key in analysis_keys:
        if key in st.session_state:
            del st.session_state[key]
    
    # Reset chat history
    st.session_state.chat_history = []
    st.session_state.loaded_files = False

# Layout split: Left (60%) - Reports, Right (40%) - Chatbot
left_col, right_col = st.columns([3, 2])

# Upload section in sidebar for more space
with st.sidebar:
    st.header("Upload Documents")
    
    # Get references to currently uploaded files
    uploaded_rfp = st.file_uploader("Upload RFP Document", type=['pdf', 'docx'], key="rfp_uploader")
    uploaded_profile = st.file_uploader("Company Profile", type=['pdf', 'docx'], key="profile_uploader")
    
    # Check if files have changed
    files_changed = False
    if uploaded_rfp is not None and uploaded_rfp != st.session_state.get("current_rfp_file"):
        st.session_state.current_rfp_file = uploaded_rfp
        files_changed = True
    
    if uploaded_profile is not None and uploaded_profile != st.session_state.get("current_profile_file"):
        st.session_state.current_profile_file = uploaded_profile
        files_changed = True
    
    # If files changed, reset everything
    if files_changed:
        reset_state_for_new_files()
    
    model_choice = st.selectbox("LLM Backend", ["groq", "ollama"])
    os.environ["LLM_BACKEND"] = model_choice
    
    # Process files if both are uploaded and not yet processed
    if uploaded_rfp and uploaded_profile and not st.session_state.loaded_files:
        with st.spinner("Processing uploaded files..."):
            st.session_state.rfp_text = process_uploaded_file(uploaded_rfp)
            st.session_state.profile_text = process_uploaded_file(uploaded_profile)
            st.session_state.loaded_files = True
            st.success("Files processed!")

# Main content area
if st.session_state.get("loaded_files", False):
    with left_col:
        # Preview sections
        with st.expander("🔍 Document Previews", expanded=False):
            preview_tabs = st.tabs(["RFP Preview", "Company Profile"])
            with preview_tabs[0]:
                st.markdown(f"<div class='scrollable-box'>{st.session_state.rfp_text[:3000]}</div>", unsafe_allow_html=True)
            with preview_tabs[1]:
                st.markdown(f"<div class='scrollable-box'>{st.session_state.profile_text[:3000]}</div>", unsafe_allow_html=True)

        # Analysis tabs for better organization
        analysis_tabs = st.tabs(["Eligibility", "Submission Checklist", "Risk Analysis", "Summary Report"])
        
        # Eligibility Tab
        with analysis_tabs[0]:
            st.markdown("### ✅ Eligibility Analysis")
            if st.button("🔍 Run Eligibility Check", key="run_eligibility"):
                if not st.session_state.get("eligibility_running", False):
                    st.session_state.eligibility_running = True
                    with st.spinner("Analyzing eligibility..."):
                        try:
                            modules = load_modules()
                            result = modules["run_eligibility_check"](
                                st.session_state.rfp_text,
                                st.session_state.profile_text
                            )
                            st.session_state["eligibility_report"] = result
                            st.session_state.eligibility_running = False
                        except Exception as e:
                            st.error(f"Eligibility check failed: {e}")
                            st.session_state.eligibility_running = False
            
            if "eligibility_report" in st.session_state:
                st.markdown(f"<div class='scrollable-box'>{st.session_state['eligibility_report']}</div>", unsafe_allow_html=True)
                st.download_button(
                    "📄 Download Eligibility Report (PDF)", 
                    generate_pdf(st.session_state["eligibility_report"]), 
                    "eligibility_report.pdf",
                    key="download_eligibility"
                )
        
        # Submission Checklist Tab
        with analysis_tabs[1]:
            st.markdown("### 📋 Submission Checklist")
            if st.button("📋 Generate Submission Checklist", key="run_checklist"):
                if not st.session_state.get("checklist_running", False):
                    st.session_state.checklist_running = True
                    with st.spinner("Extracting submission requirements..."):
                        try:
                            modules = load_modules()
                            checklist = modules["generate_submission_checklist"](st.session_state.rfp_text)
                            st.session_state["submission_checklist"] = checklist
                            st.session_state.checklist_running = False
                        except Exception as e:
                            st.error(f"Checklist generation failed: {e}")
                            st.session_state.checklist_running = False
            
            if "submission_checklist" in st.session_state:
                st.markdown(f"<div class='scrollable-box'>{st.session_state['submission_checklist']}</div>", unsafe_allow_html=True)
                st.download_button(
                    "📄 Download Submission Checklist (PDF)", 
                    generate_pdf(st.session_state["submission_checklist"]), 
                    "submission_checklist.pdf",
                    key="download_checklist"
                )
        
        # Risk Analysis Tab
        with analysis_tabs[2]:
            st.markdown("### ⚠ Contract Risk Analysis")
            if st.button("⚠ Analyze Contract Risks", key="run_risks"):
                if not st.session_state.get("risks_running", False):
                    st.session_state.risks_running = True
                    with st.spinner("Analyzing contract risks..."):
                        try:
                            modules = load_modules()
                            risks = modules["analyze_contract_risks"](st.session_state.rfp_text)
                            st.session_state["contract_risks"] = risks
                            st.session_state.risks_running = False
                        except Exception as e:
                            st.error(f"Risk analysis failed: {e}")
                            st.session_state.risks_running = False
            
            if "contract_risks" in st.session_state:
                st.markdown(f"<div class='scrollable-box'>{st.session_state['contract_risks']}</div>", unsafe_allow_html=True)
                st.download_button(
                    "📄 Download Contract Risk Report (PDF)", 
                    generate_pdf(st.session_state["contract_risks"]), 
                    "contract_risk_analysis.pdf",
                    key="download_risks"
                )
        
        # Summary Report Tab
        with analysis_tabs[3]:
            st.markdown("### 🧾 Executive Summary Report")
            if st.button("🧾 Generate Summary Report", key="run_summary"):
                if all(k in st.session_state for k in ["eligibility_report", "submission_checklist", "contract_risks"]):
                    if not st.session_state.get("summary_running", False):
                        st.session_state.summary_running = True
                        with st.spinner("Generating comprehensive summary report..."):
                            try:
                                modules = load_modules()
                                summary_text, summary_pdf = modules["generate_summary_pdf"](
                                    st.session_state["eligibility_report"],
                                    st.session_state["submission_checklist"],
                                    st.session_state["contract_risks"],
                                    st.session_state.rfp_text,
                                    st.session_state.profile_text
                                )
                                st.session_state["summary_text"] = summary_text
                                st.session_state["summary_pdf"] = summary_pdf
                                st.session_state.summary_running = False
                            except Exception as e:
                                st.error(f"Summary generation failed: {e}")
                                st.session_state.summary_running = False
                else:
                    st.warning("Please generate all three individual reports first.")
            
            if "summary_text" in st.session_state:
                st.markdown(f"<div class='scrollable-box'>{st.session_state['summary_text']}</div>", unsafe_allow_html=True)
                st.download_button(
                    "📄 Download Summary Report (PDF)", 
                    st.session_state["summary_pdf"], 
                    "summary_report.pdf",
                    key="download_summary"
                )
    
    # Right column for chat
    with right_col:
        st.markdown("## 💬 Ask About the RFP")
        
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        user_query = st.chat_input("Ask your query here...")
        
        if user_query and not st.session_state.get("query_running", False):
            st.session_state.query_running = True
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            with st.chat_message("user"):
                st.markdown(user_query)
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        modules = load_modules()
                        reply = modules["ask_query_from_context"](
                            user_query,
                            st.session_state.rfp_text,
                            st.session_state.profile_text,
                            st.session_state.get("eligibility_report", ""),
                            st.session_state.get("submission_checklist", ""),
                            st.session_state.get("contract_risks", "")
                        )
                        st.markdown(reply)
                        st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    except Exception as e:
                        error_msg = f"Error processing query: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                    
                    st.session_state.query_running = False
    
    # Reset button in sidebar
    with st.sidebar:
        if st.button("🔄 Reset All", key="reset_all"):
            for key in list(st.session_state.keys()):
                if key != "initialized":
                    del st.session_state[key]
            st.session_state.chat_history = []
            st.session_state.loaded_files = False
            st.experimental_rerun()

else:
    st.info("Please upload both the RFP and Company Profile documents (PDF or DOCX) in the sidebar.")
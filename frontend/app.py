# frontend/app.py - Streamlit Legal Chatbot UI
import streamlit as st
import requests
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import time

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from frontend.auth_ui import AuthUI
from frontend.components.protected_route import protected_route, require_role

# Streamlit app configuration
st.set_page_config(
    page_title="Legal Chatbot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Production SaaS styling: clean spacing, typography, chat bubbles
st.markdown("""
<style>
    /* Main content spacing */
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 900px; }
    /* Section headers — 1080p readable */
    h1 { font-size: 2rem !important; font-weight: 600 !important; color: #1e293b !important; margin-bottom: 0.5rem !important; }
    h2 { font-size: 1.25rem !important; font-weight: 600 !important; color: #334155 !important; margin-top: 1.5rem !important; }
    h3 { font-size: 1rem !important; font-weight: 600 !important; color: #475569 !important; }
    /* Sidebar: dark slate palette, aligned with main theme */
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1e293b 0%, #334155 100%); }
    [data-testid="stSidebar"] > div { max-height: 100vh; overflow-y: auto; padding: 0 1rem 1rem 1rem; }
    [data-testid="stSidebar"] .stMarkdown { font-size: 0.9rem; color: #e2e8f0; }
    [data-testid="stSidebar"] h1 { color: #f8fafc !important; font-weight: 600 !important; margin-bottom: 0.25rem !important; }
    [data-testid="stSidebar"] hr { border-color: #475569 !important; margin: 1.25rem 0 !important; }
    /* Sidebar success/error/info on dark bg */
    [data-testid="stSidebar"] [data-testid="stAlert"] { border-radius: 6px; }
    [data-testid="stSidebar"] .stCaptionContainer { color: #94a3b8 !important; }
    [data-testid="stSidebar"] .stMarkdown strong { color: #f1f5f9 !important; }
    /* Section dividers — subtle, not harsh */
    hr { margin: 1.5rem 0 !important; border-color: #e2e8f0 !important; opacity: 0.8; }
    /* Chat message bubbles: spacing, borders, role distinction */
    [data-testid="stChatMessage"] {
        padding: 1rem 1.25rem !important;
        margin-bottom: 1rem !important;
        border-radius: 10px !important;
        border: 1px solid #e2e8f0 !important;
    }
    /* User messages: light blue tint + slate accent */
    [data-testid="stChatMessage"] { background: #f8fafc !important; border-left: 4px solid #94a3b8 !important; }
    /* Assistant messages: warm neutral + stone accent (alternate with user) */
    [data-testid="stChatMessage"]:nth-of-type(even) { background: #fafaf9 !important; border-left-color: #78716c !important; }
    /* Citation section spacing */
    .citations-spacer { margin-top: 1rem; display: block; }
    /* Citation/expander blocks: accent line, warm background */
    [data-testid="stChatMessage"] [data-testid="stExpander"] {
        border: 1px solid #e7e5e4 !important;
        border-left: 3px solid #78716c !important;
        border-radius: 6px !important;
        margin: 0.5rem 0 !important;
        background: #fafaf9 !important;
    }
    /* Citation text snippet: monospace, smaller */
    [data-testid="stChatMessage"] [data-testid="stExpander"] pre,
    [data-testid="stChatMessage"] [data-testid="stExpander"] code { font-size: 0.8rem !important; font-family: ui-monospace, monospace !important; }
    /* Send button / primary action */
    [data-testid="stChatInput"] button,
    .stChatInputContainer button {
        background: #1e293b !important;
        color: #fff !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    [data-testid="stChatInput"] button:hover,
    .stChatInputContainer button:hover {
        background: #334155 !important;
    }
    /* Subtle success/error styling */
    .stSuccess { background-color: #f0fdf4 !important; border-radius: 8px; }
    .stError { background-color: #fef2f2 !important; border-radius: 8px; }
    .stInfo { background-color: #f0f9ff !important; border-radius: 8px; }
    .stWarning { background-color: #fffbeb !important; border-radius: 8px; }
    /* Caption text */
    .stCaptionContainer { color: #64748b; font-size: 0.85rem; }
    /* Sidebar: role badge, user info, mode selection — dark theme */
    .sidebar-role-badge { display: inline-block; padding: 0.25rem 0.6rem; border-radius: 6px; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 0.5rem; }
    .sidebar-role-public { background: rgba(134, 239, 172, 0.2); color: #86efac; border: 1px solid rgba(134, 239, 172, 0.3); }
    .sidebar-role-solicitor { background: rgba(253, 224, 71, 0.2); color: #fde047; border: 1px solid rgba(253, 224, 71, 0.3); }
    .sidebar-role-admin { background: rgba(252, 165, 165, 0.2); color: #fca5a5; border: 1px solid rgba(252, 165, 165, 0.3); }
    .sidebar-user-name { font-weight: 600; color: #f8fafc; font-size: 1rem; letter-spacing: -0.01em; line-height: 1.4; display: block; margin-bottom: 0.25rem; }
    .sidebar-user-email { color: #94a3b8; font-size: 0.85rem; line-height: 1.4; display: block; margin-top: 0.25rem; }
    .sidebar-mode-label { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.35rem; }
    /* Profile section spacing */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div { margin-bottom: 0.5rem; }
    /* Citation badges */
    .cite-badge { display: inline-block; padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; background: #1e293b; color: #fff; margin-right: 0.5rem; }
    .cite-badge-green { background: #059669; color: #fff; }
    .source-title { font-weight: 600; color: #1e293b; font-size: 0.95rem; }
    .source-meta { color: #64748b; font-size: 0.85rem; margin-top: 0.25rem; }
    .source-snippet-label { font-size: 0.8rem; color: #64748b; margin-top: 0.5rem; }
    /* Chat hero: demo-ready, high contrast, centered */
    .chat-hero { margin-bottom: 1.75rem; padding-bottom: 1.25rem; text-align: center; }
    .chat-hero-title { font-size: 2.5rem !important; font-weight: 700 !important; color: #0f172a !important; letter-spacing: -0.03em !important; line-height: 1.15 !important; margin-bottom: 0.5rem !important; }
    .chat-hero-subtitle { font-size: 1.1rem !important; color: #334155 !important; line-height: 1.5 !important; max-width: 42em; margin: 0 auto 1rem auto !important; font-weight: 500 !important; }
    .chat-hero-features { font-size: 0.9rem !important; color: #475569 !important; line-height: 1.7 !important; margin: 0 auto !important; max-width: 32em; text-align: left; display: inline-block; }
    .chat-hero-features li { margin-bottom: 0.25rem; }
    /* Typography: consistent hierarchy, WCAG-friendly */
    p { line-height: 1.6 !important; }
    /* Sidebar: softer dark, reduced noise, intentional spacing */
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #1e293b 0%, #2d3a4f 50%, #334155 100%); }
    [data-testid="stSidebar"] > div { max-height: 100vh; overflow-y: auto; padding: 0 1rem 1.25rem 1rem !important; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div { margin-bottom: 0.4rem !important; }
    [data-testid="stSidebar"] hr { margin: 1rem 0 !important; opacity: 0.6; }
    /* Chat input: primary CTA, focal point, subtle glow */
    [data-testid="stChatInput"],
    .stChatInputContainer > div {
        padding: 1rem 1.2rem !important;
        border-radius: 14px !important;
        border: 2px solid #cbd5e1 !important;
        background: #ffffff !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08), 0 0 0 1px rgba(30, 41, 59, 0.04) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    [data-testid="stChatInput"]:hover,
    .stChatInputContainer:hover > div {
        border-color: #94a3b8 !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.1), 0 0 0 1px rgba(30, 41, 59, 0.06) !important;
    }
    [data-testid="stChatInput"]:focus-within,
    .stChatInputContainer:focus-within > div {
        border-color: #1e293b !important;
        box-shadow: 0 0 0 4px rgba(30, 41, 59, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

class LegalChatbotUI:
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.session_state = st.session_state
        self.auth_ui = AuthUI(api_base_url=self.api_base_url)
        
        # Initialize session state
        if "messages" not in self.session_state:
            self.session_state.messages = []
        if "api_status" not in self.session_state:
            self.session_state.api_status = "unknown"
        if "current_page" not in self.session_state:
            self.session_state.current_page = "chat"
    
    def check_api_status(self) -> bool:
        """Check if the FastAPI server is running"""
        try:
            response = requests.get(f"{self.api_base_url}/api/v1/health", timeout=5)
            if response.status_code == 200:
                self.session_state.api_status = "connected"
                return True
            else:
                self.session_state.api_status = "error"
                return False
        except requests.exceptions.RequestException:
            self.session_state.api_status = "disconnected"
            return False
    
    def send_chat_request(self, query: str, mode: str, top_k: int = 3) -> Dict[str, Any]:
        """Send chat request to FastAPI with authentication"""
        try:
            # Ensure token is valid
            if not self.auth_ui.ensure_authenticated():
                return {
                    "error": "Authentication Error",
                    "detail": "Please login to use the chat feature"
                }
            
            headers = self.auth_ui.get_auth_headers()
            
            response = requests.post(
                f"{self.api_base_url}/api/v1/chat",
                json={
                    "query": query,
                    "mode": mode,
                    "top_k": top_k
                },
                headers=headers,
                timeout=120  # Increased timeout for RAG + OpenAI processing
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                # Token expired, try refresh
                if self.auth_ui.refresh_access_token():
                    headers = self.auth_ui.get_auth_headers()
                    response = requests.post(
                        f"{self.api_base_url}/api/v1/chat",
                        json={
                            "query": query,
                            "mode": mode,
                            "top_k": top_k
                        },
                        headers=headers,
                        timeout=120  # Increased timeout for RAG + OpenAI processing
                    )
                    if response.status_code == 200:
                        return response.json()
                
                return {
                    "error": "Authentication Error",
                    "detail": "Session expired. Please login again."
                }
            else:
                return {
                    "error": f"API Error: {response.status_code}",
                    "detail": response.text
                }
        except requests.exceptions.RequestException as e:
            return {
                "error": "Connection Error",
                "detail": str(e)
            }
    
    def display_citations(self, citations: list):
        """Display citations in an expandable format — presentation only, no logic changes"""
        if not citations:
            st.info("No citations available")
            return
        
        n = len(citations)
        cited_badge = '<span class="cite-badge cite-badge-green">Cited</span>'
        
        st.markdown("<div class='citations-spacer'></div>", unsafe_allow_html=True)
        st.markdown(f"{cited_badge} This answer is grounded in the following sources.", unsafe_allow_html=True)
        
        with st.expander(f"Sources ({n})", expanded=False):
            for i, citation in enumerate(citations):
                # Handle both old format (dict) and new format (Source object)
                if isinstance(citation, dict):
                    citation_id = citation.get('chunk_id', citation.get('id', f"citation_{i}"))
                    title = citation.get('title', 'Unknown')
                    section = citation.get('section', citation.get('metadata', {}).get('section', 'Unknown'))
                    text_snippet = citation.get('text_snippet', citation.get('text', ''))
                    similarity_score = citation.get('similarity_score', 0.0)
                    url = citation.get('url')
                    metadata = citation.get('metadata', {})
                else:
                    citation_id = citation.chunk_id if hasattr(citation, 'chunk_id') else f"citation_{i}"
                    title = citation.title if hasattr(citation, 'title') else 'Unknown'
                    section = citation.metadata.get('section', 'Unknown') if hasattr(citation, 'metadata') else 'Unknown'
                    text_snippet = citation.text_snippet if hasattr(citation, 'text_snippet') else ''
                    similarity_score = citation.similarity_score if hasattr(citation, 'similarity_score') else 0.0
                    url = citation.url if hasattr(citation, 'url') else None
                    metadata = citation.metadata if hasattr(citation, 'metadata') else {}
                
                # Expander title: numbered badge [1], [2] + title, section if present
                num = i + 1
                expander_title = f"[{num}] {title}"
                if section and section != 'Unknown':
                    expander_title += f" — {section}"
                
                with st.expander(expander_title):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"**{title}**")
                        if section and section != 'Unknown':
                            st.caption(f"Section: {section}")
                    with col2:
                        st.caption(f"Relevance: {similarity_score:.2f}")
                        if url:
                            st.caption(f"[Link]({url})")
                    
                    st.caption("Excerpt")
                    st.text(text_snippet[:500])
                    
                    if metadata:
                        meta_parts = []
                        if metadata.get('corpus'):
                            meta_parts.append(f"Corpus: {metadata['corpus']}")
                        if metadata.get('jurisdiction'):
                            meta_parts.append(f"Jurisdiction: {metadata['jurisdiction']}")
                        if meta_parts:
                            st.caption(" · ".join(meta_parts))
    
    def display_response_metadata(self, response: Dict[str, Any]):
        """Display response metadata and validation info"""
        import re
        
        with st.expander("Response details"):
            col1, col2 = st.columns(2)
            
            # Extract citation count from answer text (count [1], [2], etc.)
            answer = response.get('answer', '')
            citation_pattern = r'\[(\d+)\]'
            found_citations = re.findall(citation_pattern, answer)
            citation_count = len(set(found_citations))  # Count unique citations
            
            with col1:
                # Get model and mode from response
                model = response.get('model_used', response.get('model', 'N/A'))
                mode = response.get('mode', response.get('response_mode', 'N/A'))
                
                st.write("**Model:**", model)
                st.write("**Mode:**", mode)
                st.write("**Citations:**", citation_count)
            
            with col2:
                # Check guardrails status
                guardrails_applied = response.get('guardrails_applied', False)
                safety = response.get('safety', {})
                
                # Display guardrails status with color
                if isinstance(safety, dict):
                    is_safe = safety.get('is_safe', True)
                    if guardrails_applied:
                        if is_safe:
                            st.success("Guardrails applied")
                        else:
                            st.warning("Guardrails applied (warnings)")
                    else:
                        st.error("Guardrails not applied")
                else:
                    st.write("**Guardrails:**", "Applied" if guardrails_applied else "Not applied")
                
                # Display guardrails result if available
                if response.get('guardrails_result'):
                    gr = response['guardrails_result']
                    rules_applied = gr.get('rules_applied', [])
                    if rules_applied:
                        st.write("**Rules Applied:**", ", ".join(rules_applied))
            
            # Display retrieval info
            sources = response.get('sources', [])
            if sources:
                st.write("**Sources Retrieved:**", len(sources))
            
            # Display metrics if available
            metrics = response.get('metrics', {})
            if metrics and isinstance(metrics, dict):
                st.write("**Metrics:**")
                st.write(f"- Retrieval time: {metrics.get('retrieval_time_ms', 0):.1f}ms")
                st.write(f"- Generation time: {metrics.get('generation_time_ms', 0):.1f}ms")
                st.write(f"- Total time: {metrics.get('total_time_ms', 0):.1f}ms")
                st.write(f"- Confidence: {metrics.get('retrieval_score', 0):.3f}")
                
                # Calculate average similarity from sources if available
                sources_list = response.get('sources', [])
                if sources_list:
                    similarity_scores = []
                    for source in sources_list:
                        if isinstance(source, dict):
                            sim = source.get('similarity_score', 0.0)
                        else:
                            sim = getattr(source, 'similarity_score', 0.0) if hasattr(source, 'similarity_score') else 0.0
                        if sim > 0:
                            similarity_scores.append(sim)
                    
                    if similarity_scores:
                        avg_similarity = sum(similarity_scores) / len(similarity_scores)
                        st.write(f"- Avg similarity: {avg_similarity:.3f}")
    
    def render_sidebar(self):
        """Render the sidebar with controls and user profile"""
        st.sidebar.title("Legal Chatbot")
        st.sidebar.caption("UK Law · RAG-powered · Role-based access")
        
        # Check authentication first
        if not self.auth_ui.ensure_authenticated():
            st.sidebar.info("Please sign in to continue")
            if st.sidebar.button("Sign in", use_container_width=True):
                self.session_state.current_page = "login"
                st.rerun()
            return None, None
        
        # Display user profile
        self.auth_ui.render_user_profile()
        
        # Navigation (Admin: list + role; Solicitor/Admin: documents)
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Navigation**")
        nav_options = ["Chat", "Settings"]
        if self.auth_ui.has_role("solicitor", "admin"):
            nav_options.insert(1, "Documents")
        if self.auth_ui.has_role("admin"):
            nav_options.insert(len(nav_options) - 1, "Admin")
        page_index = 0
        if self.session_state.current_page == "chat":
            page_index = 0
        elif self.session_state.current_page == "documents":
            page_index = nav_options.index("Documents") if "Documents" in nav_options else 0
        elif self.session_state.current_page == "admin":
            page_index = nav_options.index("Admin") if "Admin" in nav_options else 0
        else:
            page_index = nav_options.index("Settings")
        page = st.sidebar.radio("Page", nav_options, index=page_index, label_visibility="collapsed")
        if page == "Chat":
            self.session_state.current_page = "chat"
        elif page == "Documents":
            self.session_state.current_page = "documents"
        elif page == "Admin":
            self.session_state.current_page = "admin"
        elif page == "Settings":
            self.session_state.current_page = "settings"
        
        # API Status
        st.sidebar.markdown("---")
        api_connected = self.check_api_status()
        
        if api_connected:
            st.sidebar.success("API connected")
        else:
            st.sidebar.error("API disconnected")
            st.sidebar.caption("Start the backend server to connect.")
        
        # Mode Selection (only show in chat page)
        if self.session_state.current_page == "chat":
            st.sidebar.markdown("---")
            st.sidebar.markdown('<span class="sidebar-mode-label">Response mode</span>', unsafe_allow_html=True)
            
            user_role = self.auth_ui.get_user_role()
            if user_role in ["solicitor", "admin"]:
                mode_options = ["solicitor", "public"]
                mode_labels = {"solicitor": "Solicitor — technical", "public": "Public — plain language"}
                mode = st.sidebar.radio(
                    "Mode",
                    mode_options,
                    index=0,
                    format_func=lambda x: mode_labels.get(x, x),
                    label_visibility="collapsed",
                    help="Solicitor: Technical legal language. Public: Plain language."
                )
            else:
                mode = "public"
                st.sidebar.caption("Mode: Public (plain language)")
            
            st.sidebar.markdown("")  # spacing
            
            # Advanced Settings
            st.sidebar.markdown("**Advanced**")
            top_k = st.sidebar.slider(
                "Sources to retrieve",
                min_value=1,
                max_value=10,
                value=3,
                help="How many legal sources to retrieve for the answer"
            )
            
            # Clear Chat
            if st.sidebar.button("Clear chat", use_container_width=True):
                self.session_state.messages = []
                st.rerun()
            
            # About Section
            st.sidebar.markdown("---")
            st.sidebar.markdown("**About**")
            st.sidebar.caption(
                "Answers based on UK legislation (Sale of Goods Act, Employment Rights Act, Data Protection Act). "
                "Educational use only — not legal advice."
            )
            
            return mode, top_k
        else:
            return None, None
    
    def render_main_interface(self, mode: Optional[str] = None, top_k: Optional[int] = None):
        """Render the main interface based on current page"""
        page = self.session_state.current_page
        
        if page == "chat":
            self._render_chat_interface(mode or "public", top_k or 3)
        elif page == "documents":
            self._render_documents_interface()
        elif page == "admin":
            self._render_admin_interface()
        elif page == "settings":
            self._render_settings_interface()
        else:
            self._render_chat_interface(mode or "public", top_k or 3)
    
    def _render_chat_interface(self, mode: str, top_k: int):
        """Render the main chat interface"""
        st.markdown("""
        <div class="chat-hero">
            <h1 class="chat-hero-title">Legal Chatbot</h1>
            <p class="chat-hero-subtitle">AI-powered legal assistant for UK law with verified citations.</p>
            <ul class="chat-hero-features">
                <li>Retrieval-Augmented Generation (RAG)</li>
                <li>Source-backed legal answers</li>
                <li>Role-based access (Public / Solicitor / Admin)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Empty state: clean, screenshot-ready
        if not self.session_state.messages:
            st.markdown("<div style='margin-top: 0.75rem; margin-bottom: 1.25rem;'></div>", unsafe_allow_html=True)
            st.caption("Answers are grounded in UK legislation. Type a question below to begin.")
        
        # Display chat messages
        for message in self.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display citations if available
                if message["role"] == "assistant" and "citations" in message:
                    self.display_citations(message["citations"])
                
                # Display metadata if available
                if message["role"] == "assistant" and "metadata" in message:
                    self.display_response_metadata(message["metadata"])
        
        # Chat input
        if prompt := st.chat_input("Ask about UK consumer rights…"):
            # Add user message
            self.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("Searching sources & generating answer..."):
                    response = self.send_chat_request(prompt, mode, top_k)
                
                if "error" in response:
                    st.error(f"**{response['error']}**")
                    st.caption(response.get('detail', ''))
                else:
                    # Display answer
                    answer = response.get('answer', 'No answer provided')
                    st.markdown(answer)
                    
                    # Extract metadata from response
                    safety = response.get('safety', {})
                    metrics = response.get('metrics', {})
                    
                    # Determine guardrails status from safety reasoning
                    reasoning = safety.get('reasoning', '') if isinstance(safety, dict) else ''
                    guardrails_applied = 'Guardrails' in reasoning
                    
                    # Extract model and mode from response
                    model_used = response.get('model_used', response.get('model', 'N/A'))
                    mode = response.get('response_mode', response.get('mode', 'N/A'))
                    
                    # Add metadata to response dict for display
                    response_with_metadata = response.copy()
                    response_with_metadata['guardrails_applied'] = guardrails_applied
                    response_with_metadata['model_used'] = model_used
                    response_with_metadata['mode'] = mode
                    response_with_metadata['safety'] = safety if isinstance(safety, dict) else {}
                    response_with_metadata['metrics'] = metrics if isinstance(metrics, dict) else {}
                    
                    # Store response with metadata
                    self.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": response.get('sources', []),
                        "metadata": response_with_metadata
                    })
                    
                    # Display citations
                    sources = response.get('sources', [])
                    if sources:
                        self.display_citations(sources)
                    
                    # Display metadata
                    if response_with_metadata:
                        self.display_response_metadata(response_with_metadata)
        
        # Footer disclaimer — subtle for screenshots
        st.markdown("---")
        st.caption("Educational use only. Not legal advice.")
    
    def _render_documents_interface(self):
        """Render documents management interface"""
        st.title("My Documents")
        st.caption("Upload and manage your private legal corpus. Documents are searchable in chat.")
        
        # Check if user can manage documents (solicitor/admin only)
        if not require_role("solicitor", "admin"):
            st.warning("Document management is only available for Solicitor and Admin users.")
            return
        
        # List documents
        try:
            headers = self.auth_ui.get_auth_headers()
            response = requests.get(
                f"{self.api_base_url}/api/v1/documents",
                headers=headers,
                params={"skip": 0, "limit": 100},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                documents = data.get("documents", [])
                total = data.get("total", 0)
                
                st.success(f"You have {total} document(s)")
                
                if documents:
                    for doc in documents:
                        doc_id = doc.get("id")
                        title = doc.get("title") or doc.get("original_filename", "Untitled")
                        status = doc.get("status", "unknown")
                        with st.expander(f"{title} — {status}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Type:** {doc.get('file_type', 'N/A')}")
                                st.write(f"**Size:** {doc.get('file_size', 0) / 1024:.2f} KB")
                                st.write(f"**Chunks:** {doc.get('chunks_count', 0)}")
                            with col2:
                                st.write(f"**Jurisdiction:** {doc.get('jurisdiction', 'N/A')}")
                                st.write(f"**Created:** {doc.get('created_at', 'N/A')}")
                                if doc.get("tags"):
                                    st.write(f"**Tags:** {doc.get('tags')}")
                            if doc.get("description"):
                                st.write(f"**Description:** {doc.get('description')}")
                            st.markdown("---")
                            btn_col1, btn_col2, _ = st.columns([1, 1, 3])
                            with btn_col1:
                                if st.button("Reprocess", key=f"reprocess_{doc_id}", use_container_width=True):
                                    try:
                                        with st.spinner("Reprocessing..."):
                                            r = requests.post(
                                                f"{self.api_base_url}/api/v1/documents/{doc_id}/reprocess",
                                                headers=headers,
                                                timeout=120,
                                            )
                                        if r.status_code == 200:
                                            st.success("Reprocessing started.")
                                            st.rerun()
                                        else:
                                            st.error(r.json().get("detail", r.text))
                                    except Exception as e:
                                        st.error(str(e))
                            with btn_col2:
                                if st.button("Delete", key=f"delete_{doc_id}", use_container_width=True):
                                    try:
                                        r = requests.delete(
                                            f"{self.api_base_url}/api/v1/documents/{doc_id}",
                                            headers=headers,
                                            timeout=10,
                                        )
                                        if r.status_code == 204 or r.status_code == 200:
                                            st.success("Document deleted.")
                                            st.rerun()
                                        else:
                                            st.error(r.json().get("detail", r.text))
                                    except Exception as e:
                                        st.error(str(e))
                else:
                    st.info("No documents uploaded yet. Use the upload button to add documents.")
            else:
                st.error(f"Failed to load documents: {response.text}")
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")
        
        # Upload button
        st.markdown("---")
        st.subheader("Upload document")
        st.caption("PDF, DOCX, or TXT. Documents are indexed and searchable in chat.")
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "txt"],
            help="Upload a legal document to add to your private corpus"
        )
        
        if uploaded_file:
            col1, col2, col3 = st.columns(3)
            title = col1.text_input("Title (optional)")
            jurisdiction = col2.text_input("Jurisdiction", value="UK")
            tags = col3.text_input("Tags (comma-separated)")
            
            if st.button("Upload Document", use_container_width=True):
                with st.spinner("Uploading and indexing..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        data = {}
                        if title:
                            data["title"] = title
                        if jurisdiction:
                            data["jurisdiction"] = jurisdiction
                        if tags:
                            data["tags"] = tags
                        
                        headers = self.auth_ui.get_auth_headers()
                        response = requests.post(
                            f"{self.api_base_url}/api/v1/documents/upload",
                            headers=headers,
                            files=files,
                            data=data,
                            timeout=60
                        )
                        
                        if response.status_code == 201:
                            st.success("Document uploaded successfully.")
                            st.rerun()
                        else:
                            st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Upload error: {str(e)}")

    def _render_admin_interface(self):
        """Render Admin UI: list users, view roles, change role (existing API)."""
        if not require_role("admin"):
            st.warning("Admin access required.")
            return
        st.title("Admin")
        st.caption("Manage users and roles.")
        try:
            headers = self.auth_ui.get_auth_headers()
            response = requests.get(
                f"{self.api_base_url}/api/v1/auth/users",
                headers=headers,
                params={"skip": 0, "limit": 200},
                timeout=10
            )
            if response.status_code != 200:
                st.error(f"Failed to load users: {response.text}")
                return
            data = response.json()
            users = data.get("users", [])
            total = data.get("total", 0)
            st.caption(f"Total users: {total}")
            if not users:
                st.info("No users found.")
                return
            role_options = ["public", "solicitor", "admin"]
            # Header row
            h1, h2, h3, h4, h5 = st.columns([2, 1.5, 1, 0.8, 1.2])
            with h1:
                st.markdown("**Email**")
            with h2:
                st.markdown("**Full name**")
            with h3:
                st.markdown("**Role**")
            with h4:
                st.markdown("**Active**")
            with h5:
                st.markdown("**Change role**")
            st.divider()
            for u in users:
                uid = u.get("id")
                email = u.get("email", "")
                full_name = u.get("full_name") or ""
                role = u.get("role", "public")
                is_active = u.get("is_active", True)
                with st.container():
                    c1, c2, c3, c4, c5 = st.columns([2, 1.5, 1, 0.8, 1.2])
                    with c1:
                        st.text(email)
                    with c2:
                        st.text(full_name or "—")
                    with c3:
                        st.text(role)
                    with c4:
                        st.text("Active" if is_active else "Inactive")
                    with c5:
                        with st.form(key=f"admin_role_form_{uid}"):
                            new_role = st.selectbox(
                                "Role",
                                role_options,
                                index=role_options.index(role) if role in role_options else 0,
                                key=f"admin_role_select_{uid}"
                            )
                            if st.form_submit_button("Update"):
                                try:
                                    r = requests.put(
                                        f"{self.api_base_url}/api/v1/auth/users/{uid}",
                                        headers=headers,
                                        json={"role": new_role},
                                        timeout=10
                                    )
                                    if r.status_code == 200:
                                        st.success("Role updated.")
                                        st.rerun()
                                    else:
                                        st.error(r.json().get("detail", r.text))
                                except Exception as e:
                                    st.error(str(e))
                    st.divider()
        except Exception as e:
            st.error(f"Error loading users: {str(e)}")

    def _render_settings_interface(self):
        """Render user settings interface"""
        st.title("Settings")
        st.caption("Manage your profile and security.")
        
        user = self.auth_ui.session_state.user
        if not user:
            st.error("User information not available")
            return
        
        # Profile information
        st.subheader("Profile")
        
        with st.form("profile_form"):
            email = st.text_input("Email", value=user.get("email", ""), disabled=True)
            full_name = st.text_input("Full Name", value=user.get("full_name", ""))
            username = st.text_input("Username", value=user.get("username", ""))
            
            submit = st.form_submit_button("Update Profile", use_container_width=True)
            
            if submit:
                try:
                    headers = self.auth_ui.get_auth_headers()
                    response = requests.put(
                        f"{self.api_base_url}/api/v1/auth/me",
                        headers=headers,
                        json={
                            "full_name": full_name,
                            "username": username
                        },
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        updated_user = response.json()
                        self.auth_ui._store_user(updated_user)
                        st.success("Profile updated successfully.")
                        st.rerun()
                    else:
                        st.error(f"Update failed: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Update error: {str(e)}")
        
        # Change password
        st.markdown("---")
        st.subheader("Change password")
        
        with st.form("password_form"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            submit = st.form_submit_button("Change Password", use_container_width=True)
            
            if submit:
                if new_password != confirm_password:
                    st.error("New passwords do not match")
                elif len(new_password) < 8:
                    st.error("Password must be at least 8 characters")
                else:
                    try:
                        headers = self.auth_ui.get_auth_headers()
                        response = requests.post(
                            f"{self.api_base_url}/api/v1/auth/change-password",
                            headers=headers,
                            json={
                                "current_password": current_password,
                                "new_password": new_password
                            },
                            timeout=10
                        )
                        
                        if response.status_code == 200 or response.status_code == 204:
                            st.success("Password changed successfully.")
                        else:
                            st.error(f"Password change failed: {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Password change error: {str(e)}")
    
    def run(self):
        """Run the Streamlit app with authentication"""
        # Check for OAuth callback (code in query params)
        query_params = st.query_params
        if "code" in query_params:
            code = query_params.get("code")
            # Try to get provider from query params or session
            provider = query_params.get("provider") or self.session_state.get("oauth_provider")
            
            if provider and code:
                if self.auth_ui.handle_oauth_callback(code, provider=provider):
                    st.success("OAuth login successful.")
                    time.sleep(1)
                    # Clear query params
                    st.query_params.clear()
                    st.rerun()
            else:
                st.warning("OAuth callback received but provider information missing")
        
        # Check if user is authenticated
        if not self.auth_ui.ensure_authenticated():
            # Show login page
            self.auth_ui.render_authentication_page()
            return
        
        # Render sidebar and main interface
        mode, top_k = self.render_sidebar()
        
        if mode is None and top_k is None:
            # Not on chat page, render current page
            self.render_main_interface()
        else:
            # On chat page
            self.render_main_interface(mode, top_k)

# Run the app
if __name__ == "__main__":
    ui = LegalChatbotUI()
    ui.run()

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
                timeout=30
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
                        timeout=30
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
        """Display citations in an expandable format"""
        if not citations:
            st.info("No citations available")
            return
        
        st.subheader("📚 Sources & Citations")
        
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
                # Source object (Pydantic model)
                citation_id = citation.chunk_id if hasattr(citation, 'chunk_id') else f"citation_{i}"
                title = citation.title if hasattr(citation, 'title') else 'Unknown'
                section = citation.metadata.get('section', 'Unknown') if hasattr(citation, 'metadata') else 'Unknown'
                text_snippet = citation.text_snippet if hasattr(citation, 'text_snippet') else ''
                similarity_score = citation.similarity_score if hasattr(citation, 'similarity_score') else 0.0
                url = citation.url if hasattr(citation, 'url') else None
                metadata = citation.metadata if hasattr(citation, 'metadata') else {}
            
            # Create expander title with section info
            expander_title = f"Source [{citation_id}]"
            if section and section != 'Unknown':
                expander_title += f" - {section}"
            
            with st.expander(expander_title):
                st.write(f"**Title:** {title}")
                if section and section != 'Unknown':
                    st.write(f"**Section:** {section}")
                if url:
                    st.write(f"**URL:** {url}")
                st.write(f"**Similarity Score:** {similarity_score:.3f}")
                st.write(f"**Text Snippet:**")
                st.text(text_snippet[:500])  # Show first 500 chars
                
                # Show additional metadata if available
                if metadata:
                    if metadata.get('corpus'):
                        st.write(f"**Corpus:** {metadata['corpus']}")
                    if metadata.get('jurisdiction'):
                        st.write(f"**Jurisdiction:** {metadata['jurisdiction']}")
    
    def display_response_metadata(self, response: Dict[str, Any]):
        """Display response metadata and validation info"""
        import re
        
        with st.expander("🔍 Response Details"):
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
                            st.success("✅ **Guardrails Applied:** True")
                        else:
                            st.warning("⚠️ **Guardrails Applied:** True (warnings)")
                    else:
                        st.error("❌ **Guardrails Applied:** False")
                else:
                    st.write("**Guardrails Applied:**", "✅ True" if guardrails_applied else "❌ False")
                
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
                st.write(f"- Avg similarity: {ri.get('avg_similarity_score', 0):.3f}")
    
    def render_sidebar(self):
        """Render the sidebar with controls and user profile"""
        st.sidebar.title("⚖️ Legal Chatbot")
        
        # Check authentication first
        if not self.auth_ui.ensure_authenticated():
            st.sidebar.info("🔐 Please login to continue")
            if st.sidebar.button("Go to Login", use_container_width=True):
                self.session_state.current_page = "login"
                st.rerun()
            return None, None
        
        # Display user profile
        self.auth_ui.render_user_profile()
        
        # Navigation
        st.sidebar.markdown("---")
        st.sidebar.subheader("🧭 Navigation")
        
        page = st.sidebar.radio(
            "Page",
            ["💬 Chat", "📄 Documents", "⚙️ Settings"],
            index=0 if self.session_state.current_page == "chat" else 1 if self.session_state.current_page == "documents" else 2
        )
        
        if page == "💬 Chat":
            self.session_state.current_page = "chat"
        elif page == "📄 Documents":
            self.session_state.current_page = "documents"
        elif page == "⚙️ Settings":
            self.session_state.current_page = "settings"
        
        # API Status
        st.sidebar.markdown("---")
        api_connected = self.check_api_status()
        
        if api_connected:
            st.sidebar.success("🟢 API Connected")
        else:
            st.sidebar.error("🔴 API Disconnected")
            st.sidebar.info("Start the FastAPI server with: `uvicorn app.api.main:app --reload --port 8000`")
        
        # Mode Selection (only show in chat page)
        if self.session_state.current_page == "chat":
            st.sidebar.markdown("---")
            st.sidebar.subheader("🎯 Response Mode")
            
            user_role = self.auth_ui.get_user_role()
            if user_role in ["solicitor", "admin"]:
                mode_options = ["solicitor", "public"]
            else:
                mode_options = ["public"]
            
            mode = st.sidebar.selectbox(
                "Choose response style:",
                mode_options,
                index=0,
                help="Solicitor: Technical legal language. Public: Plain language explanations."
            )
            
            # Advanced Settings
            st.sidebar.subheader("⚙️ Advanced Settings")
            top_k = st.sidebar.slider(
                "Number of sources to retrieve:",
                min_value=1,
                max_value=10,
                value=3,
                help="How many legal sources to retrieve for the answer"
            )
            
            # Clear Chat
            if st.sidebar.button("🗑️ Clear Chat History"):
                self.session_state.messages = []
                st.rerun()
            
            # About Section
            st.sidebar.markdown("---")
            st.sidebar.subheader("ℹ️ About")
            st.sidebar.info("""
            This legal chatbot provides answers based on UK law using:
            - Sale of Goods Act 1979
            - Employment Rights Act 1996
            - Data Protection Act 2018
            
            **Note:** This is for educational purposes only and does not constitute legal advice.
            """)
            
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
        elif page == "settings":
            self._render_settings_interface()
        else:
            self._render_chat_interface(mode or "public", top_k or 3)
    
    def _render_chat_interface(self, mode: str, top_k: int):
        """Render the main chat interface"""
        st.title("⚖️ Legal Chatbot")
        st.markdown("Ask questions about UK law and get answers with proper citations!")
        
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
        if prompt := st.chat_input("Ask a legal question..."):
            # Add user message
            self.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = self.send_chat_request(prompt, mode, top_k)
                
                if "error" in response:
                    st.error(f"Error: {response['error']}")
                    st.text(response.get('detail', ''))
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
                    
                    # Extract model and mode from safety reasoning or defaults
                    model_used = response.get('model_used', 'N/A')
                    mode = response.get('mode', 'N/A')
                    
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
    
    def _render_documents_interface(self):
        """Render documents management interface"""
        st.title("📄 My Documents")
        
        # Check if user can manage documents (solicitor/admin only)
        if not require_role("solicitor", "admin"):
            st.warning("⚠️ Document management is only available for Solicitor and Admin users.")
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
                
                st.success(f"📚 You have {total} document(s)")
                
                if documents:
                    for doc in documents:
                        with st.expander(f"📄 {doc.get('title') or doc.get('original_filename', 'Untitled')} - {doc.get('status', 'unknown')}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Type:** {doc.get('file_type', 'N/A')}")
                                st.write(f"**Size:** {doc.get('file_size', 0) / 1024:.2f} KB")
                                st.write(f"**Chunks:** {doc.get('chunks_count', 0)}")
                            with col2:
                                st.write(f"**Jurisdiction:** {doc.get('jurisdiction', 'N/A')}")
                                st.write(f"**Created:** {doc.get('created_at', 'N/A')}")
                                if doc.get('tags'):
                                    st.write(f"**Tags:** {doc.get('tags')}")
                            
                            if doc.get('description'):
                                st.write(f"**Description:** {doc.get('description')}")
                else:
                    st.info("No documents uploaded yet. Use the upload button to add documents.")
            else:
                st.error(f"Failed to load documents: {response.text}")
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")
        
        # Upload button
        st.markdown("---")
        st.subheader("📤 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file (PDF, DOCX, TXT)",
            type=["pdf", "docx", "txt"],
            help="Upload a legal document to add to your private corpus"
        )
        
        if uploaded_file:
            col1, col2, col3 = st.columns(3)
            title = col1.text_input("Title (optional)")
            jurisdiction = col2.text_input("Jurisdiction", value="UK")
            tags = col3.text_input("Tags (comma-separated)")
            
            if st.button("Upload Document", use_container_width=True):
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
                        st.success("✅ Document uploaded successfully!")
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Upload error: {str(e)}")
    
    def _render_settings_interface(self):
        """Render user settings interface"""
        st.title("⚙️ Settings")
        
        user = self.auth_ui.session_state.user
        if not user:
            st.error("User information not available")
            return
        
        # Profile information
        st.subheader("👤 Profile Information")
        
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
                        st.success("✅ Profile updated successfully!")
                        st.rerun()
                    else:
                        st.error(f"Update failed: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Update error: {str(e)}")
        
        # Change password
        st.markdown("---")
        st.subheader("🔒 Change Password")
        
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
                            st.success("✅ Password changed successfully!")
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
                    st.success("OAuth login successful!")
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

# frontend/app.py - Streamlit Legal Chatbot UI
import streamlit as st
import requests
import json
from typing import Dict, Any
import time

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
        
        # Initialize session state
        if "messages" not in self.session_state:
            self.session_state.messages = []
        if "api_status" not in self.session_state:
            self.session_state.api_status = "unknown"
    
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
        """Send chat request to FastAPI"""
        try:
            response = requests.post(
                f"{self.api_base_url}/api/v1/chat",
                json={
                    "query": query,
                    "mode": mode,
                    "top_k": top_k
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
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
            with st.expander(f"Citation [{citation['id']}] - {citation['section']}"):
                st.write(f"**Title:** {citation['title']}")
                st.write(f"**Section:** {citation['section']}")
                st.write(f"**Text Snippet:**")
                st.text(citation['text_snippet'])
    
    def display_response_metadata(self, response: Dict[str, Any]):
        """Display response metadata and validation info"""
        with st.expander("🔍 Response Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model:**", response.get('model_used', 'N/A'))
                st.write("**Mode:**", response.get('mode', 'N/A'))
                st.write("**Citations:**", len(response.get('citations', [])))
            
            with col2:
                st.write("**Guardrails Applied:**", response.get('guardrails_applied', False))
                
                # Display validation info
                if response.get('query_validation'):
                    qv = response['query_validation']
                    st.write("**Query Validation:**", qv.get('reason', 'N/A'))
                
                if response.get('response_validation'):
                    rv = response['response_validation']
                    st.write("**Response Validation:**", rv.get('reason', 'N/A'))
            
            # Display retrieval info
            if response.get('retrieval_info'):
                ri = response['retrieval_info']
                st.write("**Retrieval Info:**")
                st.write(f"- Chunks retrieved: {ri.get('num_chunks_retrieved', 0)}")
                st.write(f"- Max similarity: {ri.get('max_similarity_score', 0):.3f}")
                st.write(f"- Avg similarity: {ri.get('avg_similarity_score', 0):.3f}")
    
    def render_sidebar(self):
        """Render the sidebar with controls"""
        st.sidebar.title("⚖️ Legal Chatbot")
        
        # API Status
        api_connected = self.check_api_status()
        
        if api_connected:
            st.sidebar.success("🟢 API Connected")
        else:
            st.sidebar.error("🔴 API Disconnected")
            st.sidebar.info("Start the FastAPI server with: `uvicorn app:app --reload --port 8000`")
        
        # Mode Selection
        st.sidebar.subheader("🎯 Response Mode")
        mode = st.sidebar.selectbox(
            "Choose response style:",
            ["solicitor", "public"],
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
        st.sidebar.subheader("ℹ️ About")
        st.sidebar.info("""
        This legal chatbot provides answers based on UK law using:
        - Sale of Goods Act 1979
        - Employment Rights Act 1996
        - Data Protection Act 2018
        
        **Note:** This is for educational purposes only and does not constitute legal advice.
        """)
        
        return mode, top_k
    
    def render_main_interface(self, mode: str, top_k: int):
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
                    st.markdown(response['answer'])
                    
                    # Store response with metadata
                    self.session_state.messages.append({
                        "role": "assistant",
                        "content": response['answer'],
                        "citations": response.get('citations', []),
                        "metadata": response
                    })
                    
                    # Display citations
                    self.display_citations(response.get('citations', []))
                    
                    # Display metadata
                    self.display_response_metadata(response)
    
    def run(self):
        """Run the Streamlit app"""
        mode, top_k = self.render_sidebar()
        self.render_main_interface(mode, top_k)

# Run the app
if __name__ == "__main__":
    ui = LegalChatbotUI()
    ui.run()

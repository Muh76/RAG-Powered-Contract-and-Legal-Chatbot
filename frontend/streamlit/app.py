# Legal Chatbot - Streamlit Frontend

import streamlit as st
import requests
import json
from datetime import datetime
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go


# Page configuration
st.set_page_config(
    page_title="Legal Chatbot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .source-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    .safety-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"


def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if "user_id" not in st.session_state:
        st.session_state.user_id = "anonymous_user"


def call_chat_api(query: str, mode: str = "public", top_k: int = 10) -> Dict[str, Any]:
    """Call the chat API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={
                "query": query,
                "mode": mode,
                "user_id": st.session_state.user_id,
                "session_id": st.session_state.session_id,
                "top_k": top_k
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def display_sources(sources: List[Dict[str, Any]]):
    """Display source documents"""
    if not sources:
        return
    
    st.markdown("### 📚 Sources")
    for i, source in enumerate(sources, 1):
        with st.expander(f"Source {i}: {source.get('title', 'Unknown')}"):
            st.markdown(f"**Similarity Score:** {source.get('similarity_score', 0):.3f}")
            if source.get('url'):
                st.markdown(f"**URL:** {source['url']}")
            st.markdown(f"**Content:**")
            st.text(source.get('text_snippet', 'No content available'))


def display_safety_report(safety: Dict[str, Any]):
    """Display safety report"""
    if not safety.get('is_safe', True):
        st.markdown("### ⚠️ Safety Notice")
        st.markdown(
            f"""
            <div class="safety-warning">
                <strong>Warning:</strong> {safety.get('reasoning', 'Content flagged for safety')}
                <br><strong>Flags:</strong> {', '.join(safety.get('flags', []))}
            </div>
            """,
            unsafe_allow_html=True
        )


def display_metrics(metrics: Dict[str, Any]):
    """Display performance metrics"""
    with st.expander("📊 Performance Metrics"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Retrieval Time", f"{metrics.get('retrieval_time_ms', 0):.1f}ms")
            st.metric("Generation Time", f"{metrics.get('generation_time_ms', 0):.1f}ms")
        
        with col2:
            st.metric("Total Time", f"{metrics.get('total_time_ms', 0):.1f}ms")
            st.metric("Retrieval Score", f"{metrics.get('retrieval_score', 0):.3f}")
        
        with col3:
            st.metric("Answer Relevance", f"{metrics.get('answer_relevance_score', 0):.3f}")


def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">⚖️ Legal Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("AI-Powered Legal Assistant with RAG (Retrieval-Augmented Generation)")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Chat mode selection
        mode = st.selectbox(
            "Response Mode",
            ["public", "solicitor"],
            help="Public: Plain language explanations\nSolicitor: Technical legal details"
        )
        
        # Top-k retrieval
        top_k = st.slider(
            "Number of Sources",
            min_value=1,
            max_value=20,
            value=10,
            help="Number of source documents to retrieve"
        )
        
        # User ID
        user_id = st.text_input("User ID", value=st.session_state.user_id)
        st.session_state.user_id = user_id
        
        # Clear chat
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        # API Status
        st.header("📡 API Status")
        try:
            health_response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if health_response.status_code == 200:
                st.success("✅ API Connected")
            else:
                st.error("❌ API Error")
        except:
            st.error("❌ API Unavailable")
    
    # Main chat interface
    st.header("💬 Chat with Legal Assistant")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if message.get("sources"):
                display_sources(message["sources"])
            
            # Display safety report if available
            if message.get("safety"):
                display_safety_report(message["safety"])
            
            # Display metrics if available
            if message.get("metrics"):
                display_metrics(message["metrics"])
    
    # Chat input
    if prompt := st.chat_input("Ask a legal question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = call_chat_api(prompt, mode, top_k)
            
            if response:
                # Display bot response
                st.markdown(response["answer"])
                
                # Add bot message to session
                bot_message = {
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response.get("sources", []),
                    "safety": response.get("safety", {}),
                    "metrics": response.get("metrics", {})
                }
                st.session_state.messages.append(bot_message)
                
                # Display additional information
                if response.get("sources"):
                    display_sources(response["sources"])
                
                if response.get("safety"):
                    display_safety_report(response["safety"])
                
                if response.get("metrics"):
                    display_metrics(response["metrics"])
            else:
                st.error("Failed to get response from the API")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p><strong>⚠️ Disclaimer:</strong> This chatbot provides educational information only and does not constitute legal advice. 
            Always consult with qualified legal professionals for specific legal matters.</p>
            <p>Built with ❤️ for the legal community</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

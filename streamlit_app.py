import streamlit as st
import requests
import json
import os

BACKEND_URL="http://localhost:8000"

st.title("🔬 AI Research Agent")
st.markdown("Get comprehensive research reports on any topic!")

st.sidebar.write(f"🌏 Backend: {BACKEND_URL}")

# Input form
query = st.text_input("What would you like to research?", 
                     placeholder="e.g., Latest developments in AI")

if st.button("Start Research", type="primary"):
    if query:
        with st.spinner("Conducting research... This may take 1-2 minutes"):
            try:
                response = requests.post(
                    "http://localhost:8000/research",
                    json={"query": query},
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result["success"]:
                        st.success("Research completed!")
                        
                        # Display report
                        st.markdown("## 📄 Research Report")
                        st.markdown(result["report"])
                        
                        # Display citations
                        if result["citations"]:
                            st.markdown("## 📚 Sources")
                            for citation in result["citations"]:
                                st.markdown(f"- {citation}")
                        
                        # Display metrics
                        if result["quality_metrics"]:
                            st.markdown("## 📊 Quality Metrics")
                            metrics = result["quality_metrics"]
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Completeness", f"{metrics['completeness']:.1%}")
                            with col2:
                                st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                            with col3:
                                st.metric("Relevance", f"{metrics['relevance']:.1%}")
                    else:
                        st.error("Research failed. Please try again.")
                else:
                    st.error(f"API Error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a research query.")

# Run with: streamlit run streamlit_app.py

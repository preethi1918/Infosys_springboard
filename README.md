Swiggy Review Intelligence Platform

The Swiggy Review Intelligence Platform is an AI-driven analytics solution designed to transform unstructured customer feedback into actionable business intelligence. By integrating semantic search with Retrieval-Augmented Generation (RAG), the platform enables stakeholders to identify operational bottlenecks, sentiment trends, and service failures through a conversational interface and an interactive dashboard.

🔑 Key Features

Review Preprocessing: Automated cleaning and normalization of raw customer feedback.

Sentiment and Aspect Tagging: High-precision classification of reviews by emotional tone and service categories (e.g., delivery speed, food quality).

Semantic Search: FAISS-powered vector database for contextual retrieval beyond keyword matches.

RAG-based AI Assistant: Natural language interface for querying complex patterns like refund issues or support quality.

Interactive Dashboard: Streamlit interface for visualizing analytics and granular reports.

🏗️ System Architecture

The system follows a modular architecture to separate:

Data ingestion

Vector storage

Application layer

This separation ensures scalability, maintainability, and ease of deployment.

📁 File Structure
app.py                  # Main entry point for Streamlit dashboard
analytics.py            # Logic for visual insights and trend data
rag_chain.py            # LLM orchestration & RAG pipeline
retriever.py            # Functions for querying vector store
faiss_store/            # Local vector database (index.faiss, metadata.pkl)
pages/                  # Multi-page Streamlit structure
2_AI_Assistant.py       # Interface for RAG chatbot
4_Reports_Alerts.py     # Automated reporting and alerts
utils/                  # Helper scripts (analytics_utils.py, rag_utils.py, retriever_utils.py)
🛠️ Technical Specifications
Component	Technology
Language	Python
Web Framework	Streamlit
Vector Database	FAISS
Orchestration	LangChain
Embeddings	HuggingFace
LLM API	Groq (Llama 3 / Mixtral)
⚙️ Setup and Installation
1. Repository Initialization
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
2. Virtual Environment Configuration

Windows:

python -m venv venv
venv\Scripts\activate

Mac/Linux:

python -m venv venv
source venv/bin/activate
3. Dependency Installation
pip install -r requirements.txt
4. Environment Variables

Configure your API keys for AI functionalities.

Windows:

set GROQ_API_KEY=your_api_key_here

Mac/Linux:

export GROQ_API_KEY=your_api_key_here
5. Vector Store Initialization

Ensure faiss_store/ contains required index files.
If missing, build them using:

python build_faiss.py
6. Execution
streamlit run app.py
🚀 Deployment Guidelines

Cloud Hosting: Use CPU-optimized builds (e.g., Torch) to comply with Streamlit Cloud limits.

API Security: Do not commit .env files or hardcoded keys.

Data Management: Keep FAISS index files under 100 MB for faster loading.

Path Management: Use relative paths to ensure cross-platform compatibility.

💡 Usage

Open the Streamlit dashboard via app.py

Explore interactive reports and visualizations

Query the RAG-based AI assistant for customer review insights

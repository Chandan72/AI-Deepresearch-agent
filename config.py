# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/research_db")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # LLM Configuration
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    
    # Research Configuration
    MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "10"))
    MAX_RESEARCH_ITERATIONS = int(os.getenv("MAX_RESEARCH_ITERATIONS", "5"))
    QUALITY_THRESHOLD = float(os.getenv("QUALITY_THRESHOLD", "0.7"))
    
    # Deployment
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

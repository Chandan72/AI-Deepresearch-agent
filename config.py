
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    # Database Configuration
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/research_db")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # LLM Configuration
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-pro")
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    
    # Research Configuration
    MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    MAX_RESEARCH_ITERATIONS = int(os.getenv("MAX_RESEARCH_ITERATIONS", "2"))
    QUALITY_THRESHOLD = float(os.getenv("QUALITY_THRESHOLD", "0.4"))
    MIN_SOURCES_REQUIRED=int(os.getenv("MIN_SOURCES_REQUIRED","3"))
    REQUEST_TIMEOUT=int(os.getenv("REQUEST_TIMEOUT", "60"))
    MAX_CONTENT_LENGTH= int(os.getenv("MAX_CONTENT_LENGTH", "5000"))
    
    # Deployment
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    # config.py - Complete implementation with validation

    # **MISSING METHOD** - Add this validation function
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is present"""
        missing_keys = []
        
        # Check critical API keys
        if not cls.GEMINI_API_KEY:
            missing_keys.append("GEMINI_API_KEY")
        if not cls.TAVILY_API_KEY:
            missing_keys.append("TAVILY_API_KEY")
        
        if missing_keys:
            print(f"\n❌ Missing required environment variables: {', '.join(missing_keys)}")
            print("\n📝 Please add them to your .env file:")
            for key in missing_keys:
                print(f"   {key}=your_actual_key_here")
            print("\n💡 You can get API keys from:")
            print("   • OpenAI: https://platform.openai.com/api-keys")
            print("   • Tavily: https://tavily.com/")
            print("\n🔧 Create a .env file in your project root with these keys.")
            return False
        
        # Validation passed
        print("✅ Configuration validation passed")
        print(f"   • GEMINI API Key: {'✅ Set' if cls.GEMINI_API_KEY else '❌ Missing'}")
        print(f"   • Tavily API Key: {'✅ Set' if cls.TAVILY_API_KEY else '❌ Missing'}")
        print(f"   • Model: {cls.MODEL_NAME}")
        print(f"   • Debug Mode: {cls.DEBUG}")
        return True
    
    @classmethod
    def get_config_summary(cls) -> dict:
        """Get a summary of current configuration"""
        return {
            "model": cls.MODEL_NAME,
            "has_openai_key": bool(cls.GEMINI_API_KEY),
            "has_tavily_key": bool(cls.TAVILY_API_KEY),
            "max_iterations": cls.MAX_RESEARCH_ITERATIONS,
            "max_search_results": cls.MAX_SEARCH_RESULTS,
            "debug": cls.DEBUG,
            "host": cls.HOST,
            "port": cls.PORT
        }


"""
Configuration module for CSV Q&A Agent
Handles environment variables and application settings
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    """Application configuration settings"""
    
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY', '')
    
    # File Processing
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '100'))
    DEFAULT_ENCODING = os.getenv('DEFAULT_ENCODING', 'utf-8')
    SUPPORTED_EXTENSIONS = ['.csv', '.zip']
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / 'data'
    LOGS_DIR = PROJECT_ROOT / 'logs'
    TEMPLATES_DIR = PROJECT_ROOT / 'templates'
    
    # Streamlit
    STREAMLIT_SERVER_PORT = int(os.getenv('STREAMLIT_SERVER_PORT', '8501'))
    STREAMLIT_SERVER_ADDRESS = os.getenv('STREAMLIT_SERVER_ADDRESS', 'localhost')
    
    # Agent Settings
    MAX_RETRIES = 3
    TIMEOUT_SECONDS = 30
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present"""
        required_keys = ['OPENAI_API_KEY']
        missing_keys = [key for key in required_keys if not getattr(cls, key)]
        
        if missing_keys:
            print(f"Warning: Missing required configuration: {', '.join(missing_keys)}")
            return False
        return True
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        for directory in [cls.DATA_DIR, cls.LOGS_DIR, cls.TEMPLATES_DIR]:
            directory.mkdir(exist_ok=True)

# Create directories on import
Config.create_directories() 
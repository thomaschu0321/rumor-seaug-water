"""
Configuration File - All hyperparameters and path settings
"""
import torch
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class"""
    
    # ========== Project Paths ==========
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    # Project data storage paths
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
    
    # ========== Data Configuration ==========
    FEATURE_DIM = 768  # BERT feature dimension
    NUM_CLASSES = 2     # Binary classification
    PHEME_EVENT_LIMIT = 1  # Limit of PHEME events to load (None for all)
    
    # ========== Model Configuration ==========
    HIDDEN_DIM = 32       # GNN hidden layer dimension
    NUM_GNN_LAYERS = 2    # Number of GNN layers
    DROPOUT = 0.7         # Dropout rate
    GAT_HEADS = 4         # Number of attention heads for GAT
    
    # ========== Training Configuration ==========
    BATCH_SIZE = 32     # Batch size
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-3  # L2 regularization
    NUM_EPOCHS = 100    # Maximum training epochs
    PATIENCE = 5        # Early stopping patience
    
    # ========== Runtime Configuration ==========
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 0     # DataLoader worker processes
    SEED = 42           # Random seed
    
    # ========== Data Split ==========
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # ========== Output Configuration ==========
    SAVE_DIR = os.path.join(PROJECT_ROOT, 'results')
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.PROCESSED_DIR, exist_ok=True)
        os.makedirs(cls.SAVE_DIR, exist_ok=True)
        print("Project directories created")
    
    # ========== LLM Configuration ==========
    # LLM Provider: 'azure' or 'deepseek'
    LLM_PROVIDER = os.environ.get('LLM_PROVIDER', 'deepseek').lower()
    
    # Azure OpenAI Configuration (CUHK-style)
    AZURE_API_KEY = os.environ.get('AZURE_API_KEY')
    AZURE_ENDPOINT = os.environ.get('AZURE_ENDPOINT', 'https://cuhk-apip.azure-api.net')
    AZURE_MODEL = os.environ.get('AZURE_MODEL', 'gpt-4o-mini')
    API_VERSION = os.environ.get('API_VERSION', '2023-05-15')
    
    # DeepSeek API Configuration
    DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
    DEEPSEEK_MODEL = os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat')
    DEEPSEEK_BASE_URL = os.environ.get('DEEPSEEK_BASE_URL', 'https://www.chataiapi.com')
    
    # LLM Parameters
    LLM_MAX_TOKENS = int(os.environ.get('LLM_MAX_TOKENS', '500'))
    LLM_TEMPERATURE = float(os.environ.get('LLM_TEMPERATURE', '0.7'))
    LLM_BATCH_SIZE = int(os.environ.get('LLM_BATCH_SIZE', '50'))
    
    # Network/Timeout Configuration
    LLM_TIMEOUT = int(os.environ.get('LLM_TIMEOUT', '120'))
    LLM_MAX_RETRIES = int(os.environ.get('LLM_MAX_RETRIES', '3'))
    LLM_RETRY_DELAY = int(os.environ.get('LLM_RETRY_DELAY', '5'))
    
    # Enable/Disable LLM
    USE_LLM = os.environ.get('USE_LLM', 'false').lower() == 'true'

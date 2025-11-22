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
    
    # Data path configuration (multiple methods by priority)
    # 1. Read from environment variable (highest priority, convenient for changing computers)
    # 2. From current project's data/raw directory
    # 3. From original BiGCN project (only when on the same computer)
    
    _env_data_dir = os.environ.get('RUMOR_DATA_DIR')  # Can specify via environment variable
    
    if _env_data_dir and os.path.exists(_env_data_dir):
        # Option 1: Use path specified by environment variable
        BIGCN_DATA_DIR = _env_data_dir
        print(f"✓ Using data path from environment variable: {BIGCN_DATA_DIR}")
    elif os.path.exists(os.path.join(PROJECT_ROOT, 'data', 'raw')):
        # Option 2: Use project's data/raw directory (data has been copied)
        BIGCN_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
        print(f"✓ Using data from project: {BIGCN_DATA_DIR}")
    else:
        # Option 3: Default to original BiGCN project (same computer)
        BIGCN_DATA_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), 'BiGCN-master', 'data')
        if not os.path.exists(BIGCN_DATA_DIR):
            print(f"⚠️  Warning: Data path does not exist: {BIGCN_DATA_DIR}")
            print(f"   Please set environment variable or copy data (run python setup_data.py)")
    
    # Project data storage paths
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
    
    # ========== Data Configuration ==========
    # Use full data (no sampling)
    SAMPLE_RATIO = 1.0  # Use 100% of data
    
    # Feature configuration
    FEATURE_DIM = 768  # BERT feature dimension (always use BERT)
    NUM_CLASSES = 2     # 2 classes: non-rumor(0) vs rumor(1)
    
    # ========== Model Configuration ==========
    GNN_BACKBONE = 'gcn'  # GNN backbone: 'gcn' or 'gat'
    HIDDEN_DIM = 32       # GNN hidden layer dimension
    NUM_GNN_LAYERS = 2    # Number of GNN layers
    DROPOUT = 0.7         # Dropout rate (increased to reduce overfitting)
    GAT_HEADS = 4         # Number of attention heads for GAT
    
    # ========== XGBoost Adaptive Classifier Configuration ==========
    USE_XGBOOST = True  # Whether to use XGBoost adaptive classifier
    XGBOOST_N_ESTIMATORS = 100  # Number of trees
    XGBOOST_MAX_DEPTH = 6       # Maximum tree depth
    XGBOOST_LEARNING_RATE = 0.1 # Learning rate (shrinkage)
    XGBOOST_SUBSAMPLE = 0.8     # Sample ratio for each tree
    XGBOOST_COLSAMPLE = 0.8     # Feature ratio for each tree
    XGBOOST_GAMMA = 0           # Minimum loss reduction for split
    XGBOOST_REG_ALPHA = 0       # L1 regularization
    XGBOOST_REG_LAMBDA = 1      # L2 regularization
    XGBOOST_CONFIDENCE_THRESHOLD = 0.6  # Low confidence threshold
    
    # ========== Training Configuration ==========
    BATCH_SIZE = 32     # Batch size (can be larger with full data)
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-3  # L2 regularization (increased to reduce overfitting)
    NUM_EPOCHS = 100    # Maximum training epochs (full data needs more epochs)
    
    # Early stopping
    PATIENCE = 5       # Early stopping patience (reduced to stop sooner)
    
    # ========== Runtime Configuration ==========
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 0     # DataLoader worker processes
    SEED = 42           # Random seed
    
    # ========== Data Split ==========
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # ========== Output Configuration ==========
    SAVE_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
    LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
    PRINT_FREQ = 10     # Print every N batches
    
    @classmethod
    def display(cls):
        """Display current configuration"""
        print("=" * 50)
        print("Current Configuration:")
        print("=" * 50)
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value):
                print(f"{key:20s}: {value}")
        print("=" * 50)
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories"""
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.PROCESSED_DIR, exist_ok=True)
        os.makedirs(cls.SAVE_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        print("✓ Project directories created")
    
    # ========== LLM Configuration ==========
    # Azure OpenAI Configuration (CUHK-style)
    AZURE_API_KEY = os.environ.get('AZURE_API_KEY')
    AZURE_ENDPOINT = os.environ.get('AZURE_ENDPOINT', 'https://cuhk-apip.azure-api.net')
    AZURE_MODEL = os.environ.get('AZURE_MODEL', 'gpt-4o-mini')
    API_VERSION = os.environ.get('API_VERSION', '2023-05-15')  # CUHK uses 2023-05-15
    
    # LLM Parameters
    LLM_MAX_TOKENS = int(os.environ.get('LLM_MAX_TOKENS', '500'))
    LLM_TEMPERATURE = float(os.environ.get('LLM_TEMPERATURE', '0.7'))
    LLM_AUGMENTATION_FACTOR = int(os.environ.get('LLM_AUGMENTATION_FACTOR', '5'))
    LLM_BATCH_SIZE = int(os.environ.get('LLM_BATCH_SIZE', '10'))
    
    # Cost Control
    LLM_MAX_SAMPLES = int(os.environ.get('LLM_MAX_SAMPLES', '50'))
    LLM_ENABLE_CACHE = os.environ.get('LLM_ENABLE_CACHE', 'true').lower() == 'true'
    LLM_CACHE_PATH = os.path.join(PROJECT_ROOT, 'data', 'llm_cache.pkl')
    
    # Enable/Disable LLM
    USE_LLM = os.environ.get('USE_LLM', 'false').lower() == 'true'

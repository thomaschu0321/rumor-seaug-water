"""
Node-Level Augmentation Module for SeAug Framework (Stage 3)

This module implements node-level LLM augmentation and LM encoding.
Unlike graph-level augmentation, this enhances individual nodes' features.

Pipeline:
1. Extract text from selected nodes
2. Use LLM to generate enhanced/paraphrased versions
3. Use Language Model (BERT/RoBERTa) to encode enhanced text
4. Attach augmented features to nodes
"""

import os
import time
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import Config

# Import transformers for LM encoding (required)
from transformers import AutoTokenizer, AutoModel

# Import LLM client
try:
    from openai import AzureOpenAI, OpenAI
    from openai import APITimeoutError, APIConnectionError, APIError
    try:
        import httpx
        HTTPX_AVAILABLE = True
    except ImportError:
        HTTPX_AVAILABLE = False
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    HTTPX_AVAILABLE = False
    print("⚠️  Warning: openai package not installed. LLM augmentation disabled.")


class LanguageModelEncoder:
    """
    Encode text using pre-trained language models (BERT, RoBERTa, etc.)
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = None
    ):
        """
        Initialize Language Model Encoder
        
        Args:
            model_name: HuggingFace model name
                - "sentence-transformers/all-MiniLM-L6-v2": Fast, 384-dim
                - "bert-base-uncased": Standard BERT, 768-dim
                - "roberta-base": RoBERTa, 768-dim
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading Language Model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Get embedding dimension
        with torch.no_grad():
            dummy_input = self.tokenizer("test", return_tensors="pt", 
                                        padding=True, truncation=True)
            dummy_input = {k: v.to(self.device) for k, v in dummy_input.items()}
            dummy_output = self.model(**dummy_input)
            self.embedding_dim = dummy_output.last_hidden_state.shape[-1]
        
        print(f"✓ Language Model loaded successfully")
        print(f"  Device: {self.device}")
        print(f"  Embedding dimension: {self.embedding_dim}")
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode texts to embeddings
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
        
        Returns:
            embeddings: numpy array [num_texts, embedding_dim]
        """
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Encode
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding or mean pooling
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    batch_embeddings = outputs.pooler_output
                else:
                    # Mean pooling
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            
            embeddings.append(batch_embeddings.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text"""
        return self.encode([text])[0]


class NodeAugmentor:
    """
    Node-level LLM augmentation and LM encoding
    """
    
    def __init__(
        self,
        lm_encoder: LanguageModelEncoder = None,
        use_llm: bool = None,
        api_key: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        batch_size: int = None
    ):
        """
        Initialize Node Augmentor
        
        Args:
            lm_encoder: Language Model encoder instance
            use_llm: Whether to use LLM for augmentation (defaults to Config.USE_LLM)
            api_key: API key (optional, defaults to Config.AZURE_API_KEY or Config.DEEPSEEK_API_KEY based on LLM_PROVIDER)
            model: Model name (optional, defaults to Config.AZURE_MODEL or Config.DEEPSEEK_MODEL based on LLM_PROVIDER)
            temperature: Generation temperature (defaults to Config.LLM_TEMPERATURE)
            max_tokens: Max tokens per generation (defaults to Config.LLM_MAX_TOKENS)
            batch_size: Number of nodes per API call (defaults to Config.LLM_BATCH_SIZE)
        
        Note: Supports both Azure OpenAI and DeepSeek providers. Set LLM_PROVIDER in config to switch.
        """
        # Initialize LM encoder
        if lm_encoder is None:
            self.lm_encoder = LanguageModelEncoder()
        else:
            self.lm_encoder = lm_encoder
        
        # Use Config defaults if not provided
        self.use_llm = (use_llm if use_llm is not None else Config.USE_LLM) and LLM_AVAILABLE
        self.temperature = temperature if temperature is not None else Config.LLM_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else Config.LLM_MAX_TOKENS
        self.batch_size = batch_size if batch_size is not None else Config.LLM_BATCH_SIZE
        
        # Initialize LLM client
        if self.use_llm:
            self._init_llm_client(api_key, model)
        
        # Cache
        self.cache = {}
        self.stats = {
            'nodes_augmented': 0,
            'llm_calls': 0,
            'cache_hits': 0,
            'quota_exceeded': 0,
            'rate_limited': 0,
            'network_errors': 0,
            'retries': 0
        }
    
    def _init_llm_client(self, api_key: str = None, model: str = None):
        """Initialize LLM client - supports Azure OpenAI and DeepSeek"""
        provider = None
        try:
            provider = Config.LLM_PROVIDER.lower()
            
            if provider == 'deepseek':
                # DeepSeek API (OpenAI-compatible)
                if not Config.DEEPSEEK_API_KEY:
                    raise ValueError("DEEPSEEK_API_KEY is not set. Please configure it in your .env file.")
                
                self.provider = 'deepseek'
                self.api_key = api_key or Config.DEEPSEEK_API_KEY
                self.model = model or Config.DEEPSEEK_MODEL
                
                # Ensure base URL ends with /v1 for OpenAI-compatible APIs
                base_url = Config.DEEPSEEK_BASE_URL.rstrip('/')
                if not base_url.endswith('/v1'):
                    base_url = f"{base_url}/v1"
                self.base_url = base_url
                
                # Use OpenAI SDK (DeepSeek is OpenAI-compatible)
                from openai import OpenAI
                # Configure timeout to prevent hanging (connect_timeout, read_timeout)
                timeout = (10.0, Config.LLM_TIMEOUT)
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=timeout
                )
                print(f"✓ LLM Client initialized (DeepSeek)")
                print(f"  Model: {self.model}")
                print(f"  Base URL: {self.base_url}")
                
            elif provider == 'azure':
                # Azure OpenAI API
                if not Config.AZURE_API_KEY:
                    raise ValueError("AZURE_API_KEY is not set. Please configure it in your .env file.")
                
                self.provider = 'azure'
                self.use_apim = True
                self.api_key = api_key or Config.AZURE_API_KEY
                self.endpoint = Config.AZURE_ENDPOINT
                self.model = model or Config.AZURE_MODEL
                self.api_version = Config.API_VERSION
                
                # Use AzureOpenAI SDK
                from openai import AzureOpenAI
                # Configure timeout to prevent hanging (connect_timeout, read_timeout)
                timeout = (10.0, Config.LLM_TIMEOUT)
                self.client = AzureOpenAI(
                    azure_endpoint=self.endpoint,
                    api_version=self.api_version,
                    api_key=self.api_key,
                    timeout=timeout
                )
                print(f"✓ LLM Client initialized (Azure OpenAI)")
                print(f"  Model: {self.model}")
                print(f"  Endpoint: {self.endpoint}")
            else:
                raise ValueError(f"Unknown LLM provider: {provider}. Use 'azure' or 'deepseek'.")
            
        except Exception as e:
            print(f"⚠️  Error initializing LLM: {e}")
            if provider == 'deepseek':
                print(f"   Make sure DEEPSEEK_API_KEY is set in your .env file")
            elif provider == 'azure':
                print(f"   Make sure AZURE_API_KEY is set in your .env file")
            else:
                print(f"   Make sure LLM_PROVIDER is set to 'azure' or 'deepseek' in your .env file")
            self.use_llm = False
    
    def _call_llm(self, prompt: str, retry_count: int = 0) -> str:
        """
        Call LLM API with retry logic for network/SSL errors
        
        Args:
            prompt: The prompt to send to the LLM
            retry_count: Current retry attempt (internal use)
        """
        if not self.use_llm:
            return None
        
        # Check cache first
        cache_key = hash(prompt)
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]
        
        try:
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for text paraphrasing."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Check if response is valid
            if isinstance(response, str):
                raise ValueError(f"API returned string instead of object: {response[:200]}")
            
            if not hasattr(response, 'choices'):
                # Try to get more info about the response
                response_str = str(response)
                if len(response_str) > 500:
                    response_str = response_str[:500] + "..."
                raise ValueError(f"Response missing 'choices' attribute. Type: {type(response)}, Response: {response_str}")
            
            if not response.choices or len(response.choices) == 0:
                raise ValueError("Response has no choices")
            
            text = response.choices[0].message.content.strip()
            
            # Cache result
            self.cache[cache_key] = text
            self.stats['llm_calls'] += 1
            
            return text
            
        except (APITimeoutError, APIConnectionError) as e:
            # Network/SSL/Connection errors from OpenAI SDK - retry with exponential backoff
            self.stats['network_errors'] += 1
            
            if retry_count < Config.LLM_MAX_RETRIES:
                delay = Config.LLM_RETRY_DELAY * (2 ** retry_count)  # Exponential backoff
                self.stats['retries'] += 1
                print(f"⚠️  Network/SSL error (attempt {retry_count + 1}/{Config.LLM_MAX_RETRIES}): {type(e).__name__}")
                print(f"   → Retrying in {delay} seconds...")
                time.sleep(delay)
                return self._call_llm(prompt, retry_count + 1)
            else:
                print(f"⚠️  Network/SSL error after {Config.LLM_MAX_RETRIES} retries: {e}")
                print(f"   → Skipping this request")
                return None
                
        except (ConnectionError, OSError) as e:
            # General connection/SSL errors - retry with exponential backoff
            self.stats['network_errors'] += 1
            
            if retry_count < Config.LLM_MAX_RETRIES:
                delay = Config.LLM_RETRY_DELAY * (2 ** retry_count)
                self.stats['retries'] += 1
                print(f"⚠️  Connection error (attempt {retry_count + 1}/{Config.LLM_MAX_RETRIES}): {type(e).__name__}")
                print(f"   → Retrying in {delay} seconds...")
                time.sleep(delay)
                return self._call_llm(prompt, retry_count + 1)
            else:
                print(f"⚠️  Connection error after {Config.LLM_MAX_RETRIES} retries: {e}")
                print(f"   → Skipping this request")
                return None
            
        except APIError as e:
            error_str = str(e)
            
            # Handle quota exceeded (403)
            if "403" in error_str or "quota" in error_str.lower():
                print(f"⚠️  Quota exceeded: {e}")
                self.stats['quota_exceeded'] += 1
                # Disable LLM for remaining calls
                print("   → Disabling LLM for remaining nodes")
                self.use_llm = False
                return None
            
            # Handle rate limit (429)
            elif "429" in error_str or "rate limit" in error_str.lower():
                print(f"⚠️  Rate limit hit: {e}")
                self.stats['rate_limited'] += 1
                if retry_count < Config.LLM_MAX_RETRIES:
                    delay = 60  # Wait 1 minute for rate limits
                    self.stats['retries'] += 1
                    print(f"   → Retrying in {delay} seconds...")
                    time.sleep(delay)
                    return self._call_llm(prompt, retry_count + 1)
                else:
                    print(f"   → Max retries reached, skipping")
                    return None
            
            else:
                print(f"⚠️  LLM API error: {e}")
                return None
                
        except Exception as e:
            # Catch-all for other errors (including httpx timeouts if available)
            error_str = str(e)
            error_type = type(e).__name__
            
            # Check if it's an httpx timeout error (if httpx is available)
            if HTTPX_AVAILABLE and isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException)):
                self.stats['network_errors'] += 1
                if retry_count < Config.LLM_MAX_RETRIES:
                    delay = Config.LLM_RETRY_DELAY * (2 ** retry_count)
                    self.stats['retries'] += 1
                    print(f"⚠️  Timeout error (attempt {retry_count + 1}/{Config.LLM_MAX_RETRIES}): {error_type}")
                    print(f"   → Retrying in {delay} seconds...")
                    time.sleep(delay)
                    return self._call_llm(prompt, retry_count + 1)
                else:
                    print(f"⚠️  Timeout error after {Config.LLM_MAX_RETRIES} retries: {e}")
                    return None
            
            # Check if it's a network-related error by error message
            elif any(keyword in error_str.lower() for keyword in ['ssl', 'socket', 'connection', 'timeout', 'network', 'read']):
                self.stats['network_errors'] += 1
                if retry_count < Config.LLM_MAX_RETRIES:
                    delay = Config.LLM_RETRY_DELAY * (2 ** retry_count)
                    self.stats['retries'] += 1
                    print(f"⚠️  Network-related error (attempt {retry_count + 1}/{Config.LLM_MAX_RETRIES}): {error_type}")
                    print(f"   → Retrying in {delay} seconds...")
                    time.sleep(delay)
                    return self._call_llm(prompt, retry_count + 1)
                else:
                    print(f"⚠️  Network error after {Config.LLM_MAX_RETRIES} retries: {error_type}: {e}")
                    return None
            else:
                print(f"⚠️  LLM call failed: {error_type}: {e}")
                return None
    
    def augment_node_text(self, text: str) -> str:
        """
        Augment a single node's text using LLM
        
        Args:
            text: Original text
        
        Returns:
            Augmented text (or original if augmentation fails)
        """
        if not self.use_llm or not text or len(text) < 5:
            return text
        
        prompt = f"""Paraphrase the following social media post while keeping the same meaning:

Original: "{text}"

Paraphrased version (one line only):"""
        
        augmented = self._call_llm(prompt)
        return augmented if augmented else text
    
    def augment_batch_texts(self, texts: List[str], batch_size: int = None) -> List[str]:
        """
        Augment multiple texts in batched API calls (TOKEN EFFICIENT!)
        
        This method batches multiple nodes together in single API calls,
        significantly reducing token usage by sharing system prompt and
        instructions across multiple texts.
        
        Token savings example:
        - Individual calls (20 nodes): ~2,600 tokens, 20 API calls
        - Batched call (20 nodes): ~1,080 tokens, 1 API call
        - Savings: 58% tokens, 95% fewer API calls!
        
        Args:
            texts: List of texts to augment
            batch_size: Number of texts per API call (defaults to self.batch_size, recommended: 10-20)
        
        Returns:
            List of augmented texts (same length as input)
        """
        if batch_size is None:
            batch_size = self.batch_size
        if not self.use_llm or not texts:
            return texts
        
        augmented_results = []
        
        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Filter out empty/short texts
            valid_indices = [j for j, t in enumerate(batch_texts) if t and len(t) >= 5]
            valid_texts = [batch_texts[j] for j in valid_indices]
            
            if not valid_texts:
                # All texts invalid, keep originals
                augmented_results.extend(batch_texts)
                continue
            
            # Create batched prompt
            batch_prompt = self._create_batch_prompt(valid_texts)
            
            # Call LLM once for entire batch
            response = self._call_llm(batch_prompt)
            
            if response:
                # Parse response into individual texts
                parsed_texts = self._parse_batch_response(response, len(valid_texts))
                
                # Map back to original order
                result_batch = []
                valid_idx = 0
                for j in range(len(batch_texts)):
                    if j in valid_indices:
                        # Use augmented text if parsing succeeded
                        if valid_idx < len(parsed_texts) and parsed_texts[valid_idx]:
                            result_batch.append(parsed_texts[valid_idx])
                        else:
                            result_batch.append(batch_texts[j])  # Fallback to original
                        valid_idx += 1
                    else:
                        result_batch.append(batch_texts[j])
                
                augmented_results.extend(result_batch)
            else:
                # API call failed, keep originals
                augmented_results.extend(batch_texts)
        
        return augmented_results
    
    def _create_batch_prompt(self, texts: List[str]) -> str:
        """
        Create a batched prompt for multiple texts
        
        Args:
            texts: List of texts to augment
        
        Returns:
            Formatted prompt string
        """
        # Create numbered list of texts
        text_list = "\n".join([f"{i+1}. \"{text}\"" for i, text in enumerate(texts)])
        
        prompt = f"""Paraphrase the following {len(texts)} social media posts while keeping the same meaning for each.
Provide one paraphrased version per line, in the same order.

Original posts:
{text_list}

Paraphrased versions (one per line, no numbering):"""
        
        return prompt
    
    def _parse_batch_response(self, response: str, expected_count: int) -> List[str]:
        """
        Parse batched LLM response into individual texts
        
        Args:
            response: LLM response containing multiple paraphrased texts
            expected_count: Expected number of texts
        
        Returns:
            List of parsed texts
        """
        import re
        
        # Split by newlines and clean up
        lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
        
        # Remove numbering if present (e.g., "1.", "1)", etc.)
        cleaned_lines = []
        for line in lines:
            # Remove leading numbers and punctuation
            cleaned = re.sub(r'^\d+[\.\):\-]\s*', '', line)
            # Remove quotes if present
            cleaned = cleaned.strip('"\'')
            if cleaned:
                cleaned_lines.append(cleaned)
        
        # Return expected number of results (pad with empty if needed)
        while len(cleaned_lines) < expected_count:
            cleaned_lines.append("")
        
        return cleaned_lines[:expected_count]
    
    def augment_graph_nodes(
        self,
        data: Data,
        selected_node_indices: np.ndarray,
        node_texts: List[str] = None,
        use_batching: bool = True,
        batch_size: int = None
    ) -> Data:
        """
        Augment selected nodes in a graph
        
        Args:
            data: Original PyG Data object
            selected_node_indices: Indices of nodes to augment
            node_texts: Original texts for each node (if available)
            use_batching: If True, batch multiple nodes in single API calls (RECOMMENDED)
            batch_size: Number of nodes per API call (defaults to self.batch_size, recommended: 10-20)
        
        Returns:
            Augmented Data object with x_aug field
        """
        if batch_size is None:
            batch_size = self.batch_size
        num_nodes = data.x.shape[0]
        
        # If no texts provided, create dummy texts
        if node_texts is None:
            node_texts = [f"Node {i} placeholder text" for i in range(num_nodes)]
        
        # Augment selected nodes using BATCHED approach (token efficient!)
        if use_batching and self.use_llm and len(selected_node_indices) > 0:
            # Collect texts for selected nodes only
            selected_texts = [node_texts[i] for i in selected_node_indices]
            
            # Batch augment selected texts (saves tokens!)
            augmented_selected = self.augment_batch_texts(selected_texts, batch_size=batch_size)
            
            # Create final text list with augmented texts in correct positions
            augmented_texts = []
            selected_idx = 0
            for i in range(num_nodes):
                if i in selected_node_indices:
                    augmented_texts.append(augmented_selected[selected_idx])
                    selected_idx += 1
                    self.stats['nodes_augmented'] += 1
                else:
                    augmented_texts.append(node_texts[i])
        else:
            # Fallback to individual augmentation (old method, less efficient)
            augmented_texts = []
            for i in range(num_nodes):
                if i in selected_node_indices:
                    # Augment this node individually
                    original_text = node_texts[i]
                    augmented_text = self.augment_node_text(original_text)
                    augmented_texts.append(augmented_text)
                    self.stats['nodes_augmented'] += 1
                else:
                    # Keep original
                    augmented_texts.append(node_texts[i])
        
        # Encode all texts to embeddings
        embeddings = self.lm_encoder.encode(augmented_texts)
        
        # Create augmented data
        data_aug = data.clone()
        data_aug.x_aug = torch.FloatTensor(embeddings)
        data_aug.augmented_node_mask = torch.zeros(num_nodes, dtype=torch.bool)
        # Convert to list to avoid negative stride issues with numpy arrays
        if isinstance(selected_node_indices, np.ndarray):
            selected_node_indices = selected_node_indices.copy()
        data_aug.augmented_node_mask[selected_node_indices] = True
        
        return data_aug
    
    def augment_batch(
        self,
        graph_list: List[Data],
        selected_nodes_list: List[np.ndarray],
        texts_list: List[List[str]] = None,
        verbose: bool = True,
        batch_size: int = None
    ) -> List[Data]:
        """
        Augment a batch of graphs with batched API calls
        
        Args:
            graph_list: List of PyG Data objects
            selected_nodes_list: List of selected node indices for each graph
            texts_list: List of node texts for each graph
            verbose: Show progress bar
            batch_size: Number of nodes per API call (defaults to self.batch_size)
        
        Returns:
            List of augmented Data objects
        """
        if batch_size is None:
            batch_size = self.batch_size
        augmented_graphs = []
        
        iterator = tqdm(zip(graph_list, selected_nodes_list), 
                       total=len(graph_list),
                       desc="Augmenting nodes (batched)",
                       disable=not verbose)
        
        for i, (data, selected_indices) in enumerate(iterator):
            node_texts = texts_list[i] if texts_list else None
            data_aug = self.augment_graph_nodes(
                data, 
                selected_indices, 
                node_texts,
                use_batching=True,
                batch_size=batch_size
            )
            augmented_graphs.append(data_aug)
        
        return augmented_graphs
    
    def get_statistics(self) -> Dict:
        """Get augmentation statistics"""
        return self.stats.copy()

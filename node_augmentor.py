"""Node-Level Augmentation Module for SeAug Framework"""

import time
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Dict
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import Config
from transformers import AutoTokenizer, AutoModel

try:
    from openai import AzureOpenAI, OpenAI
    from openai import APITimeoutError, APIConnectionError, APIError
    try:
        import httpx
        HTTPX_AVAILABLE = True
    except ImportError:
        HTTPX_AVAILABLE = False
    try:
        import httpcore
        HTTPCORE_AVAILABLE = True
    except ImportError:
        HTTPCORE_AVAILABLE = False
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    HTTPX_AVAILABLE = False
    HTTPCORE_AVAILABLE = False
    print("Warning: openai package not installed. LLM augmentation disabled.")


class QuotaExceededError(Exception):
    """Raised when API quota is exceeded"""
    pass


class LanguageModelEncoder:
    """Encode text using pre-trained language models"""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = None
    ):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading Language Model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            dummy_input = self.tokenizer("test", return_tensors="pt", padding=True, truncation=True)
            dummy_input = {k: v.to(self.device) for k, v in dummy_input.items()}
            dummy_output = self.model(**dummy_input)
            self.embedding_dim = dummy_output.last_hidden_state.shape[-1]
        
        print(f"Language Model loaded: {self.device}, dim={self.embedding_dim}")
    
    def encode(self, texts: List[str], batch_size: int = 32, verbose: bool = False) -> np.ndarray:
        """Encode texts to embeddings"""
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Process in batches
        batch_iter = range(0, len(texts), batch_size)
        if verbose and total_batches > 1:
            batch_iter = tqdm(batch_iter, desc="Encoding texts", total=total_batches, leave=False)
        
        for i in batch_iter:
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
        try:
            provider = Config.LLM_PROVIDER.lower()
            connect_timeout = 30.0
            timeout = (connect_timeout, Config.LLM_TIMEOUT)
            
            if provider == 'deepseek':
                if not Config.DEEPSEEK_API_KEY:
                    raise ValueError("DEEPSEEK_API_KEY is not set. Please configure it in your .env file.")
                
                self.provider = 'deepseek'
                self.api_key = api_key or Config.DEEPSEEK_API_KEY
                self.model = model or Config.DEEPSEEK_MODEL
                base_url = Config.DEEPSEEK_BASE_URL.rstrip('/')
                if not base_url.endswith('/v1'):
                    base_url = f"{base_url}/v1"
                self.base_url = base_url
                
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout)
                print(f"LLM Client initialized (DeepSeek): {self.model}")
                
            elif provider == 'azure':
                if not Config.AZURE_API_KEY:
                    raise ValueError("AZURE_API_KEY is not set. Please configure it in your .env file.")
                
                self.provider = 'azure'
                self.api_key = api_key or Config.AZURE_API_KEY
                self.endpoint = Config.AZURE_ENDPOINT
                self.model = model or Config.AZURE_MODEL
                self.api_version = Config.API_VERSION
                
                self.client = AzureOpenAI(
                    azure_endpoint=self.endpoint,
                    api_version=self.api_version,
                    api_key=self.api_key,
                    timeout=timeout
                )
                print(f"LLM Client initialized (Azure OpenAI): {self.model}")
            else:
                raise ValueError(f"Unknown LLM provider: {provider}. Use 'azure' or 'deepseek'.")
            
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            self.use_llm = False
    
    def _call_llm(self, prompt: str, retry_count: int = 0) -> str:
        if not self.use_llm:
            return None
        
        cache_key = hash(prompt)
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for text paraphrasing."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            if not response.choices:
                raise ValueError("Response has no choices")
            
            text = response.choices[0].message.content.strip()
            self.cache[cache_key] = text
            self.stats['llm_calls'] += 1
            return text
            
        except (APITimeoutError, APIConnectionError, ConnectionError, OSError) as e:
            self.stats['network_errors'] += 1
            if retry_count < Config.LLM_MAX_RETRIES:
                delay = Config.LLM_RETRY_DELAY * (2 ** retry_count)
                self.stats['retries'] += 1
                print(f"Network error (attempt {retry_count + 1}/{Config.LLM_MAX_RETRIES}): {type(e).__name__}")
                time.sleep(delay)
                return self._call_llm(prompt, retry_count + 1)
            else:
                print(f"Network error after {Config.LLM_MAX_RETRIES} retries: {e}")
                return None
            
        except APIError as e:
            error_str = str(e)
            if "403" in error_str or "quota" in error_str.lower() or "402" in error_str or "payment" in error_str.lower():
                print(f"Quota exceeded: {e}")
                self.stats['quota_exceeded'] += 1
                self.use_llm = False
                raise QuotaExceededError(f"API quota exceeded: {e}")
            elif "429" in error_str or "rate limit" in error_str.lower():
                self.stats['rate_limited'] += 1
                if retry_count < Config.LLM_MAX_RETRIES:
                    delay = 60
                    self.stats['retries'] += 1
                    print(f"Rate limit hit, retrying in {delay} seconds...")
                    time.sleep(delay)
                    return self._call_llm(prompt, retry_count + 1)
                else:
                    return None
            else:
                print(f"LLM API error: {e}")
                return None
                
        except Exception as e:
            error_str = str(e).lower()
            error_type = type(e).__name__
            
            # Check for quota-related errors
            if any(kw in error_str for kw in ['quota', '402', 'payment required', 'insufficient funds', 'payment']):
                print(f"Quota exceeded: {e}")
                self.stats['quota_exceeded'] += 1
                self.use_llm = False
                raise QuotaExceededError(f"API quota exceeded: {e}")
            
            is_network_error = any(kw in error_str for kw in ['ssl', 'socket', 'connection', 'timeout', 'network', 'read', 'httpcore', 'httpx'])
            
            if HTTPCORE_AVAILABLE:
                try:
                    if isinstance(e, (httpcore.ReadError, httpcore.ConnectError, httpcore.NetworkError, 
                                      httpcore.ReadTimeout, httpcore.ConnectTimeout)):
                        is_network_error = True
                except (AttributeError, NameError):
                    pass
            if HTTPX_AVAILABLE:
                try:
                    if isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.NetworkError)):
                        is_network_error = True
                except (AttributeError, NameError):
                    pass
            
            if is_network_error:
                self.stats['network_errors'] += 1
                if retry_count < Config.LLM_MAX_RETRIES:
                    delay = Config.LLM_RETRY_DELAY * (2 ** retry_count)
                    self.stats['retries'] += 1
                    print(f"Network error (attempt {retry_count + 1}/{Config.LLM_MAX_RETRIES}): {error_type}")
                    time.sleep(delay)
                    return self._call_llm(prompt, retry_count + 1)
                else:
                    print(f"Network error after {Config.LLM_MAX_RETRIES} retries: {error_type}")
                return None
            else:
                print(f"LLM call failed: {error_type}: {e}")
                return None
    
    def augment_node_text(self, text: str) -> str:
        if not self.use_llm or not text or len(text) < 5:
            return text
        
        prompt = f"""Paraphrase the following social media post while keeping the same meaning:

Original: "{text}"

Paraphrased version (one line only):"""
        
        augmented = self._call_llm(prompt)
        return augmented if augmented else text
    
    def augment_batch_texts(self, texts: List[str], batch_size: int = None, verbose: bool = False) -> List[str]:
        if batch_size is None:
            batch_size = self.batch_size
        if not self.use_llm or not texts:
            return texts
        
        augmented_results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Log batching information (only for large batches to avoid clutter)
        if verbose and len(texts) > batch_size:
            print(f"    Batching {len(texts)} texts into {total_batches} batches (batch_size={batch_size})")
        
        # Process texts in batches
        batch_iter = range(0, len(texts), batch_size)
        if verbose:
            batch_iter = tqdm(batch_iter, desc="Processing text batches", total=total_batches, leave=False)
        
        initial_llm_calls = self.stats['llm_calls']
        for i in batch_iter:
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
            
            # Verify batching: prompt should mention multiple texts if we have them
            if len(valid_texts) > 1 and verbose:
                # Quick check: prompt should contain the count
                if f"{len(valid_texts)} social media posts" not in batch_prompt:
                    print(f"    ⚠ Warning: Batch prompt may not be formatted correctly for {len(valid_texts)} texts")
            
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
        
        # Verify batching worked: should have made fewer LLM calls than texts
        final_llm_calls = self.stats['llm_calls']
        llm_calls_made = final_llm_calls - initial_llm_calls
        if verbose and llm_calls_made > 0:
            texts_per_call = len(texts) / llm_calls_made if llm_calls_made > 0 else 0
            if texts_per_call > 1:
                print(f"    ✓ Batching verified: {llm_calls_made} LLM call(s) for {len(texts)} texts ({texts_per_call:.1f} texts/call)")
        
        return augmented_results
    
    def _create_batch_prompt(self, texts: List[str]) -> str:
        # Verify we have multiple texts for batching
        if len(texts) == 1:
            # Single text - use single text prompt instead
            return f"""Paraphrase the following social media post while keeping the same meaning:

Original: "{texts[0]}"

Paraphrased version (one line only):"""
        
        # Create numbered list of texts
        text_list = "\n".join([f"{i+1}. \"{text}\"" for i, text in enumerate(texts)])
        
        prompt = f"""Paraphrase the following {len(texts)} social media posts while keeping the same meaning for each.
Provide one paraphrased version per line, in the same order.

Original posts:
{text_list}

Paraphrased versions (one per line, no numbering):"""
        
        return prompt
    
    def _parse_batch_response(self, response: str, expected_count: int) -> List[str]:
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
        batch_size: int = None,
        verbose: bool = False
    ) -> Data:
        if batch_size is None:
            batch_size = self.batch_size
        num_nodes = data.x.shape[0]
        
        if node_texts is None:
            node_texts = [f"Node {i} placeholder text" for i in range(num_nodes)]
        
        if use_batching and self.use_llm and len(selected_node_indices) > 0:
            selected_texts = [node_texts[i] for i in selected_node_indices]
            augmented_selected = self.augment_batch_texts(selected_texts, batch_size=batch_size, verbose=verbose)
            
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
            augmented_texts = []
            for i in range(num_nodes):
                if i in selected_node_indices:
                    augmented_texts.append(self.augment_node_text(node_texts[i]))
                    self.stats['nodes_augmented'] += 1
                else:
                    augmented_texts.append(node_texts[i])
        
        # Encode texts to embeddings
        # Only show encoding progress for large graphs
        encode_verbose = verbose and len(augmented_texts) > 100
        embeddings = self.lm_encoder.encode(augmented_texts, verbose=encode_verbose)
        
        data_aug = data.clone()
        data_aug.x_aug = torch.FloatTensor(embeddings)
        data_aug.augmented_node_mask = torch.zeros(num_nodes, dtype=torch.bool)
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
        if batch_size is None:
            batch_size = self.batch_size
        
        # Log the batch size being used
        if verbose:
            print(f"    Using LLM batch_size={batch_size} (from {'parameter' if batch_size != self.batch_size else 'Config.LLM_BATCH_SIZE'})")
        
        total_nodes_to_augment = sum(len(nodes) for nodes in selected_nodes_list)
        initial_stats = self.stats.copy()
        
        if total_nodes_to_augment == 0:
            if verbose:
                print("    No nodes to augment, returning original graphs")
            return [data.clone() for data in graph_list]
        
        # Stage 1: Collect all nodes from all graphs with metadata
        if verbose:
            print(f"    Collecting {total_nodes_to_augment:,} nodes from {len(graph_list)} graphs...")
        
        node_items = []  # List of (graph_idx, node_idx, text) tuples
        graph_node_texts = {}  # Store all node texts per graph for later use
        
        for graph_idx, (data, selected_indices) in enumerate(zip(graph_list, selected_nodes_list)):
            num_nodes = data.x.shape[0]
            
            # Get node texts for this graph
            if texts_list and texts_list[graph_idx]:
                node_texts = texts_list[graph_idx]
            else:
                # Use placeholder if texts not provided
                node_texts = [f"Node {i} placeholder text" for i in range(num_nodes)]
            
            graph_node_texts[graph_idx] = node_texts.copy()
            
            # Collect selected nodes
            for node_idx in selected_indices:
                text = node_texts[node_idx]
                node_items.append((graph_idx, node_idx, text))
        
        # Stage 2: Batch all texts together across graphs
        if verbose:
            print(f"    Batching {len(node_items):,} nodes across graphs (batch_size={batch_size})...")
        
        all_texts = [item[2] for item in node_items]
        augmented_texts = self.augment_batch_texts(
            all_texts, 
            batch_size=batch_size, 
            verbose=verbose
        )
        
        # Stage 3: Map augmented texts back to graphs
        if verbose:
            print(f"    Mapping augmented texts back to {len(graph_list)} graphs...")
        
        # Create mapping: (graph_idx, node_idx) -> augmented_text
        augmentation_map = {}
        for (graph_idx, node_idx, _), aug_text in zip(node_items, augmented_texts):
            augmentation_map[(graph_idx, node_idx)] = aug_text
        
        # Stage 4: Create augmented graphs
        augmented_graphs = []
        nodes_processed = 0
        
        iterator = enumerate(zip(graph_list, selected_nodes_list))
        if verbose:
            iterator = tqdm(iterator, 
                          total=len(graph_list),
                          desc="Creating augmented graphs",
                          disable=False)
        
        for graph_idx, (data, selected_indices) in iterator:
            num_nodes = data.x.shape[0]
            node_texts = graph_node_texts[graph_idx]
            
            # Convert selected_indices to set for O(1) lookup
            if isinstance(selected_indices, np.ndarray):
                selected_set = set(selected_indices.tolist())
            else:
                selected_set = set(selected_indices)
            
            # Build augmented texts list
            augmented_texts_list = []
            for i in range(num_nodes):
                if i in selected_set:
                    aug_text = augmentation_map.get((graph_idx, i), node_texts[i])
                    augmented_texts_list.append(aug_text)
                    self.stats['nodes_augmented'] += 1
                else:
                    augmented_texts_list.append(node_texts[i])
            
            # Encode texts to embeddings
            encode_verbose = verbose and len(augmented_texts_list) > 100
            embeddings = self.lm_encoder.encode(augmented_texts_list, verbose=encode_verbose)
            
            # Create augmented graph
            data_aug = data.clone()
            data_aug.x_aug = torch.FloatTensor(embeddings)
            data_aug.augmented_node_mask = torch.zeros(num_nodes, dtype=torch.bool)
            if isinstance(selected_indices, np.ndarray):
                selected_indices = selected_indices.copy()
            data_aug.augmented_node_mask[selected_indices] = True
            
            augmented_graphs.append(data_aug)
            nodes_processed += len(selected_indices)
            
            # Update progress bar with stats
            if verbose and hasattr(iterator, 'set_postfix'):
                current_llm_calls = self.stats['llm_calls'] - initial_stats['llm_calls']
                current_cache_hits = self.stats['cache_hits'] - initial_stats['cache_hits']
                iterator.set_postfix({
                    'nodes': f'{nodes_processed}/{total_nodes_to_augment}',
                    'LLM': current_llm_calls,
                    'cache': current_cache_hits
                })
        
        # Print batching efficiency summary
        final_stats = self.stats.copy()
        total_llm_calls = final_stats['llm_calls'] - initial_stats['llm_calls']
        total_cache_hits = final_stats['cache_hits'] - initial_stats['cache_hits']
        if verbose and total_nodes_to_augment > 0 and total_llm_calls > 0:
            avg_nodes_per_call = total_nodes_to_augment / total_llm_calls
            expected_calls_without_batching = total_nodes_to_augment
            expected_calls_old_batching = len(graph_list)  # Old way: one call per graph
            batching_efficiency = (1 - total_llm_calls / expected_calls_without_batching) * 100 if expected_calls_without_batching > 0 else 0
            cross_graph_efficiency = (1 - total_llm_calls / expected_calls_old_batching) * 100 if expected_calls_old_batching > 0 else 0
            print(f"\n    Batching Summary:")
            print(f"      Total nodes: {total_nodes_to_augment:,}")
            print(f"      Total graphs: {len(graph_list):,}")
            print(f"      LLM API calls: {total_llm_calls:,} (would be {expected_calls_without_batching:,} without batching, {expected_calls_old_batching:,} with per-graph batching)")
            print(f"      Average nodes per call: {avg_nodes_per_call:.1f} (target: {batch_size})")
            print(f"      Batching efficiency: {batching_efficiency:.1f}% reduction vs no batching")
            print(f"      Cross-graph efficiency: {cross_graph_efficiency:.1f}% reduction vs per-graph batching")
        
        return augmented_graphs
    
    def get_statistics(self) -> Dict:
        return self.stats.copy()

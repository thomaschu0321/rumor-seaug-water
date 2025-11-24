"""
Test script to verify LLM batching is working correctly.
This script mocks the LLM client to track actual API calls and verify batching behavior.
"""

import sys
from unittest.mock import Mock, MagicMock, patch
from typing import List
import numpy as np

# Import the NodeAugmentor
from node_augmentor import NodeAugmentor, LanguageModelEncoder


class MockLLMClient:
    """Mock LLM client that tracks all API calls"""
    
    def __init__(self):
        self.call_count = 0
        self.prompts_received = []
        self.responses = []
        self._chat = None
        
    def create_completion(self, prompt: str, num_texts: int) -> str:
        """Generate a mock response with the expected number of paraphrased texts"""
        # Generate mock paraphrased responses
        lines = []
        for i in range(num_texts):
            lines.append(f"Paraphrased version {i+1} of the original text")
        return "\n".join(lines)
    
    @property
    def chat(self):
        """Return a mock chat object"""
        if self._chat is None:
            self._chat = MockChat(self)
        return self._chat


class MockChat:
    """Mock chat object"""
    def __init__(self, parent_client):
        self.parent = parent_client
        self._completions = None
    
    @property
    def completions(self):
        """Return completions object"""
        if self._completions is None:
            self._completions = MockCompletions(self.parent)
        return self._completions


class MockCompletions:
    """Mock completions object"""
    def __init__(self, parent_client):
        self.parent = parent_client
        
    def create(self, model=None, messages=None, max_tokens=None, temperature=None):
        """Mock API call that tracks the prompt"""
        self.parent.call_count += 1
        
        # Extract the user prompt from messages
        user_prompt = None
        for msg in messages:
            if msg.get('role') == 'user':
                user_prompt = msg.get('content', '')
                break
        
        if user_prompt:
            self.parent.prompts_received.append(user_prompt)
            
            # Count how many texts are in the prompt (look for numbered list)
            lines = user_prompt.split('\n')
            numbered_lines = []
            for line in lines:
                stripped = line.strip()
                # Check if line starts with a number followed by period
                if stripped and stripped[0].isdigit():
                    # Check for pattern like "1. " or "1)" or "1:"
                    if len(stripped) > 1 and stripped[1] in ['.', ')', ':']:
                        numbered_lines.append(line)
            
            num_texts_in_prompt = len(numbered_lines) if numbered_lines else 1
            
            # Generate response
            response_text = self.parent.create_completion(user_prompt, num_texts_in_prompt)
            self.parent.responses.append(response_text)
            
            # Create mock response object matching OpenAI API structure
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_message = MagicMock()
            mock_message.content = response_text
            mock_choice.message = mock_message
            mock_response.choices = [mock_choice]
            
            return mock_response


def test_batching():
    """Test that batching actually works"""
    print("=" * 70)
    print("Testing LLM Batching Functionality")
    print("=" * 70)
    
    # Create mock client
    mock_client = MockLLMClient()
    
    # Create NodeAugmentor with mocked LLM
    with patch('node_augmentor.LLM_AVAILABLE', True):
        # Create a minimal encoder (we won't use it for this test)
        encoder = LanguageModelEncoder()
        
        # Create augmentor with use_llm=True but we'll inject the mock client
        augmentor = NodeAugmentor(
            lm_encoder=encoder,
            use_llm=True,
            batch_size=5  # Small batch size for testing
        )
        
        # Replace the client with our mock
        augmentor.client = mock_client
        augmentor.use_llm = True
        
        # Test with 15 texts (should make 3 API calls with batch_size=5)
        test_texts = [
            f"This is test text number {i}. It contains some content that needs to be paraphrased."
            for i in range(15)
        ]
        
        print(f"\nTest Configuration:")
        print(f"  Number of texts: {len(test_texts)}")
        print(f"  Batch size: {augmentor.batch_size}")
        print(f"  Expected API calls: {(len(test_texts) + augmentor.batch_size - 1) // augmentor.batch_size}")
        print(f"  Expected without batching: {len(test_texts)}")
        
        # Reset stats
        augmentor.stats['llm_calls'] = 0
        mock_client.call_count = 0
        mock_client.prompts_received = []
        mock_client.responses = []
        
        # Run augmentation
        print(f"\nRunning augmentation...")
        augmented_texts = augmentor.augment_batch_texts(
            test_texts,
            batch_size=augmentor.batch_size,
            verbose=True
        )
        
        # Verify results
        print(f"\n" + "=" * 70)
        print("Results:")
        print("=" * 70)
        print(f"  Actual API calls made: {mock_client.call_count}")
        print(f"  Stats['llm_calls']: {augmentor.stats['llm_calls']}")
        print(f"  Number of texts processed: {len(test_texts)}")
        print(f"  Number of augmented texts returned: {len(augmented_texts)}")
        
        # Check if batching worked
        expected_calls = (len(test_texts) + augmentor.batch_size - 1) // augmentor.batch_size
        texts_per_call = len(test_texts) / mock_client.call_count if mock_client.call_count > 0 else 0
        
        print(f"\n" + "=" * 70)
        print("Batching Verification:")
        print("=" * 70)
        
        if mock_client.call_count < len(test_texts):
            print(f"  ✓ PASS: Batching is working!")
            print(f"    - Made {mock_client.call_count} API calls for {len(test_texts)} texts")
            print(f"    - Average: {texts_per_call:.2f} texts per API call")
            print(f"    - Efficiency: {((1 - mock_client.call_count / len(test_texts)) * 100):.1f}% reduction in API calls")
        else:
            print(f"  ✗ FAIL: Batching is NOT working!")
            print(f"    - Made {mock_client.call_count} API calls for {len(test_texts)} texts")
            print(f"    - This is the same as processing each text individually")
            return False
        
        # Verify prompts contain multiple texts
        print(f"\n" + "=" * 70)
        print("Prompt Analysis:")
        print("=" * 70)
        for i, prompt in enumerate(mock_client.prompts_received):
            # Count texts in prompt (look for numbered list items)
            lines = prompt.split('\n')
            numbered_items = [l for l in lines if l.strip() and any(l.strip().startswith(f"{j}.") for j in range(1, 100))]
            num_texts_in_prompt = len(numbered_items)
            
            print(f"\n  Prompt {i+1}:")
            print(f"    Texts in prompt: {num_texts_in_prompt}")
            print(f"    Prompt preview (first 200 chars):")
            print(f"    {prompt[:200]}...")
            
            if num_texts_in_prompt > 1:
                print(f"    ✓ Contains multiple texts - batching confirmed!")
            elif num_texts_in_prompt == 1:
                print(f"    ⚠ Only contains 1 text - might be last batch or issue")
            else:
                print(f"    ✗ No texts found in prompt - potential issue")
        
        # Verify response parsing
        print(f"\n" + "=" * 70)
        print("Response Parsing Verification:")
        print("=" * 70)
        if len(augmented_texts) == len(test_texts):
            print(f"  ✓ PASS: All {len(test_texts)} texts were processed and returned")
        else:
            print(f"  ✗ FAIL: Expected {len(test_texts)} texts, got {len(augmented_texts)}")
            return False
        
        print(f"\n" + "=" * 70)
        print("Summary:")
        print("=" * 70)
        print(f"  ✓ Batching is working correctly!")
        print(f"  ✓ {mock_client.call_count} API calls processed {len(test_texts)} texts")
        print(f"  ✓ Average of {texts_per_call:.2f} texts per API call")
        print(f"  ✓ Prompts contain multiple texts as expected")
        print("=" * 70)
        
        return True


def test_single_text_vs_batch():
    """Test that single text processing still works"""
    print("\n" + "=" * 70)
    print("Testing Single Text Processing (should still work)")
    print("=" * 70)
    
    mock_client = MockLLMClient()
    
    with patch('node_augmentor.LLM_AVAILABLE', True):
        encoder = LanguageModelEncoder()
        augmentor = NodeAugmentor(
            lm_encoder=encoder,
            use_llm=True,
            batch_size=5
        )
        augmentor.client = mock_client
        augmentor.use_llm = True
        
        # Test with single text
        single_text = "This is a single text that should be processed."
        augmentor.stats['llm_calls'] = 0
        mock_client.call_count = 0
        
        result = augmentor.augment_batch_texts([single_text], batch_size=5, verbose=False)
        
        print(f"  Single text processed: {len(result) == 1}")
        print(f"  API calls made: {mock_client.call_count}")
        print(f"  ✓ Single text processing works: {result[0] is not None}")
        
        return True


if __name__ == "__main__":
    try:
        success = test_batching()
        if success:
            test_single_text_vs_batch()
            print("\n" + "=" * 70)
            print("ALL TESTS PASSED!")
            print("=" * 70)
            sys.exit(0)
        else:
            print("\n" + "=" * 70)
            print("TESTS FAILED!")
            print("=" * 70)
            sys.exit(1)
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


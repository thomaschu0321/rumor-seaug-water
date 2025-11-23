#!/usr/bin/env python3
"""
Test script to verify DeepSeek API configuration and connection
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import config
from config import Config

def test_config():
    """Test if configuration is properly loaded"""
    print("=" * 70)
    print("DeepSeek Configuration Test")
    print("=" * 70)
    
    # Check LLM Provider
    print(f"\n1. LLM Provider: {Config.LLM_PROVIDER}")
    if Config.LLM_PROVIDER != 'deepseek':
        print(f"   ⚠️  WARNING: LLM_PROVIDER is '{Config.LLM_PROVIDER}', expected 'deepseek'")
        print(f"   → Set LLM_PROVIDER=deepseek in your .env file")
    
    # Check API Key
    print(f"\n2. DeepSeek API Key: {'✓ SET' if Config.DEEPSEEK_API_KEY else '✗ NOT SET'}")
    if not Config.DEEPSEEK_API_KEY:
        print(f"   ⚠️  ERROR: DEEPSEEK_API_KEY is not set!")
        print(f"   → Add DEEPSEEK_API_KEY=your_key_here to your .env file")
        return False
    
    # Check Base URL
    print(f"\n3. DeepSeek Base URL: {Config.DEEPSEEK_BASE_URL}")
    if not Config.DEEPSEEK_BASE_URL:
        print(f"   ⚠️  WARNING: DEEPSEEK_BASE_URL is not set, using default")
    
    # Check Model
    print(f"\n4. DeepSeek Model: {Config.DEEPSEEK_MODEL}")
    
    # Check other parameters
    print(f"\n5. Other Parameters:")
    print(f"   - Max Tokens: {Config.LLM_MAX_TOKENS}")
    print(f"   - Temperature: {Config.LLM_TEMPERATURE}")
    print(f"   - Batch Size: {Config.LLM_BATCH_SIZE}")
    print(f"   - Use LLM: {Config.USE_LLM}")
    
    return True

def test_api_connection():
    """Test actual API connection"""
    print("\n" + "=" * 70)
    print("DeepSeek API Connection Test")
    print("=" * 70)
    
    try:
        from openai import OpenAI
        
        # Ensure base URL ends with /v1 for OpenAI-compatible APIs
        base_url = Config.DEEPSEEK_BASE_URL.rstrip('/')
        if not base_url.endswith('/v1'):
            base_url = f"{base_url}/v1"
        
        # Initialize client
        client = OpenAI(
            api_key=Config.DEEPSEEK_API_KEY,
            base_url=base_url
        )
        
        print(f"\n✓ Client initialized")
        print(f"  Model: {Config.DEEPSEEK_MODEL}")
        print(f"  Base URL (original): {Config.DEEPSEEK_BASE_URL}")
        print(f"  Base URL (adjusted): {base_url}")
        
        # Make a simple test call
        print(f"\n📡 Making test API call...")
        try:
            response = client.chat.completions.create(
                model=Config.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say 'Hello, DeepSeek is working!' in one sentence."}
                ],
                max_tokens=50,
                temperature=0.7
            )
        except Exception as api_error:
            print(f"\n✗ API Call Failed with exception!")
            print(f"   Error type: {type(api_error).__name__}")
            print(f"   Error message: {str(api_error)}")
            if hasattr(api_error, 'response'):
                print(f"   Response status: {getattr(api_error.response, 'status_code', 'N/A')}")
                if hasattr(api_error.response, 'text'):
                    print(f"   Response text: {api_error.response.text[:500]}")
            raise
        
        # Debug: Check response type
        print(f"   Response type: {type(response)}")
        if isinstance(response, str):
            print(f"   Response is a string: {response[:200]}")
            raise ValueError(f"API returned string instead of object: {response[:200]}")
        
        # Check if response has expected attributes
        if not hasattr(response, 'choices'):
            response_str = str(response)
            if len(response_str) > 500:
                response_str = response_str[:500] + "..."
            print(f"   Response attributes: {dir(response)[:15]}")
            print(f"   Response content: {response_str}")
            raise ValueError(f"Response object missing 'choices' attribute. Type: {type(response)}")
        
        if not response.choices or len(response.choices) == 0:
            raise ValueError("Response has no choices")
        
        result = response.choices[0].message.content.strip()
        print(f"\n✓ API Call Successful!")
        print(f"  Response: {result}")
        print(f"  Model: {response.model}")
        print(f"  Tokens used: {response.usage.total_tokens if hasattr(response, 'usage') else 'N/A'}")
        
        return True
        
    except ImportError:
        print(f"\n✗ ERROR: 'openai' package not installed")
        print(f"   → Run: pip install openai")
        return False
    except Exception as e:
        print(f"\n✗ API Call Failed!")
        print(f"   Error: {str(e)}")
        print(f"\n   Common issues:")
        print(f"   - Invalid API key")
        print(f"   - Incorrect base URL")
        print(f"   - Network connectivity issues")
        print(f"   - Model name incorrect")
        return False

def test_node_augmentor():
    """Test NodeAugmentor integration"""
    print("\n" + "=" * 70)
    print("NodeAugmentor Integration Test")
    print("=" * 70)
    
    try:
        from node_augmentor import NodeAugmentor
        
        print(f"\n📦 Initializing NodeAugmentor...")
        augmentor = NodeAugmentor(use_llm=True)
        
        if not augmentor.use_llm:
            print(f"\n✗ NodeAugmentor LLM is disabled")
            return False
        
        print(f"\n✓ NodeAugmentor initialized")
        print(f"  Provider: {augmentor.provider}")
        print(f"  Model: {augmentor.model}")
        
        # Test single text augmentation
        test_text = "This is a test tweet about a breaking news story."
        print(f"\n📝 Testing text augmentation...")
        print(f"  Original: {test_text}")
        
        augmented = augmentor.augment_node_text(test_text)
        if augmented and augmented != test_text:
            print(f"  Augmented: {augmented}")
            print(f"\n✓ Text augmentation successful!")
        else:
            print(f"  ⚠️  Augmentation returned original text (may be cached or failed)")
        
        # Show stats
        print(f"\n📊 Augmentation Stats:")
        for key, value in augmentor.stats.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ NodeAugmentor test failed!")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("DeepSeek API Test Suite")
    print("=" * 70)
    
    # Test 1: Configuration
    config_ok = test_config()
    
    if not config_ok:
        print("\n" + "=" * 70)
        print("✗ Configuration test failed. Please fix .env file first.")
        print("=" * 70)
        sys.exit(1)
    
    # Test 2: API Connection
    api_ok = test_api_connection()
    
    if not api_ok:
        print("\n" + "=" * 70)
        print("✗ API connection test failed. Please check your API key and network.")
        print("=" * 70)
        sys.exit(1)
    
    # Test 3: NodeAugmentor Integration
    augmentor_ok = test_node_augmentor()
    
    # Final summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Configuration: {'✓ PASS' if config_ok else '✗ FAIL'}")
    print(f"API Connection: {'✓ PASS' if api_ok else '✗ FAIL'}")
    print(f"NodeAugmentor: {'✓ PASS' if augmentor_ok else '✗ FAIL'}")
    
    if config_ok and api_ok and augmentor_ok:
        print("\n🎉 All tests passed! DeepSeek is ready to use.")
        print("\n💡 Next steps:")
        print("   - Run: python node_augmentor.py (test full augmentation)")
        print("   - Run: python seaug_pipeline.py --enable_augmentation (full pipeline)")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
    
    print("=" * 70)


"""
Test Azure OpenAI API Connection

This script makes a single API call to verify your Azure OpenAI setup is working.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_azure_api():
    """Test Azure OpenAI API with a simple call"""
    print("="*70)
    print("Testing Azure OpenAI API Connection")
    print("="*70)
    
    # Load configuration
    azure_api_key = os.environ.get('AZURE_API_KEY')
    azure_endpoint = os.environ.get('AZURE_ENDPOINT', 'https://cuhk-apip.azure-api.net')
    azure_model = os.environ.get('AZURE_MODEL', 'gpt-4o-mini')
    api_version = os.environ.get('API_VERSION', '2023-05-15')
    
    # Check if API key is set
    if not azure_api_key:
        print("\n❌ ERROR: AZURE_API_KEY is not set!")
        print("\nPlease set it in your .env file:")
        print("AZURE_API_KEY=your_api_key_here")
        return False
    
    # Display configuration (mask API key)
    print("\n📋 Configuration:")
    print(f"  Endpoint: {azure_endpoint}")
    print(f"  Model: {azure_model}")
    print(f"  API Version: {api_version}")
    print(f"  API Key: {'*' * 20}{azure_api_key[-4:] if len(azure_api_key) > 4 else '****'}")
    
    # Try to import openai
    try:
        from openai import AzureOpenAI
        print("\n✓ OpenAI package imported successfully")
    except ImportError:
        print("\n❌ ERROR: openai package not installed!")
        print("Install it with: pip install openai")
        return False
    
    # Initialize client
    print("\n🔧 Initializing Azure OpenAI client...")
    try:
        client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            api_key=azure_api_key
        )
        print("✓ Client initialized successfully")
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize client: {e}")
        return False
    
    # Make a test API call
    print("\n📞 Making test API call...")
    test_prompt = "Say 'Hello! The Azure API is working correctly.' in one short sentence."
    
    try:
        response = client.chat.completions.create(
            model=azure_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": test_prompt}
            ],
            max_tokens=50,
            temperature=0.7
        )
        
        # Extract response
        response_text = response.choices[0].message.content.strip()
        
        print("\n✅ SUCCESS! API call completed successfully")
        print("\n📨 Response:")
        print(f"  {response_text}")
        
        # Show token usage
        if hasattr(response, 'usage'):
            print("\n📊 Token Usage:")
            print(f"  Prompt tokens: {response.usage.prompt_tokens}")
            print(f"  Completion tokens: {response.usage.completion_tokens}")
            print(f"  Total tokens: {response.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: API call failed!")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error message: {e}")
        
        # Provide troubleshooting hints
        error_str = str(e)
        if "401" in error_str or "authentication" in error_str.lower():
            print("\n💡 Troubleshooting: This looks like an authentication error.")
            print("  - Check that your AZURE_API_KEY is correct")
            print("  - Verify the key hasn't expired")
        elif "403" in error_str or "quota" in error_str.lower():
            print("\n💡 Troubleshooting: This looks like a quota/permission error.")
            print("  - Check your API quota limits")
            print("  - Verify you have permission to use this model")
        elif "404" in error_str or "not found" in error_str.lower():
            print("\n💡 Troubleshooting: This looks like a resource not found error.")
            print("  - Check that your AZURE_ENDPOINT is correct")
            print("  - Verify the AZURE_MODEL name is correct")
        elif "429" in error_str or "rate limit" in error_str.lower():
            print("\n💡 Troubleshooting: Rate limit exceeded.")
            print("  - Wait a moment and try again")
        
        return False
    
    finally:
        print("\n" + "="*70)


if __name__ == '__main__':
    success = test_azure_api()
    
    if success:
        print("✅ Azure OpenAI API test completed successfully!")
        print("\nYou can now use the API in your main pipeline.")
        print("To enable it, set USE_LLM=true in your .env file")
        print("or use the --use_llm flag when running seaug_pipeline.py")
    else:
        print("❌ Azure OpenAI API test failed!")
        print("\nPlease fix the issues above before using the API.")
    
    print("="*70)


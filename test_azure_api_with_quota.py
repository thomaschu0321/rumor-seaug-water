"""
Test Azure OpenAI API Connection WITH Quota Tracking

This script makes a single API call AND tracks it in the quota system.
"""

import os
from dotenv import load_dotenv
from rate_limiter import RateLimiter

# Load environment variables
load_dotenv()

def test_azure_api_with_tracking():
    """Test Azure OpenAI API with proper quota tracking"""
    print("="*70)
    print("Testing Azure OpenAI API Connection (WITH Quota Tracking)")
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
    
    # Initialize rate limiter
    print("\n📊 Checking quota status...")
    rate_limiter = RateLimiter(
        calls_per_minute=5,   # CUHK limit
        calls_per_week=100    # CUHK limit
    )
    
    # Show current quota
    status = rate_limiter.get_quota_status()
    print(f"  Weekly quota: {status['weekly_used']}/{status['weekly_total']} used")
    print(f"  Remaining: {status['weekly_remaining']} calls")
    
    # Check if we can make a call
    can_call, reason, wait_time = rate_limiter.can_make_call()
    if not can_call:
        print(f"\n❌ ERROR: {reason}")
        if "Weekly quota" in reason:
            print(f"   Quota will reset in {wait_time/86400:.1f} days")
        else:
            print(f"   Need to wait {wait_time:.0f} seconds")
        return False
    
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
    print("\n📞 Making test API call (will be tracked in quota)...")
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
        
        # ⭐ RECORD THE CALL IN QUOTA TRACKER
        rate_limiter.record_call()
        
        print("\n✅ SUCCESS! API call completed and tracked")
        print("\n📨 Response:")
        print(f"  {response_text}")
        
        # Show token usage
        if hasattr(response, 'usage'):
            print("\n📊 Token Usage:")
            print(f"  Prompt tokens: {response.usage.prompt_tokens}")
            print(f"  Completion tokens: {response.usage.completion_tokens}")
            print(f"  Total tokens: {response.usage.total_tokens}")
        
        # Show updated quota
        print("\n📊 Updated Quota Status:")
        status = rate_limiter.get_quota_status()
        print(f"  Weekly quota: {status['weekly_used']}/{status['weekly_total']} used")
        print(f"  Remaining: {status['weekly_remaining']} calls")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: API call failed!")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error message: {e}")
        
        # Provide troubleshooting hints
        error_str = str(e)
        if "401" in error_str or "authentication" in error_str.lower():
            print("\n💡 Troubleshooting: Authentication error")
            print("  - Check that your AZURE_API_KEY is correct")
        elif "403" in error_str or "quota" in error_str.lower():
            print("\n💡 Troubleshooting: Quota/permission error")
            print("  - Your Azure API quota may be exhausted")
        elif "404" in error_str:
            print("\n💡 Troubleshooting: Resource not found")
            print("  - Check your AZURE_ENDPOINT and AZURE_MODEL")
        elif "429" in error_str or "rate limit" in error_str.lower():
            print("\n💡 Troubleshooting: Rate limit exceeded")
            print("  - Wait a moment and try again")
        
        return False
    
    finally:
        print("\n" + "="*70)


if __name__ == '__main__':
    success = test_azure_api_with_tracking()
    
    if success:
        print("✅ Azure OpenAI API test completed successfully!")
        print("   This call has been recorded in your quota tracker.")
    else:
        print("❌ Azure OpenAI API test failed!")
    
    print("="*70)


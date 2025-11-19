#!/usr/bin/env python3
"""
Reset API Quota Tracker

Use this script to reset the local quota tracking file when:
1. The week has actually passed but the quota hasn't reset
2. The quota file has incorrect data
3. You want to start fresh with quota tracking

This does NOT affect the actual Azure API quota - it only resets
the local tracking file.
"""

import sys
from pathlib import Path
from rate_limiter import RateLimiter

def main():
    print("="*70)
    print("API Quota Reset Tool")
    print("="*70)
    
    # Create rate limiter
    limiter = RateLimiter()
    
    # Show current status
    print("\n📊 Current Quota Status:")
    limiter.print_status()
    
    # Ask for confirmation
    print("\n⚠️  This will reset the LOCAL quota tracking.")
    print("   It does NOT affect your actual Azure API quota.")
    response = input("\nDo you want to reset the quota tracker? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        limiter.reset_weekly_quota()
        print("\n📊 New Quota Status:")
        limiter.print_status()
        print("\n✓ Quota tracker reset successfully!")
    else:
        print("\n✗ Reset cancelled.")
    
    print("="*70)

if __name__ == '__main__':
    main()

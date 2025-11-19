#!/usr/bin/env python3
"""
Check API Quota Status

Quick script to view the current API quota status.
"""

from rate_limiter import RateLimiter

def main():
    limiter = RateLimiter()
    limiter.print_status()
    
    # Show quota file location
    print(f"\nQuota file location: {limiter.quota_file}")
    
    # Show week start time
    print(f"Week started: {limiter.week_start.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if can make call
    can_call, reason, wait_time = limiter.can_make_call()
    print(f"\nCan make API call now? {'✓ Yes' if can_call else '✗ No'}")
    if not can_call:
        print(f"Reason: {reason}")
        if wait_time > 86400:  # More than a day
            print(f"Wait time: {wait_time/86400:.1f} days")
        elif wait_time > 3600:  # More than an hour
            print(f"Wait time: {wait_time/3600:.1f} hours")
        else:
            print(f"Wait time: {wait_time:.0f} seconds")

if __name__ == '__main__':
    main()

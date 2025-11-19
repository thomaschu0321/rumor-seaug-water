"""
Rate Limiter for CUHK API
Handles the strict API quota limits:
- 5 calls per minute
- 100 calls per week
"""

import time
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

class RateLimiter:
    """
    Rate limiter with quota tracking
    """
    
    def __init__(
        self,
        calls_per_minute: int = 5,
        calls_per_week: int = 100,
        quota_file: str = None
    ):
        """
        Initialize rate limiter
        
        Args:
            calls_per_minute: Maximum calls per minute
            calls_per_week: Maximum calls per week
            quota_file: File to store quota information
        """
        self.calls_per_minute = calls_per_minute
        self.calls_per_week = calls_per_week
        
        # Quota tracking file (in project directory)
        if quota_file is None:
            project_root = Path(__file__).parent
            quota_file = project_root / 'data' / '.seaug_api_quota.json'
            # Ensure data directory exists
            quota_file.parent.mkdir(parents=True, exist_ok=True)
        self.quota_file = Path(quota_file)
        
        # Load existing quota
        self.load_quota()
        
        # Call timestamps (for per-minute rate limiting)
        self.call_history = []
    
    def load_quota(self):
        """Load quota from file"""
        if self.quota_file.exists():
            try:
                with open(self.quota_file, 'r') as f:
                    data = json.load(f)
                
                self.weekly_calls = data.get('weekly_calls', 0)
                self.week_start = datetime.fromisoformat(data.get('week_start', datetime.now().isoformat()))
                
                # Reset if new week
                if datetime.now() - self.week_start > timedelta(weeks=1):
                    self.weekly_calls = 0
                    self.week_start = datetime.now()
                    self.save_quota()
            except:
                self.weekly_calls = 0
                self.week_start = datetime.now()
        else:
            self.weekly_calls = 0
            self.week_start = datetime.now()
            self.save_quota()
    
    def save_quota(self):
        """Save quota to file"""
        data = {
            'weekly_calls': self.weekly_calls,
            'week_start': self.week_start.isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.quota_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def can_make_call(self) -> tuple:
        """
        Check if a call can be made
        
        Returns:
            (can_call: bool, reason: str, wait_time: float)
        """
        # Reload quota to ensure we have the latest data
        self.load_quota()
        
        # Check weekly quota
        if self.weekly_calls >= self.calls_per_week:
            days_until_reset = 7 - (datetime.now() - self.week_start).days
            return False, f"Weekly quota exceeded ({self.weekly_calls}/{self.calls_per_week})", days_until_reset * 86400
        
        # Check per-minute rate
        now = time.time()
        
        # Remove calls older than 1 minute
        self.call_history = [t for t in self.call_history if now - t < 60]
        
        if len(self.call_history) >= self.calls_per_minute:
            wait_time = 60 - (now - self.call_history[0])
            return False, f"Rate limit ({len(self.call_history)}/{self.calls_per_minute} per minute)", wait_time
        
        return True, "OK", 0
    
    def wait_if_needed(self):
        """Wait if rate limit is exceeded"""
        can_call, reason, wait_time = self.can_make_call()
        
        if not can_call:
            if "Weekly quota" in reason:
                print(f"\n⚠️  {reason}")
                print(f"   Quota will reset in {wait_time/86400:.1f} days")
                return False
            else:
                print(f"\n⏳ {reason}, waiting {wait_time:.0f}s...")
                time.sleep(wait_time + 1)  # Add 1 second buffer
                return True
        
        return True
    
    def record_call(self):
        """Record a successful API call"""
        now = time.time()
        self.call_history.append(now)
        self.weekly_calls += 1
        self.save_quota()
    
    def get_quota_status(self) -> dict:
        """Get current quota status"""
        self.load_quota()  # Refresh from file
        
        # Calculate remaining time in current week
        week_elapsed = datetime.now() - self.week_start
        week_remaining = timedelta(weeks=1) - week_elapsed
        
        # Recent minute stats
        now = time.time()
        recent_calls = len([t for t in self.call_history if now - t < 60])
        
        return {
            'weekly_used': self.weekly_calls,
            'weekly_total': self.calls_per_week,
            'weekly_remaining': self.calls_per_week - self.weekly_calls,
            'weekly_percentage': (self.weekly_calls / self.calls_per_week) * 100,
            'week_remaining_days': week_remaining.days,
            'minute_used': recent_calls,
            'minute_total': self.calls_per_minute,
            'minute_remaining': self.calls_per_minute - recent_calls,
        }
    
    def print_status(self):
        """Print quota status"""
        status = self.get_quota_status()
        
        print("\n" + "="*70)
        print("API Quota Status")
        print("="*70)
        print(f"\nWeekly Quota:")
        print(f"  Used: {status['weekly_used']}/{status['weekly_total']} ({status['weekly_percentage']:.1f}%)")
        print(f"  Remaining: {status['weekly_remaining']} calls")
        print(f"  Resets in: {status['week_remaining_days']} days")
        
        if status['weekly_percentage'] >= 75:
            print(f"  ⚠️  WARNING: Over 75% of weekly quota used!")
        
        print(f"\nPer-Minute Rate:")
        print(f"  Current: {status['minute_used']}/{status['minute_total']}")
        print(f"  Available: {status['minute_remaining']} calls")
        print("="*70)
    
    def reset_weekly_quota(self):
        """Manually reset weekly quota (for testing)"""
        self.weekly_calls = 0
        self.week_start = datetime.now()
        self.save_quota()
        print("✓ Weekly quota reset")


if __name__ == '__main__':
    # Test the rate limiter
    limiter = RateLimiter(calls_per_minute=5, calls_per_week=100)
    limiter.print_status()


import time
from app.conversation.storage import Storage

def run_cleanup(max_age_seconds: int = 1800) -> int:
    """
    Periodic task to clean up expired sessions that have been 
    inactive for a given period of time (default 30 minutes).
    """
    removed_count = Storage.cleanup_expired(max_age_seconds=max_age_seconds)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Session Cleanup: Removed {removed_count} expired sessions.")
    return removed_count

if __name__ == "__main__":
    # Can be run directly as a script, perhaps by a cron job or background worker
    run_cleanup()

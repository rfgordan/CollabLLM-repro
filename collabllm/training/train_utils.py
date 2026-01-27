import logging
from datetime import datetime

def get_timebased_filename():
  # Get the current time
  current_time = datetime.now()

  # Format it concisely (e.g., YearMonthDay_HourMinuteSecond)
  timestamp_suffix = current_time.strftime("%Y%m%d_%H%M%S")

  log_filename = f"test_{timestamp_suffix}"
  logging.info(f"Generated log filename: {log_filename}")
  return log_filename
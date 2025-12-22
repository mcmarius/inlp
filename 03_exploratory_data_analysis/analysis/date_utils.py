import re
from datetime import datetime
from typing import List, Tuple, Optional
import dateparser

# Regex to match Romanian dates in various formats
# e.g., 20.12, 20.12.2025, 20 decembrie 2025
DATE_PATTERN = r'(\d{1,2}(?:[\./]\d{1,2}(?:[\./]\d{2,4})?|\s+(?:ianuarie|februarie|martie|aprilie|mai|iunie|iulie|august|septembrie|octombrie|noiembrie|decembrie)(?:\s+\d{2,4})?))'
INTERVAL_PATTERN = rf'{DATE_PATTERN}\s*(?:[–\-]|până\s+la|și)\s*{DATE_PATTERN}'

def extract_date_intervals(text: str) -> List[Tuple[datetime, datetime]]:
    """
    Extracts date intervals from text and reconstructs missing years.
    Returns a list of (start_date, end_date) tuples.
    """
    intervals = []
    
    # Find potential intervals
    matches = re.finditer(INTERVAL_PATTERN, text, re.IGNORECASE)
    
    for match in matches:
        start_str = match.group(1).strip()
        end_str = match.group(2).strip()
        
        # Parse the end date first to get context (like year)
        # Using dateparser for robust Romanian parsing
        end_date = dateparser.parse(end_str, languages=['ro'])
        
        if not end_date:
            continue
            
        # Try to parse the start date
        start_date = dateparser.parse(start_str, languages=['ro'])
        
        # If start_date is parsed but has a different year or no year, 
        # we check if we need to inherit the year from end_date
        # dateparser might default missing year to current year
        
        # Check if start_str actually contains a year
        has_year_start = re.search(r'\d{4}', start_str)
        
        if not has_year_start and end_date:
            # Re-parse start_date with end_date's year
            # This is a bit tricky with dateparser, let's try a manual fix if needed
            year = end_date.year
            
            # If it's just "DD.MM", append the year
            if re.match(r'^\d{1,2}[\./]\d{1,2}$', start_str):
                enriched_start = f"{start_str}.{year}"
                start_date = dateparser.parse(enriched_start, languages=['ro'])
            elif re.match(r'^\d{1,2}\s+[a-z]+$', start_str.lower()):
                 enriched_start = f"{start_str} {year}"
                 start_date = dateparser.parse(enriched_start, languages=['ro'])
        
        if start_date and end_date:
            # Final sanity check: start should be before or same as end
            # If not, maybe it crossed a year boundary (e.g., 30.12 - 02.01.2025)
            if start_date > end_date and not has_year_start:
                # Try decrementing year for start_date
                try:
                    start_date = start_date.replace(year=end_date.year - 1)
                except ValueError: # Leap year issues
                    start_date = start_date.replace(year=end_date.year - 1, day=28)
            
            if start_date <= end_date:
                intervals.append((start_date, end_date))
                
    return intervals

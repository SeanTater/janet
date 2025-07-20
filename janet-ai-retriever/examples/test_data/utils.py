"""
Utility functions for data processing and analysis.

This module provides various helper functions for working with data,
including mathematical operations, string processing, and file handling.
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime


def calculate_statistics(numbers: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of numbers.
    
    Args:
        numbers: List of numerical values
        
    Returns:
        Dictionary containing mean, median, min, max, and standard deviation
    """
    if not numbers:
        return {}
    
    sorted_nums = sorted(numbers)
    n = len(numbers)
    
    # Calculate mean
    mean = sum(numbers) / n
    
    # Calculate median
    if n % 2 == 0:
        median = (sorted_nums[n//2 - 1] + sorted_nums[n//2]) / 2
    else:
        median = sorted_nums[n//2]
    
    # Calculate standard deviation
    variance = sum((x - mean) ** 2 for x in numbers) / n
    std_dev = variance ** 0.5
    
    return {
        'mean': mean,
        'median': median,
        'min': min(numbers),
        'max': max(numbers),
        'std_dev': std_dev,
        'count': n
    }


def process_json_file(file_path: str) -> Dict[str, Any]:
    """
    Load and process a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Parsed JSON data as a dictionary
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        # Add metadata
        data['_metadata'] = {
            'file_path': file_path,
            'file_size': os.path.getsize(file_path),
            'processed_at': datetime.now().isoformat()
        }
        
        return data
        
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {file_path}: {e}")


def format_text(text: str, max_length: int = 100, uppercase: bool = False) -> str:
    """
    Format text with various options.
    
    Args:
        text: Input text to format
        max_length: Maximum length of the output text
        uppercase: Whether to convert to uppercase
        
    Returns:
        Formatted text string
    """
    if not text:
        return ""
    
    # Apply transformations
    result = text.strip()
    
    if uppercase:
        result = result.upper()
    
    # Truncate if necessary
    if len(result) > max_length:
        result = result[:max_length - 3] + "..."
    
    return result


class DataProcessor:
    """
    A class for processing and analyzing data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data processor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.processed_count = 0
        
    def process_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a list of data records.
        
        Args:
            records: List of dictionaries representing data records
            
        Returns:
            List of processed records
        """
        processed_records = []
        
        for record in records:
            processed_record = self._process_single_record(record)
            processed_records.append(processed_record)
            self.processed_count += 1
            
        return processed_records
    
    def _process_single_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single data record.
        
        Args:
            record: Dictionary representing a single data record
            
        Returns:
            Processed record dictionary
        """
        # Create a copy to avoid modifying the original
        processed = record.copy()
        
        # Add processing metadata
        processed['_processed_at'] = datetime.now().isoformat()
        processed['_processor_id'] = id(self)
        
        # Apply any configured transformations
        if 'transformations' in self.config:
            for transformation in self.config['transformations']:
                processed = self._apply_transformation(processed, transformation)
        
        return processed
    
    def _apply_transformation(self, record: Dict[str, Any], transformation: str) -> Dict[str, Any]:
        """
        Apply a transformation to a record.
        
        Args:
            record: The record to transform
            transformation: The transformation to apply
            
        Returns:
            Transformed record
        """
        # Mock transformations for the example
        if transformation == 'normalize':
            # Normalize string values
            for key, value in record.items():
                if isinstance(value, str):
                    record[key] = value.lower().strip()
        
        elif transformation == 'add_timestamp':
            record['timestamp'] = datetime.now().isoformat()
        
        return record
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary containing processing statistics
        """
        return {
            'processed_count': self.processed_count,
            'config': self.config,
            'processor_id': id(self)
        }
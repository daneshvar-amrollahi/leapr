from dataclasses import dataclass
from typing import Set
import csv
from collections import defaultdict


@dataclass
class SMTInstance:
    """Represents an SMT benchmark with option sets that solved it."""
    benchmark: str  # The .smt2 file content
    option_sets: Set[str]  # Set of option strings that solved this benchmark
    path: str = ""  # Path to the benchmark file
    

def load_smt_data(csv_path: str, max_instances: int = None) -> list[SMTInstance]:
    """
    Load SMT instances from CSV file.
    
    CSV format: path,benchmark,option,total_time
    
    Groups rows by benchmark, collecting all option sets for each.
    """
    # Group by benchmark
    benchmark_data = defaultdict(lambda: {"options": set(), "path": ""})
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row.get('path', '')  # Get path, default to empty string if missing
            benchmark = row['benchmark']
            option_set = row['option']
            
            benchmark_data[benchmark]["options"].add(option_set)
            if path:  # Store path (all rows with same benchmark should have same path)
                benchmark_data[benchmark]["path"] = path
    
    # Convert to SMTInstance objects
    instances = [
        SMTInstance(
            benchmark=bench,
            option_sets=data["options"],
            path=data["path"],
        )
        for bench, data in benchmark_data.items()
    ]
    
    # Limit if requested
    if max_instances and len(instances) > max_instances:
        instances = instances[:max_instances]
    
    return instances
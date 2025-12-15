# Manual features for testing SMT solver domain
# These will be used instead of LLM-generated features during testing

def feature(benchmark: str) -> float:
    """Count number of quantifiers (forall and exists)"""
    return float(benchmark.count('forall') + benchmark.count('exists'))

def feature(benchmark: str) -> float:
    """Check if uses bit-vector logic"""
    return 1.0 if 'BitVec' in benchmark else 0.0

def feature(benchmark: str) -> float:
    """Length of benchmark in characters"""
    return float(len(benchmark))

def feature(benchmark: str) -> float:
    """Number of assert statements"""
    return float(benchmark.count('(assert'))

def feature(benchmark: str) -> float:
    """Number of declare-fun statements"""
    return float(benchmark.count('(declare-fun'))

def feature(benchmark: str) -> float:
    """Check if uses arrays"""
    return 1.0 if 'Array' in benchmark else 0.0

def feature(benchmark: str) -> float:
    """Number of check-sat calls"""
    return float(benchmark.count('(check-sat'))

def feature(benchmark: str) -> float:
    """Has set-logic directive"""
    return 1.0 if '(set-logic' in benchmark else 0.0

#!/usr/bin/env python3

"""
Implementation of the SMT Solver Configuration Domain.
"""

from typing import Any, Optional
import math
import re
from collections import Counter

from . import Domain
from smt_instance import SMTInstance, load_smt_data
from feature_engine import Feature
from prompt_builder import load_prompt_template
from trainer.random_forest import RandomForestTrainer

DataPoint = SMTInstance


class SMTSolver(Domain):
    def __init__(self):
        self._split_prompt_template = load_prompt_template("prompts/smt_split.txt")
        self._funsearch_prompt_template = load_prompt_template("prompts/smt_funsearch.txt")

    def domain_name(self) -> str:
        return "smt_solver"

    def load_dataset(self, path: str, max_size: int) -> list[DataPoint]:
        return load_smt_data(path, max_instances=max_size)

    def input_of(self, dp: DataPoint) -> Any:
        """The input that features operate on - the SMT benchmark string."""
        return dp.benchmark

    def label_of(self, dp: DataPoint) -> set:
        """The label we're trying to predict - set of option sets that solve this benchmark."""
        return dp.option_sets

    def leaf_prediction(self, datapoints: list[DataPoint]) -> str:
        """
        Return the option set that solves the most instances.
        Takes the union of all option sets and returns the most frequent one.
        """
        if not datapoints:
            return ""
        
        # Collect all option sets from all instances
        all_option_sets = []
        for dp in datapoints:
            all_option_sets.extend(dp.option_sets)
        
        if not all_option_sets:
            return ""
        
        # Return the most common option set
        counter = Counter(all_option_sets)
        return counter.most_common(1)[0][0]

    def leaf_error(self, datapoints: list[DataPoint]) -> float:
        """
        Return the fraction of instances that don't contain the predicted option set.
        """
        if not datapoints:
            return 0.0
        
        predicted_option_set = self.leaf_prediction(datapoints)
        
        # Count how many instances contain this option set
        correct = sum(1 for dp in datapoints if predicted_option_set in dp.option_sets)
        
        return 1.0 - (correct / len(datapoints))

    def prediction_error(self, pred: Any, label: Any) -> float:
        """
        Error is 1.0 if predicted option set is not in the label set, 0.0 if it is.
        """
        return 0.0 if pred in label else 1.0

    def code_execution_namespace(self) -> dict[str, Any]:
        """Provide namespace for feature code execution."""
        return {
            "math": math,
            "re": re,
            "len": len,
            "str": str,
            "float": float,
            "int": int,
        }

    def best_split_for_feature(
        self,
        examples: list[DataPoint],
        feature: Feature,
        min_side_ratio: float,
    ) -> tuple[Optional[Feature], float, list[DataPoint], list[DataPoint], float]:
        """Format prompt for D-ID3 feature generation. TODO: implement later."""
        raise NotImplementedError("Prompt formatting not yet implemented")


    def format_split_prompt(
        self,
        n_output_features: int,
        examples: list[Any],
        split_context: Optional[str],
    ) -> str:
        """Format prompt for generating features to split a node in the decision tree."""
        
        # TODO (Future): Add structured SMT-LIB parsing to enable more sophisticated features
        # TODO (Future): Add domain-specific knowledge about solver strategies
        
        api = """
## Available Python API

When writing features, you have access to:
- `benchmark`: The SMT-LIB benchmark as a string
- `math` module: Standard mathematical operations
- `re` module: Regular expressions for pattern matching
- Standard Python string methods: `.count()`, `.find()`, `.split()`, etc.

Example usage:
```python
import re
import math

def feature(benchmark: str) -> float:
    # Count nested quantifiers using regex
    pattern = r'\\(forall.*\\(forall'
    return float(len(re.findall(pattern, benchmark)))
```
"""
        
        def format_instance(inst: SMTInstance) -> str:
            """Format an SMT instance for the prompt."""
            option_sets_str = ", ".join(sorted(inst.option_sets))
            benchmark_preview = inst.benchmark[:300] + "..." if len(inst.benchmark) > 300 else inst.benchmark
            return f"Benchmark preview:\n{benchmark_preview}\n\nOption sets that solve it: {option_sets_str}\n" + "="*60 + "\n"
        
        examples_str = "\n".join([format_instance(ex) for ex in examples[:5]])  # Show first 5 examples
        
        context_str = split_context if split_context else "Root node (all training data)"
        
        return self._split_prompt_template.format(
            api_description=api,
            subtree_path=context_str,
            examples=examples_str,
            num_features=n_output_features,
        )

    def format_funsearch_prompt(
        self,
        n_output_features: int,
        existing_features_with_importances: list[tuple[Feature, float]],
    ) -> str:
        """Format prompt for FunSearch iterative feature generation."""
        
        # TODO (Future): Add structured SMT-LIB parsing to enable more sophisticated features
        # TODO (Future): Add domain-specific knowledge about solver strategies
        
        api = """
Available Python API:
- `benchmark: str` - The SMT-LIB benchmark
- `math` module - Mathematical operations
- `re` module - Regular expressions
- Standard Python string operations
"""
        
        def format_feature_with_importance(f: Feature, importance: float) -> str:
            """Format a feature with its importance score."""
            return f"Importance: {importance:.3f}\n{f.code}\n" + "-"*60
        
        if existing_features_with_importances:
            features_str = "\n\n".join([
                format_feature_with_importance(f, imp)
                for f, imp in existing_features_with_importances
            ])
        else:
            features_str = "<No existing features yet - this is the first iteration>"
        
        return self._funsearch_prompt_template.format(
            api_description=api,
            num_features=n_output_features,
            features=features_str,
        )

    def train_and_evaluate_simple_predictor(
        self,
        all_features: list[Feature],
        training_set: list[DataPoint],
        validation_set: list[DataPoint],
        training_parameters: dict[str, Any] = {},
    ) -> tuple[Any, float, float]:
        """Train a random forest on the features."""
        
        trainer = RandomForestTrainer(
            features_spec={"features": [f.code for f in all_features]},
            task_type="classification",
            domain_name=self.domain_name(),
            model_type="base_predictor",
            **training_parameters,
        )

        model, metrics = trainer.train(training_set, validation_set, None)
        
        # Return accuracy as (1 - error rate)
        train_error = 1.0 - metrics["train"]["accuracy"]
        valid_error = 1.0 - metrics["valid"]["accuracy"]
        
        return model, train_error, valid_error

import pytest
from typing import Dict, Any, List
from tunalab.evaluation import EvaluationRunner, register_handler

# --- Mocks and Helpers ---

class MockModel:
    def __init__(self):
        self.call_count = 0

    @register_handler("mock_eval")
    def evaluate_mock(self, batch: List[int]):
        self.call_count += 1
        # Return squared values as "predictions"
        return [x * x for x in batch]

    def some_other_method(self):
        pass

class MockEvaluationRunner(EvaluationRunner):
    def __init__(self, model: Any):
        super().__init__(model, evaluation_type="mock_eval")

    def _initialize_metrics(self) -> Dict[str, Any]:
        return {"sum_squares": 0, "count": 0}

    def _process_results_batch(
        self,
        batch: List[int],
        model_outputs: List[int]
    ) -> None:
        self.results["sum_squares"] += sum(model_outputs)
        self.results["count"] += len(batch)

    def _compute_final_metrics(self) -> Dict[str, Any]:
        return {
            "average_square": self.results["sum_squares"] / self.results["count"] if self.results["count"] > 0 else 0
        }

# --- Tests ---

def test_register_handler_decorator():
    """Test that the decorator correctly attaches the attribute."""
    model = MockModel()
    assert hasattr(model.evaluate_mock, "_evaluation_type")
    assert model.evaluate_mock._evaluation_type == "mock_eval"

def test_evaluation_runner_init_finds_handler():
    """Test that EvaluationRunner correctly finds the registered handler."""
    model = MockModel()
    runner = MockEvaluationRunner(model)
    assert runner.handler == model.evaluate_mock

def test_evaluation_runner_init_no_handler_raises_error():
    """Test that EvaluationRunner raises AttributeError if no handler is found."""
    class EmptyModel:
        pass
    
    model = EmptyModel()
    with pytest.raises(AttributeError) as excinfo:
        MockEvaluationRunner(model)
    
    assert "No handler registered" in str(excinfo.value)

def test_evaluation_runner_run_flow():
    """Test the full execution flow of the runner."""
    model = MockModel()
    runner = MockEvaluationRunner(model)
    
    dataset = [1, 2, 3, 4, 5]
    metrics = runner.run(dataset, batch_size=2)
    
    # Expected sum of squares: 1+4+9+16+25 = 55
    # Count: 5
    # Average: 11.0
    assert metrics["average_square"] == 11.0
    
    # Check batching behavior (5 items, batch_size 2 -> 3 batches)
    assert model.call_count == 3

def test_evaluation_runner_run_with_limit():
    """Test that the limit parameter is respected."""
    model = MockModel()
    runner = MockEvaluationRunner(model)
    
    dataset = [1, 2, 3, 4, 5]
    metrics = runner.run(dataset, batch_size=2, limit=3)
    
    # Processed: 1, 2, 3
    # Squares: 1, 4, 9 -> Sum: 14
    # Count: 3
    # Average: 14/3 = 4.666...
    assert metrics["average_square"] == pytest.approx(14/3)
    
    # Batch size 2, limit 3 -> 2 batches (first batch of 2, second batch of 1)
    assert model.call_count == 2


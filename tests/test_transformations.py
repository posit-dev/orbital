"""Test the transformations module functionality."""

import pytest
import ibis
import math

from orbital.translation.transformations import apply_post_transform
from orbital.translation.variables import ValueVariablesGroup, NumericVariablesGroup


def test_apply_post_transform_single_value():
    """Test applying transformations to a single numeric value."""
    # Create a simple numeric value
    value = ibis.literal(2.0)

    # Test NONE transformation (identity)
    result = apply_post_transform(value, "NONE")
    assert isinstance(result, ibis.expr.types.NumericValue)

    # Test LOGISTIC transformation
    result = apply_post_transform(value, "LOGISTIC")
    assert isinstance(result, ibis.expr.types.NumericValue)


def test_apply_post_transform_variables_group():
    """Test applying transformations to a VariablesGroup."""
    # Create a VariablesGroup with numeric values
    group_data = {
        "class_0": ibis.literal(1.0),
        "class_1": ibis.literal(-0.5),
        "class_2": ibis.literal(2.0),
    }
    variables_group = ValueVariablesGroup(group_data)

    # Test NONE transformation
    result = apply_post_transform(variables_group, "NONE")
    assert isinstance(result, NumericVariablesGroup)
    assert len(result) == 3
    assert "class_0" in result
    assert "class_1" in result
    assert "class_2" in result

    # Test LOGISTIC transformation
    result = apply_post_transform(variables_group, "LOGISTIC")
    assert isinstance(result, NumericVariablesGroup)
    assert len(result) == 3


def test_apply_post_transform_invalid_transform():
    """Test that invalid transformations raise NotImplementedError."""
    value = ibis.literal(1.0)

    with pytest.raises(
        NotImplementedError, match="Post transform 'INVALID' is not implemented"
    ):
        apply_post_transform(value, "INVALID")


def test_apply_post_transform_preserves_group_type():
    """Test that the function preserves the exact type of VariablesGroup."""
    # Create a NumericVariablesGroup
    group_data = {
        "value1": ibis.literal(1.0),
        "value2": ibis.literal(2.0),
    }
    numeric_group = NumericVariablesGroup(group_data)

    # Apply transformation
    result = apply_post_transform(numeric_group, "LOGISTIC")

    # Check that the result is still a NumericVariablesGroup
    assert isinstance(result, NumericVariablesGroup)
    assert len(result) == 2


def test_logistic_transform_computes_correct_values():
    """Test that LOGISTIC transformation computes the expected sigmoid values."""
    # Test with a simple DuckDB backend to compute actual values
    backend = ibis.duckdb.connect()

    # Test LOGISTIC transformation on single value
    test_value = 2.0
    input_expr = ibis.literal(test_value)
    logistic_result = apply_post_transform(input_expr, "LOGISTIC")

    # Compute the actual result
    computed_value = backend.execute(logistic_result)
    expected_value = 1 / (1 + math.exp(-test_value))  # sigmoid function

    assert abs(computed_value - expected_value) < 1e-10, (
        f"LOGISTIC: expected {expected_value}, got {computed_value}"
    )

    # Test LOGISTIC transformation on NumericVariablesGroup
    test_values = {"class_0": 1.0, "class_1": -0.5, "class_2": 0.0}
    group_data = {name: ibis.literal(val) for name, val in test_values.items()}
    numeric_group = NumericVariablesGroup(group_data)

    logistic_group = apply_post_transform(numeric_group, "LOGISTIC")

    # Check each value in the group
    for name, expected_input in test_values.items():
        computed_output = backend.execute(logistic_group[name])
        expected_output = 1 / (1 + math.exp(-expected_input))

        assert abs(computed_output - expected_output) < 1e-10, (
            f"LOGISTIC group {name}: expected {expected_output}, got {computed_output}"
        )


def test_none_transform_computes_correct_values():
    """Test that NONE transformation returns identity values (unchanged)."""
    # Test with a simple DuckDB backend to compute actual values
    backend = ibis.duckdb.connect()

    # Test NONE transformation on single value
    test_value = 2.0
    input_expr = ibis.literal(test_value)
    none_result = apply_post_transform(input_expr, "NONE")
    computed_none = backend.execute(none_result)

    assert computed_none == test_value, (
        f"NONE: expected {test_value}, got {computed_none}"
    )

    # Test NONE transformation on NumericVariablesGroup
    test_values = {"class_0": 1.0, "class_1": -0.5, "class_2": 0.0}
    group_data = {name: ibis.literal(val) for name, val in test_values.items()}
    numeric_group = NumericVariablesGroup(group_data)

    none_group = apply_post_transform(numeric_group, "NONE")

    # Check each value in the group remains unchanged
    for name, expected_value in test_values.items():
        computed_value = backend.execute(none_group[name])

        assert computed_value == expected_value, (
            f"NONE group {name}: expected {expected_value}, got {computed_value}"
        )


def test_softmax_transform_computes_correct_values():
    """Test that SOFTMAX transformation computes the expected normalized values."""
    # Test with a simple DuckDB backend to compute actual values
    backend = ibis.duckdb.connect()

    # Test SOFTMAX transformation on single value (should return 1.0)
    test_value = 2.0
    input_expr = ibis.literal(test_value)
    softmax_result = apply_post_transform(input_expr, "SOFTMAX")

    computed_value = backend.execute(softmax_result)
    assert computed_value == 1.0, (
        f"SOFTMAX single value: expected 1.0, got {computed_value}"
    )

    # Test SOFTMAX transformation on NumericVariablesGroup
    test_values = {"class_0": 1.0, "class_1": 2.0, "class_2": 3.0}
    group_data = {name: ibis.literal(val) for name, val in test_values.items()}
    numeric_group = NumericVariablesGroup(group_data)

    softmax_group = apply_post_transform(numeric_group, "SOFTMAX")

    # Check that the result is a NumericVariablesGroup
    assert isinstance(softmax_group, NumericVariablesGroup)
    assert len(softmax_group) == 3

    # Compute actual softmax values manually for verification
    import math

    values = list(test_values.values())
    max_val = max(values)
    exp_values = [math.exp(v - max_val) for v in values]
    sum_exp = sum(exp_values)
    expected_softmax = [exp_val / sum_exp for exp_val in exp_values]

    # Check each computed value
    computed_values = []
    for name in ["class_0", "class_1", "class_2"]:
        computed_val = backend.execute(softmax_group[name])
        computed_values.append(computed_val)

    # Verify the computed softmax values match expected
    for i, (name, expected) in enumerate(
        zip(["class_0", "class_1", "class_2"], expected_softmax)
    ):
        assert abs(computed_values[i] - expected) < 1e-10, (
            f"SOFTMAX group {name}: expected {expected}, got {computed_values[i]}"
        )

    # Verify that the softmax values sum to 1.0
    total_sum = sum(computed_values)
    assert abs(total_sum - 1.0) < 1e-10, (
        f"SOFTMAX values should sum to 1.0, got {total_sum}"
    )

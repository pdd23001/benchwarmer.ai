"""
Tests for sandbox execution and Plot Agent tool functions.

These test the *deterministic* sandbox logic without any API calls.
"""

import os

import pandas as pd
import pytest

from benchwarmer.utils.sandbox import execute_plot_code


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a small sample benchmark results DataFrame."""
    return pd.DataFrame({
        "algorithm_name": ["greedy", "greedy", "random", "random"],
        "instance_name": ["inst_0", "inst_1", "inst_0", "inst_1"],
        "instance_generator": ["erdos_renyi", "erdos_renyi", "erdos_renyi", "erdos_renyi"],
        "problem_size": [50, 100, 50, 100],
        "objective_value": [120.0, 250.0, 95.0, 180.0],
        "wall_time_seconds": [0.01, 0.05, 0.005, 0.02],
        "peak_memory_mb": [1.2, 2.5, 1.0, 2.0],
        "status": ["success", "success", "success", "success"],
        "run_index": [0, 0, 0, 0],
        "feasible": [True, True, True, True],
    })


class TestSandbox:
    def test_simple_plot(self, sample_df, tmp_path):
        """Test that a simple plot is generated and saved."""
        code = """
fig, ax = plt.subplots()
ax.bar(df['algorithm_name'].unique(), df.groupby('algorithm_name')['objective_value'].mean())
ax.set_title('Test Plot')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
"""
        result = execute_plot_code(code, sample_df, str(tmp_path), 0)
        assert result["success"] is True
        assert result["output_path"] is not None
        assert os.path.exists(result["output_path"])

    def test_table_output(self, sample_df, tmp_path):
        """Test that code producing stdout (like tables) works."""
        code = """
summary = df.groupby('algorithm_name')['objective_value'].mean()
print(summary.to_string())
"""
        result = execute_plot_code(code, sample_df, str(tmp_path), 0)
        assert result["success"] is True

    def test_error_handling(self, sample_df, tmp_path):
        """Test that code errors are caught and reported."""
        code = """
# This will raise a NameError
undefined_variable + 1
"""
        result = execute_plot_code(code, sample_df, str(tmp_path), 0)
        assert result["success"] is False
        assert "error" in result
        assert "traceback" in result

    def test_df_is_copied(self, sample_df, tmp_path):
        """Test that the sandbox gets a copy of the DataFrame."""
        code = """
df['new_column'] = 999
"""
        result = execute_plot_code(code, sample_df, str(tmp_path), 0)
        assert result["success"] is True
        assert "new_column" not in sample_df.columns  # original untouched

    def test_auto_save_figure(self, sample_df, tmp_path):
        """Test that unsaved figures are automatically saved."""
        code = """
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
ax.set_title('Auto-saved')
# No explicit plt.savefig — sandbox should auto-save
"""
        result = execute_plot_code(code, sample_df, str(tmp_path), 0)
        assert result["success"] is True
        assert result["output_path"] is not None
        assert os.path.exists(result["output_path"])

    def test_namespace_restrictions(self, sample_df, tmp_path):
        """Test that dangerous builtins are not available."""
        code = """
# open() should not be in the namespace
result = open('/etc/passwd', 'r')
"""
        result = execute_plot_code(code, sample_df, str(tmp_path), 0)
        # Should fail because open() is not in the restricted namespace
        assert result["success"] is False

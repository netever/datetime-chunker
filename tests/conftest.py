import pandas as pd
import pytest

from chunker import generate_frame

LARGE_ROWS = 1_000_000


@pytest.fixture
def doc_example_df() -> pd.DataFrame:
    """Пример из условия задачи."""
    return pd.DataFrame(
        {
            "dt": pd.to_datetime(
                [
                    "2023-01-01 00:00:01",
                    "2023-01-01 00:00:01",
                    "2023-01-01 00:00:02",
                    "2023-01-01 00:00:02",
                    "2023-01-01 00:00:02",
                    "2023-01-01 00:00:03",
                ]
            )
        }
    )


@pytest.fixture(scope="session")
def large_sorted_df() -> pd.DataFrame:
    return generate_frame(LARGE_ROWS, max_repeats=5, seed=2023)


@pytest.fixture(scope="session")
def large_shuffled_df() -> pd.DataFrame:
    return generate_frame(LARGE_ROWS, max_repeats=5, seed=2023, shuffle=True)

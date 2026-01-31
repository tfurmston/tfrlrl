from dataclasses import dataclass


class StatisticsException(Exception):
    """Custom exception to encompass errors raised during statistics collection."""

    pass


@dataclass
class BaseStatistics:
    """A base class for statistics collected from a statistics collector."""

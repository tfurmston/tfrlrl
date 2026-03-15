from typing import List

import numpy as np
import numpy.typing as npt

from tfrlrl.data_models.statistics import BaseStatistics, StatisticsException


def merge_optional_statistics(
    statistics: List[BaseStatistics], attribute: str, concatenate_axis: int = 0
) -> npt.ArrayLike:
    """
    Merge the optional statistic from the given list of statistics and merge them.

    Args:
        statistics: A list of statistics from which to retrieve an optional statistics and merge them.
        attribute: The (optional) statistic from the statistics class that is to be retrieved and merged.
        concatenate_axis: The axis along which to merge the statistic.

    Returns:
        The merged statistics or None.

    Raises:
        This function raises a StatisticsException exception if some, but not all, of the statistics are present.

    """
    attributes = [getattr(x, attribute) for x in statistics]
    if any([x is None for x in attributes]) and any([x is not None for x in attributes]):
        raise StatisticsException('All baseline features should either be None or a NumPy array.')
    if all([x is not None for x in attributes]):
        attributes = np.concatenate(attributes, axis=concatenate_axis)
    else:
        attributes = None
    return attributes

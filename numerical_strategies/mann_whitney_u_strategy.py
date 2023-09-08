from scipy.stats import mannwhitneyu
from numerical_strategies.numerical_strategy import NumericalStrategy
import pandas as pd

class MannWhitneyUStrategy(NumericalStrategy):
    """
    A strategy class implementing the Mann-Whitney U Test for numerical data.

    The Mann-Whitney U Test strategy calculates the p-value using the Mann-Whitney U Test,
    a non-parametric test, to assess the significance of differences between two sets of numerical data:
    'good_values' representing a reference distribution (typically, a well-behaved dataset),
    and 'bad_values' representing a potentially problematic dataset.

    The Mann-Whitney U Test is particularly useful when the assumptions of normality or equal variance
    are not met, making it suitable for analyzing non-normally distributed or skewed data.

    Parameters
    ----------
    good_values : pd.Series
        Series of values from the reference dataset.
    bad_values : pd.Series
        Series of values from the potentially problematic dataset.

    Returns
    -------
    float
        The calculated p-value.

    Explanation of the Mann-Whitney U Test process:
    ----------------------------------------------
    - Rank all data points from both 'good_values' and 'bad_values' datasets together.
    - Calculate the sum of ranks for each dataset.
    - Calculate the U statistic for the Mann-Whitney U Test, which is based on the smaller of the two
      sum of ranks.
    - The p-value is then calculated using statistical methods that relate the U statistic to the distribution
      of U under the null hypothesis (no difference between distributions).
    - A small p-value suggests that the distributions are significantly different.

    Example of the test:
    ---------------------
    Let's say we perform the Mann-Whitney U Test on exam scores across two groups.
    The 'good_data' contains scores from students following a standard curriculum,
    while the 'bad_data' contains scores from students following an alternative curriculum.

    Example usage:

        example_good_data = pd.Series([75, 82, 88, 95, 65])
        example_bad_data = pd.Series([60, 70, 72, 78, 68])
        mannwhitneyu_test = MannWhitneyUStrategy()
        p_value = mannwhitneyu_test.calculate_probability(example_good_data, example_bad_data)

        The p_value is 0.15079365079365079, suggesting a significant difference between the distributions.

    """

    def calculate_probability(self, good_values: pd.Series, bad_values: pd.Series) -> float:
        _, p_value = mannwhitneyu(good_values, bad_values, alternative='two-sided')
        return p_value

import pandas as pd
from scipy.stats import ttest_ind
from numerical_strategies.numerical_strategy import NumericalStrategy
class TTestStrategy(NumericalStrategy):
    """
    A strategy class implementing the Welch's t-test statistical test for numerical data.

    The Welch's t-test strategy calculates the p-value using the Welch's independent two-sample t-test to
    assess the significance of differences between two sets of numerical data: 'good_values'
    representing a reference distribution (typically, a well-behaved dataset), and 'bad_values'
    representing a potentially problematic dataset.

    The Welch's t-test is particularly useful when the variances of the two datasets are not assumed to be equal.
    It calculates a modified t-statistic that takes into account the differences in variances,
    providing more accurate results compared to the standard independent two-sample t-test.

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

    Explanation of the Welch's t-test process:
    -----------------------------------------
    - Calculate the means and variances of both 'good_values' and 'bad_values' datasets.
    - Calculate the difference in means (mean_bad - mean_good) and the standard error (SE) of the difference.
      SE is calculated using the formula: SE = sqrt(var_good/n_good + var_bad/n_bad),
      where var_good and var_bad are the variances of the datasets,
      and n_good and n_bad are the sample sizes of the datasets.
    - Compute the t-statistic as t_statistic = (mean_bad - mean_good) / SE.
      The t-statistic measures the difference between means in terms of standard errors.
    - Degrees of freedom are determined based on the sample sizes and used to shape the t-distribution.
    - Calculate the p-value by finding the probability of observing a t-statistic as extreme as the one calculated
      (or more extreme) under the assumption that the null hypothesis is true (means are equal).
    - For a two-sided test, find the area under the t-distribution curve beyond the absolute value of the observed
      t-statistic to determine the p-value.
    - A smaller p-value suggests that the observed data is less likely under the null hypothesis.
      It indicates a larger difference between the means.

    Example of the test:
    ---------------------
    Let's say we perform the Welch's t-test on file sizes across both groups.
    The 'good_data' contains file sizes that are expected to be reasonable,
    while the 'bad_data' contains file sizes that are potentially problematic.

    Example usage:

        example_good_data = pd.Series([20, 22, 21, 23, 18])
        example_bad_data = pd.Series([35, 32, 40, 30, 38])
        t_test = TTestStrategy()
        p_value = t_test.calculate_probability(example_good_data, example_bad_data)

        the p_value is 0.0005528697203011475, this is very low and suggest that the good_data is
        wildly different from the bad_data

        example_good_data = pd.Series([20, 22, 21, 23, 18])
        example_bad_data = pd.Series([19, 20, 21, 19, 22])
        t_test = TTestStrategy()
        p_value = t_test.calculate_probability(example_good_data, example_bad_data)

        the p_value is 0.5817008764029983, this is high, and suggest that the good_data is
        mostly the same as the bad data

    """

    def calculate_probability(self, good_values: pd.Series, bad_values: pd.Series) -> float:
        t_statistic, p_value = ttest_ind(good_values, bad_values, equal_var=False)  # Using equal_var=False for Welch's t-test
        return p_value

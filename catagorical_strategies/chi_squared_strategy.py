import pandas as pd
from scipy.stats import chi2_contingency
from categorical_strategy import CategoricalStrategy

class ChiSquaredStrategy(CategoricalStrategy):
    """
    A strategy class implementing the Chi-squared test for categorical data.

    The Chi-squared strategy calculates the p-value using the Chi-squared test to
    assess the independence between two categorical variables: 'good_categories'
    representing a reference distribution (typically, a well-behaved dataset),
    and 'bad_categories' representing a potentially problematic dataset.

    The Chi-squared test measures whether there's a significant association between
    the two categorical variables. A low p-value suggests that the two variables are
    dependent, while a high p-value suggests they are independent.

    Parameters
    ----------
    good_categories : pd.Series
        Series of categories from the reference dataset.
    bad_categories : pd.Series
        Series of categories from the potentially problematic dataset.

    Returns
    -------
    float
        The calculated p-value.

    Explanation of the Chi-squared test process:
    --------------------------------------------
    - Create a contingency table that cross-tabulates the two categorical variables.
    - Calculate the expected frequencies for each cell in the contingency table assuming
      that the variables are independent.
    - Calculate the Chi-squared statistic as the sum of squared differences between observed
      and expected frequencies, normalized by the expected frequencies.
    - Degrees of freedom are determined based on the dimensions of the contingency table.
    - Calculate the p-value by finding the probability of observing a Chi-squared statistic as
      extreme as the one calculated (or more extreme) under the assumption of independence.
    - A smaller p-value suggests that the observed data is less likely under the assumption
      of independence.

    Example of the test:
    ---------------------
    Let's say we perform the Chi-squared test on one categorical variable:
    'region'
    representing the region of customers

    Example usage:

        example_good_categories = pd.Series(['North', 'South', 'East', 'West', 'North'])
        example_bad_categories = pd.Series(['South', 'South', 'East', 'East', 'West'])
        chi_squared_test = ChiSquaredStrategy()
        p_value = chi_squared_test.calculate_probability(example_good_categories, example_bad_categories)

        The p_value is 0.7376674784345812, which is high, suggesting that the variables
        'region' and 'purchase_status' are likely independent.

        example_good_categories = pd.Series(['Yes', 'No', 'No', 'Yes', 'Yes'])
        example_bad_categories = pd.Series(['No', 'No', 'No', 'Yes', 'Yes'])
        chi_squared_test = ChiSquaredStrategy()
        p_value = chi_squared_test.calculate_probability(example_good_categories, example_bad_categories)

        The p_value is 0.014387678176921308, which is low, suggesting that the variables
        'purchase_status' have a significant association with the 'region'.

    """

    def calculate_probability(self, good_categories: pd.Series, bad_categories: pd.Series) -> float:
        contingency_table = pd.crosstab(good_categories, bad_categories)
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        return p_value



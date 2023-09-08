import scipy.stats as stats
import pandas as pd
from collections import Counter
from typing import List, Any
from numerical_strategies import NumericalStrategy
from matplotlib import pyplot as plt, cm


class ErrorAnalyzer:
    MAX_PROBABILITY = 0.99999
    MISSING_BAD_DATA = ("Missing in bad_data", MAX_PROBABILITY, True, "Missing in bad_data")
    ADDITIONAL_BAD_DATA = ("Additional in bad_data", MAX_PROBABILITY, True, "Additional in bad_data")

    def __init__(self, good_data: pd.DataFrame, bad_data: pd.DataFrame, fields_to_ignore: List[str] = None, numerical_strategy: NumericalStrategy = None):
        self.good_data = good_data
        self.bad_data = bad_data
        self.fields_to_analyze = self.good_data.columns
        self.fields_to_ignore = fields_to_ignore if fields_to_ignore else []
        self.numerical_strategy = numerical_strategy

    def _calculate_numerical_probability(self, good_values: pd.Series, bad_values: pd.Series) -> float:
        return self.numerical_strategy.calculate_probability(good_values, bad_values)

    def calculate_effect_size(self, field: str) -> float:
        if field in self.fields_to_ignore:
            return 0.0  # Effect size not calculated for ignored fields

        probabilities = self._calculate_probabilities()
        if field not in probabilities:
            return 0.0  # Effect size not calculated if field is not analyzed

        p_value, _, _ = probabilities[field]
        effect_size = self._calculate_effect_size(p_value, field)
        return effect_size

    def _calculate_effect_size(self, p_value: float, field: str) -> float:
        good_values = self.good_data[field]
        bad_values = self.bad_data[field]

        good_values_mean = good_values.mean()
        bad_values_mean = bad_values.mean()

        good_values_std = good_values.std()
        bad_values_std = bad_values.std()

        pooled_std = ((len(good_values) - 1) * good_values_std ** 2 +
                      (len(bad_values) - 1) * bad_values_std ** 2) / \
                     (len(good_values) + len(bad_values) - 2)

        cohen_d = abs(good_values_mean - bad_values_mean) / pooled_std
        return cohen_d

    def adjust_p_values(self, p_values: List[float], adjustment_method: str = 'bonferroni') -> List[float]:
        if adjustment_method == 'bonferroni':
            adjusted_p_values = self._apply_bonferroni_correction(p_values)
        else:
            raise ValueError(f"Adjustment method '{adjustment_method}' is not supported.")

        return adjusted_p_values

    def _apply_bonferroni_correction(self, p_values: List[float]) -> List[float]:
        adjusted_p_values = [p_value * len(p_values) for p_value in p_values]
        adjusted_p_values = [min(p_value, 1.0) for p_value in adjusted_p_values]  # Cap at 1.0
        return adjusted_p_values

    def run(self) -> pd.DataFrame:
        """
        Run the error analysis and return results in a DataFrame.

        Returns:
        - result_df (pd.DataFrame): DataFrame containing error analysis results.
        """
        results = []
        probabilities = self._calculate_probabilities()

        for field, (p_value, is_additional, details) in probabilities.items():
            probability = self._calculate_probability(p_value, is_additional)
            if field not in self.fields_to_ignore:
                result = {
                    "field": field,
                    "probability": probability,
                    "extra_status": is_additional,
                    "details": details
                }
                results.append(result)

        result_df = pd.DataFrame(results)
        return result_df

    def _calculate_probabilities(self) -> dict[Any, tuple[str, float, bool, str] | tuple[float, bool, None]]:
        """
        Calculate probabilities for each field.

        Returns:
        - probabilities (dict): Dictionary of field probabilities.
        """
        probabilities = {}

        for field in self.good_data.columns:
            if field in self.fields_to_ignore:
                # Field is in the ignore list, so skip it
                continue

            if field not in self.bad_data.columns:
                probabilities[field] = self.MISSING_BAD_DATA
                continue

            good_field_values = self.good_data[field]
            bad_field_values = self.bad_data[field]

            if good_field_values.apply(lambda x: isinstance(x, (int, float))).all():
                p_value = self._calculate_numerical_probability(good_field_values, bad_field_values)
                probabilities[field] = (p_value, False, None)
            elif good_field_values.apply(lambda x: isinstance(x, str)).all():
                p_value = self._calculate_categorical_probability(good_field_values, bad_field_values)
                probabilities[field] = (p_value, False, None)

        for field in self.bad_data.columns:
            if field not in self.good_data.columns:
                if field not in self.fields_to_ignore:
                    probabilities[field] = self.ADDITIONAL_BAD_DATA

        return probabilities

    @staticmethod
    def _calculate_categorical_probability(good_values: pd.Series, bad_values: pd.Series) -> float:
        """
        Calculate categorical probability using chi-squared test.

        Parameters:
        - good_values (pd.Series): Series of good data values.
        - bad_values (pd.Series): Series of bad data values.

        Returns:
        - p_value (float): Calculated p-value.
        """
        contingency_table = pd.crosstab(good_values, bad_values)
        chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
        return p_value

    def _calculate_probability(self, p_value, is_additional):
        alpha = 0.05  # Move threshold to a class parameter
        if is_additional:
            probability = self.MAX_PROBABILITY
        else:
            if p_value < alpha:
                probability = 1 - p_value
            else:
                probability = 0
        return probability * 100

    def create_graphs(self, save_path: str = None):
        """
        Create a graph with all fields' error analysis results.

        Parameters:
        - save_path (str): Path to save the graph image. If None, the graph will be displayed.

        Returns:
        - None
        """
        result_df = self.run()

        plt.style.use('dark_background')

        # Define colors based on probability values
        norm = plt.Normalize(0, 100)
        colors = cm.RdYlGn_r(norm(result_df['probability']))

        # Create a new array to hold colors based on status
        status_colors = [
            colors[i] if not is_missing else (0, 0, 1)
            for i, is_missing in enumerate(result_df['extra_status'])
        ]

        plt.figure(figsize=(12, 8))

        # Plot missing fields with a distinct color
        missing_fields = set(self.fields_to_analyze) - set(result_df['field']) - set(self.fields_to_ignore)
        for field in missing_fields:
            plt.bar(field, 100, color=(0.5, 0.5, 0.5), alpha=0.2)

        # Plot analyzed fields
        bars = plt.bar(result_df['field'], result_df['probability'], color=status_colors)

        # Add probability values as labels at the bottom of the bars
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width() / 2, 0, f'{bar.get_height():.2f}%', ha='center', va='bottom',
                     color='black')

        # Set plot labels and limits
        plt.title('Error Analysis for All Fields')
        plt.xlabel('Fields')
        plt.ylabel('Probability')
        plt.ylim(0, 100)
        plt.xticks(rotation=45, ha='right')

        # Create a colorbar legend
        sm = plt.cm.ScalarMappable(cmap=cm.RdYlGn, norm=norm)
        sm.set_array([])  # Empty array to create the colorbar
        cbar = plt.colorbar(sm, orientation='vertical')
        cbar.set_label('Probability')

        # Check if there's missing or additional data to decide whether to add the legend
        if any(result_df['extra_status']):
            legend_labels = ['Missing/Additional Data']
            legend_colors = [(0, 0, 1)]  # Blue for missing, Gray for additional
            plt.legend(handles=[plt.Line2D([0], [0], color=color, label=label)
                                for color, label in zip(legend_colors, legend_labels)])

        if save_path:
            plt.savefig(save_path)
            print(f"Graph saved at '{save_path}'.")
        else:
            plt.show()

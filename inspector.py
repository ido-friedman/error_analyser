import pandas as pd
from numerical_strategies import TTestStrategy
from analyser import ErrorAnalyzer
import random


# Example usage
def generate_random_data(num_samples=10000):
    data = []
    colors = ["green"] * 18 + ["yellow"] * 6 + ["red"] * 1

    for _ in range(num_samples):
        size = random.randint(100, 600)
        color = random.choice(colors)
        weight = random.randint(1, 5)
        data.append({"size": size, "color": color, "weight": weight})

    return pd.DataFrame(data)


def generate_good_data(num_samples=10000):
    return generate_random_data(num_samples)


def generate_bad_data(num_samples=10000):
    data = generate_random_data(num_samples)
    # Introduce some "bad" data by modifying values
    for i in range(0, num_samples, 4):
        data.at[i, "color"] = "red"
    return data


# Generate the dataframes
good_data = generate_good_data()
bad_data = generate_bad_data()

print("good_data:\n", good_data)

print("bad_data:\n", bad_data)

analyzer = ErrorAnalyzer(good_data, bad_data, numerical_strategy=TTestStrategy())
result = analyzer.run()
print(result)


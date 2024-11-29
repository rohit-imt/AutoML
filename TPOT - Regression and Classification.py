# Install Libraries
#!pip install tpot 

# Classification Example

# Below is a minimal working example with the optical recognition of handwritten digits dataset.

from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    train_size=0.75, test_size=0.25, random_state=42)

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_digits_pipeline.py')


# Regression Example


from tpot import TPOTRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
# Load California housing dataset
housing = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target,
                                                    train_size=0.75, test_size=0.25, random_state=42)
# Initialize and train TPOTRegressor
tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)
tpot.fit(X_train, y_train)
# Evaluate the model
print(tpot.score(X_test, y_test))
# Export the pipeline
tpot.export('tpot_california_pipeline.py')


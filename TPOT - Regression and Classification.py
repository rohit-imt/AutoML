#!/usr/bin/env python
# coding: utf-8

# # TPOT Examples

# ## Classification

# In[1]:


# Install packages
get_ipython().system('pip install tpot')


# ### Below is a minimal working example with the optical recognition of handwritten digits dataset.

# In[2]:


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


# ## Regression

# ### Similarly, TPOT can optimize pipelines for regression problems. Below is a minimal working example with the practice Boston housing prices data set.

# In[4]:


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


from pyspark.ml.classification import LogisticRegression
import pyspark.ml.evaluation as evals
import pyspark.ml.tuning as tune
import numpy as np
from MachineLearningPipelines import training, test


# Create a LogisticRegression Estimator
lr = LogisticRegression()

# Create a BinaryClassificationEvaluator
evaluator = evals.BinaryClassificationEvaluator(metricName='areaUnderROC')

###### Make a Grid #######

# Create the parameter grid
grid = tune.ParamGridBuilder()

# Add the hyperparameter
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0, 1])

# Build the grid
grid = grid.build()

# Create the CrossValidator
cv = tune.CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator)

######## Fit the model(s)  ###########

# Call lr.fit()
best_lr = lr.fit(training)

# Print best_lr
print(best_lr)

###### Evaluate the model ########

""" The closer the AUC is to one (1), the better the model is!"""

# Use the model to predict the test set
test_results = best_lr.transform(test)

# Evaluate the predictions
print(evaluator.evaluate(test_results))
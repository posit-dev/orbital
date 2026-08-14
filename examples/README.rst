A few examples with significant test cases
that show how to use the orbital library.

scikit-learn (``pipeline_*.py``)
---------------------------------

- ``pipeline_lineareg.py`` -- Linear Regression
- ``pipeline_logisticreg.py`` -- multiclass Logistic Regression
- ``pipeline_lasso.py`` -- Lasso Regression
- ``pipeline_elasticnet.py`` -- Elastic Net Regression
- ``pipeline_decision_tree_classifier.py`` -- Decision Tree Classifier
- ``pipeline_decision_tree_regressor.py`` -- Decision Tree Regressor
- ``pipeline_randforest_classifier.py`` -- Random Forest Classifier
- ``pipeline_boosted_tree_classifier.py`` -- Gradient Boosted Tree multiclass Classifier
- ``pipeline_boosted_tree_binary_classifier.py`` -- Gradient Boosted Tree binary Classifier
- ``pipeline_boosted_tree_regressor.py`` -- Gradient Boosted Tree Regressor
- ``pipeline_mlp_classifier.py`` -- MLP binary Classifier (``MLPClassifier``)
- ``pipeline_mlp_regressor.py`` -- MLP Regressor (``MLPRegressor``, ``tanh`` activation)

PyTorch (``pytorch_*.py``)
---------------------------

- ``pytorch_fraud_detector.py`` -- binary classification (fraud detection)
- ``pytorch_maintenance_classifier.py`` -- multiclass classification (predictive maintenance)
- ``pytorch_demand_regressor.py`` -- regression (demand forecasting)

Other
-----

- ``minimal.py`` -- smallest possible pipeline
- ``simple_tree_regressor.py`` -- Decision Tree Regressor without ibis
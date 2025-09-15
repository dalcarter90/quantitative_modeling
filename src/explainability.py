import shap
import matplotlib.pyplot as plt
import numpy as np

class ExplainabilityModule:
    """
    Explainability module using SHAP for model interpretation.
    """
    def __init__(self, model, model_type='tree'):
        self.model = model
        self.model_type = model_type
        if model_type == 'tree':
            self.explainer = shap.TreeExplainer(model)
        else:
            self.explainer = shap.Explainer(model)

    def explain(self, X, plot=True, max_display=10):
        shap_values = self.explainer.shap_values(X)
        if plot:
            shap.summary_plot(shap_values, X, max_display=max_display, show=False)
            plt.tight_layout()
            plt.show()
        return shap_values

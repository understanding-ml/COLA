import pandas as pd
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import alibi

from xai_cola.data import BaseData
from xai_cola.ml_model.model import Model
from .base_explainer import CounterFactualExplainer



class AlibiCounterfactualInstances(CounterFactualExplainer):
    def __init__(self, ml_model: Model, data: BaseData=None):
        super().__init__(ml_model, data)

        '''
        ml_model: the Model used to predict
        data : BaseData -> input the data without target column
        sample_num : the number of your aim counterfactuals
        '''
     

    def generate_counterfactuals(
            self,
            data:BaseData=None,
            target_proba=0.51, 
            tolerance=0.1, 
            lam_init=1e-3, 
            max_iter=8000, 
            max_lam_steps=30
            ) -> np.ndarray:
        self.target_proba = target_proba
        self.tol = tolerance
        self.lam_init = lam_init
        self.max_iter = max_iter
        self.max_lam_steps = max_lam_steps
        # Call the data processing logic from the parent class
        self._process_data(data)
        
        x_chosen = self.x_factual_pandas

        # Generate counterfactual for each factual instance
        counterfactuals = []

        for i in range(len(x_chosen)):
            chosen = x_chosen.iloc[[i]]
            predicted_class = self.ml_model.predict(chosen)
            print(f'Predicted class for factual instance {i}: {predicted_class}')
            
            # set the target class to the opposite of the predicted class
            target_class = 1 if predicted_class == 0 else 0
            print(f'Target class for counterfactual instance {i}: {target_class}')
            
            # define the counterfactual explainer
            cf =alibi.explainers.Counterfactual(
                self.ml_model.predict_proba,      # The model to explain
                shape=(1,) + chosen.shape[1:],     # The shape of the model input
                target_proba=0.51,                  # The target class probability
                tol=self.tol,                       # The tolerance for the loss
                target_class='other',               # The target class to obtain  
                max_iter=self.max_iter,
                lam_init=self.lam_init,
                max_lam_steps=self.max_lam_steps,  # 增加 lambda 调整步数
            )
            # generate counterfactuals
            explanation = cf.explain(chosen.values)
            counterfactuals.append(explanation['cf']['X'])
            
        # Convert list of counterfactuals to a NumPy array
        counterfactuals = np.vstack(counterfactuals)

        return x_chosen.values, counterfactuals
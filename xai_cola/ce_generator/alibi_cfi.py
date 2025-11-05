"""

The ALIBI CounterfactualInstances is copid from https://docs.seldon.io/projects/alibi/en/latest/methods/CF.html 

"""
import pandas as pd
import numpy as np
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import alibi

from xai_cola.ce_sparsifier.data import COLAData
from xai_cola.ce_sparsifier.models import Model
from .base_explainer import CounterFactualExplainer


class AlibiCounterfactualInstances(CounterFactualExplainer):
    """
    Alibi-CFI algorithm: input 1 factual instance and output 1 counterfactual instance
    """
    def __init__(self, ml_model):
        """
        Initialize Alibi CFI explainer

        Parameters:
        -----------
        ml_model : Model
            Pre-trained model wrapped in Model interface
        """
        super().__init__(ml_model)
     

    def generate_counterfactuals(
            self,
            data:COLAData=None,
            feature_range=(-1e10, 1e10),
            max_iter=8000, 
            learning_rate_init=0.1,
            target_proba=1.0, 
            tolerance=0.05, 
            lam_init=1e-3, 
            max_lam_steps=30,
            factual_class=1, 
            ) -> np.ndarray:
        """Generate counterfactual
        
        Parameters:
        :feature_range: global or feature-wise min and max values for the perturbed instance.
        :max_iter: number of loss optimization steps for each value of; the multiplier of the distance loss term.
        :learning_rate_init: initial learning rate, follows linear decay.
        :target_prob: desired target probability for the returned counterfactual instance. Defaults to 1.0, but it could be useful to reduce it to allow a looser definition of a counterfactual instance.
        :tolerance: the tolerance within the target_proba, this works in tandem with target_proba to specify a range of acceptable predicted probability values for the counterfactual.
        :lam_init: initial value of the hyperparameter. This is set to a high value and annealed during the search to find good bounds for and for most applications should be fine to leave as default.
        :max_lam_steps: the number of steps (outer loops) to search for with a different value of.
        :factual_class: The target of your factual data. -> Normally, we set the factual_class=1, and we hope the prediction of counterfactual class=0.

        """

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
                # 1.general:
                self.ml_model.predict_proba,      # The model to explain
                shape=(1,) + chosen.shape[1:],     # The shape of the model input
                feature_range=feature_range,
                debug = False,

                # 2.related to the optimizer:
                max_iter=max_iter,
                learning_rate_init=learning_rate_init,
                early_stop=None,
                init='identity',

                # 3.related to the objective function:
                distance_fn = 'l1',
                target_proba=target_proba,    # The target class probability
                tol=tolerance,                       # The tolerance for the loss
                target_class=1-factual_class,      # The target class to obtain  
                lam_init=lam_init,
                max_lam_steps=max_lam_steps,  # 增加 lambda 调整步数
            )
            """ Parameters for alibi_explainer_CFI
            Parameters(general):
            :shape: shape of the instance to be explained, starting with batch dimension. Currently only single explanations are supported, so the batch dimension should be equal to 1.
            :feature_range: global or feature-wise min and max values for the perturbed instance.
            :write_dir: write directory for Tensorboard logging of the loss terms. It can be helpful when tuning the hyperparameters for your use case. It makes it easy to verify that e.g. not 1 loss term dominates the optimization, that the number of iterations is OK etc. You can access Tensorboard by running tensorboard --logdir {write_dir} in the terminal.
            :debug: flag to enable/disable writing to Tensorboard.
            
            Parameters(related to the optimizer):
            :max_iterations: number of loss optimization steps for each value of; the multiplier of the distance loss term.
            :learning_rate_init: initial learning rate, follows linear decay.
            :decay: flag to disable learning rate decay if desired 
            :early_stop: early stopping criterion for the search. If no counterfactuals are found for this many steps or if this many counterfactuals are found in a row we change accordingly and continue the search.
            :init: how to initialize the search, currently only "identity" is supported meaning the search starts from the original instance.

            Parameters(objective function):
            :distance_fn: distance function between the test instance and the proposed counterfactual, currently only "l1" is supported.
            :target_proba: desired target probability for the returned counterfactual instance. Defaults to 1.0, but it could be useful to reduce it to allow a looser definition of a counterfactual instance.
            :tol: the tolerance within the target_proba, this works in tandem with target_proba to specify a range of acceptable predicted probability values for the counterfactual.
            :target_class: desired target class for the returned counterfactual instance. Can be either an integer denoting the specific class membership or the string other which will find a counterfactual instance whose predicted class is anything other than the class of the test instance.
            :lam_init: initial value of the hyperparameter. This is set to a high value and annealed during the search to find good bounds for and for most applications should be fine to leave as default.
            :max_lam_steps: the number of steps (outer loops) to search for with a different value of.
            """

            # generate counterfactuals
            explanation = cf.explain(chosen.values)
            counterfactuals.append(explanation['cf']['X'])
            
        # Convert list of counterfactuals to a NumPy array
        counterfactuals = np.vstack(counterfactuals)

        return x_chosen.values, counterfactuals
from .ce_module.ce_models import DiCE
from .model import Model
from .data import Data
from .policy_module import policy
import numpy as np
import pandas as pd

class COLA:
    def __init__(self, ml_model=None, data=None, counterfactual_algorithm=None, policy_name="CF_OTMatch", Avalues_method="max", limited_actions=None):
        """
        Initialize the ACE_generator class

        Parameters:
        ml_model: Machine learning model used for predictions
        data: DataFrame containing the data
        counterfactual_algorithm: Algorithm to generate counterfactual data
        limited_actions: List of limited actions to consider
        """
        self.ml_model = ml_model
        self.data = data
        self.counterfactual_algorithm = counterfactual_algorithm
        self.limited_actions = limited_actions
        self.Avalues_method = Avalues_method
        self.policy_name = policy_name
        # self.ce_algorithm =None
        # self.results = self.generate_counterfactual()

    def generate_ace_counterfactual(self):
        
        # get model
        self.ml_model = self.ml_model.load_model()

        # get ce_algorithm
        ce_algorithm = self.counterfactualexplanation_algorithm()
        x_factual, y_factual, x_counterfactual, y_counterfactual = ce_algorithm.generate_x_counterfactuals()

        # get policy
        policies = policy.compute_intervention_policy(
                            model= self.ml_model,   
                            X_train=0,    # X_train=x_train didn't use the X_train in this part
                            X_factual=x_factual,
                            X_counterfactual=x_counterfactual,
                            shapley_method=self.policy_name,
                            Avalues_method=self.Avalues_method, # 'avg'
                        )
        varphi = policies['varphi']
        p = policies['p']
        q = policies['q']

        # 1. Find the top 'action' highest probability values and their positions in the varphi matrix
        flat_indices = np.argpartition(varphi.flatten(), -self.limited_actions)[-self.limited_actions:]
        row_indices, col_indices = np.unravel_index(flat_indices, varphi.shape)

        # 2. Find the values at these positions in q
        q_values = q[row_indices, col_indices]

        # 3. Replace the corresponding values in x_factual with the values found in q
        x_action_constrained = x_factual.copy()

        # 4. get the action-constrained CE
        for row_idx, col_idx, q_val in zip(row_indices, col_indices, q_values):
            x_action_constrained[row_idx, col_idx] = q_val
        
        # 5. get the prediction of action-constrained CE
        y_counterfactual = self.ml_model.predict(x_action_constrained)

        factual = self.return_dataframe(x_factual, y_factual)
        ce = self.return_dataframe(x_counterfactual, y_counterfactual)
        ace = self.return_dataframe(x_action_constrained, y_counterfactual)
        
        return factual, ce, ace


    def counterfactualexplanation_algorithm(self):
        # get Data
        x = self.data.get_x()
        # y = self.data.get_y()
        target_name = self.data.get_target_name()
        # x_labels = self.data.get_x_labels()

        if self.counterfactual_algorithm == 'dice':
            A = DiCE(ml_model=self.ml_model, x_factual=x, target_name= target_name, sample_num=4)
            return A


    def return_dataframe(self, x, y):
        df = pd.DataFrame(x)
        df.columns = self.data.get_x_labels()
        df[self.data.get_target_name()] = y 
        return df

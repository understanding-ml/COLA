import string
import numpy as np
import pandas as pd
import copy

from xai_cola.data_interface import BaseData
from xai_cola.ml_model_interface import Model

from xai_cola.cola_policy.matching import CounterfactualOptimalTransportPolicy, CounterfactualCoarsenedExactMatchingOTPolicy, CounterfactualNearestNeighborMatchingPolicy
from xai_cola.cola_policy.data_composer import DataComposer
from xai_cola.cola_policy.feature_attributor import Attributor, PSHAP
from xai_cola.plot.highlight_dataframe import highlight_differences

class COLA:
    """This class is used for creating COLA(COunterfactual with Limited Actions).

    Parameters:
    -----------
    ml_model: Your pre-trained model, used for prediction. And it should be wrapped in our BaseModel.
    x_factual: The factual instance for which you want to generate counterfactuals. It should be a numpy array.
    x_counterfacutal: The generated counterfactuals(either generated from our provided explainer, or generated by yourself). It should be a numpy array.
    
    matcher: The matcher between factual and counterfactual, it can be "ot": Optimal Transport or "Ect": Exact Match, "nn":nearest neighbor, "cem":coarsened exact matching etc.
    attributor:  The default attributor is "pshap"- shapley value with joint probability.
    Avalues_method: The method to choose the value of q

    """
    def __init__(
            self, 
            data:BaseData,
            ml_model: Model,
            x_factual: np = None, 
            x_counterfactual: np = None,
            ):
        self.data = data
        self.ml_model = ml_model
        self.x_factual = x_factual
        self.x_counterfactual = x_counterfactual
        
        self.row_indices = None
        self.col_indices = None

    def set_policy(
            self,
            matcher: string = "ot", 
            attributor: string = "pshap",
            Avalues_method: string = "max"):
        self.matcher = matcher
        self.attributor = attributor
        self.Avalues_method = Avalues_method
        if matcher == "ot":
            matcher_name = "Optimal Transport Matching"
        elif matcher == "cem":
            matcher_name = "Coarsened Exact Matching"
        elif matcher == "nn":
            matcher_name = "Nearest Neighbor Matching"
            pass
        print(f"You choose the Policy: {attributor} With {matcher_name}, Avalues_method is {Avalues_method}")
    
    def get_refined_counterfactual(self, limited_actions):
        varphi = self._get_attributor()
        q = self._get_data_composer()
        self.limited_actions = limited_actions

        # 1. Find the top 'action' highest probability values and their positions in the varphi matrix
        flat_indices = np.argpartition(varphi.flatten(), -self.limited_actions)[-self.limited_actions:]
        row_indices, col_indices = np.unravel_index(flat_indices, varphi.shape)
        
        # store the row and column indices
        self.row_indices = row_indices
        self.col_indices = col_indices

        # 2. Find the values at these positions in q
        q_values = q[row_indices, col_indices]
        # 3. Replace the corresponding values in x_factual with the values found in q
        x_action_constrained = self.x_factual.copy()
        # 4. get the action-constrained CE
        for row_idx, col_idx, q_val in zip(row_indices, col_indices, q_values):
            x_action_constrained[row_idx, col_idx] = q_val

        """get the corresponding_counterfacutal of the factual instance(processed by the matcher) 
        
        The difference between counterfactual and corresponding_counterfactual is that the corresponding_counterfactual is processed by the matcher.
        which means that each row in counterfactual is one-to-one correspondence to each row in a factual instance
        
        if we have n row factual instances and 2n rows counterfactual instances,
        after the matcher, we will choose from 2n rows couterfactual instances, and then get n rows corresponding_counterfactual instances.
        """    
        corresponding_counterfactual = q

        # 5. get the prediction
        y_factual = self.ml_model.predict(self.x_factual)
        y_counterfactual = self.ml_model.predict(self.x_counterfactual)
        y_counterfactual_limited_actions = self.ml_model.predict(x_action_constrained)
        y_corresponding_counterfactual = self.ml_model.predict(corresponding_counterfactual)

        # 6. return the dataframes
        self.factual_dataframe = None
        self.ce_dataframe = None
        self.ace_dataframe = None
        self.corresponding_counterfactual_dataframe = None
        self.factual_dataframe = self.return_dataframe(self.x_factual, y_factual)
        self.ce_dataframe = self.return_dataframe(self.x_counterfactual, y_counterfactual)
        self.ace_dataframe = self.return_dataframe(x_action_constrained, y_counterfactual_limited_actions)
        self.corresponding_counterfactual_dataframe = self.return_dataframe(corresponding_counterfactual, y_corresponding_counterfactual)
        # apply the same data type as the original data
        for col in self.ce_dataframe.columns:
            self.corresponding_counterfactual_dataframe[col] = self.corresponding_counterfactual_dataframe[col].astype(self.ce_dataframe[col].dtype)

        return self.factual_dataframe, self.ce_dataframe, self.ace_dataframe


    def _get_matcher(self):
        if self.matcher == "ot":
            joint_prob = CounterfactualOptimalTransportPolicy(self.x_factual, self.x_counterfactual).compute_prob_matrix_of_factual_and_counterfactual()
            return joint_prob
        elif self.matcher == "cem":
            joint_prob = CounterfactualCoarsenedExactMatchingOTPolicy(self.x_factual, self.x_counterfactual).compute_prob_matrix_of_factual_and_counterfactual()
            return joint_prob
        elif self.matcher == "nn":
            joint_prob = CounterfactualNearestNeighborMatchingPolicy(self.x_factual, self.x_counterfactual).compute_prob_matrix_of_factual_and_counterfactual()
            return joint_prob
        elif self.matcher == "ect":
            pass

    def _get_attributor(self):
        if self.attributor == "pshap":
            varphi = PSHAP(
                ml_model=self.ml_model, 
                x_factual=self.x_factual, 
                x_counterfactual=self.x_counterfactual, 
                joint_prob=self._get_matcher()
                ).calculate_varphi()
            return varphi
        elif self.attributor == "randomshap":
            pass

    def _get_data_composer(self):
        q = DataComposer(x_counterfactual=self.x_counterfactual, joint_prob=self._get_matcher(), method=self.Avalues_method).calculate_q()
        return q

    def return_dataframe(self, x, y):
        df = pd.DataFrame(x)
        df.columns = self.data.get_x_labels()
        df[self.data.get_target_name()] = y 
        df.style.set_properties(**{'text-align': 'center'})
        return df
    

    def highlight_changes(self):
        """ highlight the changes from factual to ace """
        self.factual_dataframe = self.factual_dataframe.astype(object)
        self.ace_dataframe = self.ace_dataframe.astype(object)
        self.corresponding_counterfactual_dataframe = self.corresponding_counterfactual_dataframe.astype(object)
    
        ace_style = self.ace_dataframe.style.apply(lambda x: highlight_differences(x, self.factual_dataframe, self.ace_dataframe, self.data.get_target_name()), axis=None)
        ace_style_1 = copy.deepcopy(ace_style.set_properties(**{'text-align': 'center'}))

        cce_style = self.corresponding_counterfactual_dataframe.style.apply(lambda x: highlight_differences(x, self.factual_dataframe, self.corresponding_counterfactual_dataframe, self.data.get_target_name()), axis=None)
        cce_style_1 = copy.deepcopy(cce_style.set_properties(**{'text-align': 'center'}))
        return self.factual_dataframe, cce_style_1, ace_style_1
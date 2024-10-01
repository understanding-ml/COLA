import numpy as np
import pandas as pd

from .base_policy import Policy

from xai_cola.data import BaseData
from xai_cola.ml_model import BaseModel

from xai_cola.cola.data_composer import DataComposer
from xai_cola.cola.feature_attributor.base_attributor import Attributor
from xai_cola.cola.matching import BaseMatcher, CounterfactualOptimalTransportPolicy


class PshapWithOTmatcher(Policy):
    def __init__(
            self,
            data: BaseData,
            ml_model:BaseModel, 
            x_factual:np = None, 
            x_counterfactual:np = None,
            p: BaseMatcher = None, 
            varphi: Attributor = None, 
            q: DataComposer = None,
            ):
        super().__init__(data, ml_model, x_factual, x_counterfactual, p, varphi, q)

    def counterfactual_with_limited_actions(self, limited_actions):
        self.limited_actions = limited_actions

        # 1. Find the top 'action' highest probability values and their positions in the varphi matrix
        flat_indices = np.argpartition(self.varphi.flatten(), -self.limited_actions)[-self.limited_actions:]
        row_indices, col_indices = np.unravel_index(flat_indices, self.varphi.shape)

        # 2. Find the values at these positions in q
        q_values = self.q[row_indices, col_indices]

        # 3. Replace the corresponding values in x_factual with the values found in q
        x_action_constrained = self.x_factual.copy()

        # 4. get the action-constrained CE
        for row_idx, col_idx, q_val in zip(row_indices, col_indices, q_values):
            x_action_constrained[row_idx, col_idx] = q_val

        # get model
        self.ml_model_origin = self.ml_model.model  

        # 5. get the prediction
        y_factual = self.ml_model.predict(self.x_factual)
        y_counterfactual = self.ml_model.predict(self.x_counterfactual)
        y_counterfactual_limited_actions = self.ml_model.predict(x_action_constrained)


        factual = self.return_dataframe(self.x_factual, y_factual)
        ce = self.return_dataframe(self.x_counterfactual, y_counterfactual)
        ace = self.return_dataframe(x_action_constrained, y_counterfactual_limited_actions)
        
        return factual, ce, ace



    def return_dataframe(self, x, y):
        df = pd.DataFrame(x)
        df.columns = self.data.get_x_labels()
        df[self.data.get_target_name()] = y 
        return df

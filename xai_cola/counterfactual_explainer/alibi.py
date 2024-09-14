import pandas as pd
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import alibi

from xai_cola.data import BaseData
from xai_cola.ml_model.model import Model
from .base_explainer import CounterFactualExplainer



class AlibiCounterfactualInstances(CounterFactualExplainer):
    def __init__(
            self, 
            ml_model: Model, 
            data: BaseData, 
            sample_num, 
            tolerance=0.1, 
            lam_init=1e-3, 
            max_iter=8000, 
            max_lam_steps=30
            ):
        self.tol = tolerance
        self.lam_init = lam_init
        self.max_iter = max_iter
        self.max_lam_steps = max_lam_steps
        super().__init__(ml_model, data, sample_num)

        self.x_factual_pandas = self.data.get_x()

        '''
        ml_model: the Model used to predict
        data : BaseData -> input the data without target column
        sample_num : the number of your aim counterfactuals
        '''
        self.factual, self.counterfactual = self.generate_counterfactuals()  # numpy type without 'target' column

    def get_factual_indices(self):
        x_factual_ext = self.x_factual_pandas.copy()
        predict = self.ml_model.predict(x_factual_ext)
        x_factual_ext[self.target_name] = predict
        sampling_weights = np.exp(x_factual_ext[self.target_name].values.clip(min=0) * 4)
        indices = (x_factual_ext.sample(self.sample_num, weights=sampling_weights)).index
        return indices, x_factual_ext

    def generate_counterfactuals(self) -> pd.DataFrame:

        indices, x_with_target = self.get_factual_indices()
        # x_chosen = self.data.get_x().loc[indices]
        x_chosen = self.x_factual_pandas.loc[indices]
        # 对每个实例生成反事实
        counterfactuals = []

        for i in range(len(x_chosen)):
            # x_factual = x_chosen[i].reshape(1, -1)  # 取出第 i 行数据并重塑为 (1, n) 形状
            chosen = x_chosen.iloc[[i]]
            predicted_class = self.ml_model.predict(chosen)
            print(f'Predicted class for factual instance {i}: {predicted_class}')
            
            # 设置目标类，反事实的目标是与原始预测结果相反
            target_class = 1 if predicted_class == 0 else 0
            print(f'Target class for counterfactual instance {i}: {target_class}')
            
            # 定义反事实解释器
            cf =alibi.explainers.Counterfactual(
                self.ml_model.predict_proba,                              # The model to explain
                shape=(1,) + chosen.shape[1:],     # The shape of the model input
                target_proba=0.51,                  # The target class probability
                tol=self.tol,                       # The tolerance for the loss
                # target_class='other',               # The target class to obtain  
                target_class = 0,
                max_iter=self.max_iter,
                lam_init=self.lam_init,
                max_lam_steps=self.max_lam_steps,  # 增加 lambda 调整步数
            )
            # 生成反事实实例
            explanation = cf.explain(chosen.values)
            counterfactuals.append(explanation['cf']['X'])
            
        # Convert list of counterfactuals to a NumPy array
        counterfactuals = np.vstack(counterfactuals)

        return x_chosen.values, counterfactuals
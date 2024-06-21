from .base_ce import CounterFactualExplainer
import pandas as pd
import numpy as np
from model_module.pretrained_model import PreTrainedModel
# from explainers import dce
import dice_ml

FACTUAL_CLASS = 1
SHUFFLE_COUNTERFACTUAL = True

class DiCE(CounterFactualExplainer):
    def __init__(self, ml_model, x_factual:pd.DataFrame, target_name, sample_num):
        super().__init__(ml_model,x_factual)
        self.target_name = target_name
        self.sample_num = sample_num

    def get_factual_indices(self):
        x_factual_ext = self.x_factual.copy()
        predict = self.ml_model.predict(self.x_factual)
        # X_test_ext[target_name] = model.predict(X_test.values)
        x_factual_ext[self.target_name] = predict
        sampling_weights = np.exp(x_factual_ext[self.target_name].values.clip(min=0) * 4)
        indices = (x_factual_ext.sample(self.sample_num, weights=sampling_weights)).index
        ####  把target_name列按照权重选出indices，risk=1的更容易被选出来
        return indices, x_factual_ext


    #              x_factual: pd.DataFrame, ml_model_path, target_name, sample_num
    # generate_x_counterfactuals(x_test, LGBMclassier, 'Risk', 4)
    def generate_x_counterfactuals(self) -> pd.DataFrame:

        # DICE counterfactual generation logic
        indices, x_factual_ext = self.get_factual_indices()
        x_chosen = self.x_factual.loc[indices]
        y_factual = self.ml_model.predict(x_chosen)

        # Prepare for DiCE
        dice_model = dice_ml.Model(model=self.ml_model, backend='sklearn')
        dice_features = x_chosen.columns.to_list()   #exclude 'Risk'
        dice_data = dice_ml.Data(
            dataframe = x_factual_ext,             # x_factual with 'Risk'
            continuous_features = dice_features,   # exclude 'Risk'
            outcome_name =self.target_name,              # 'Risk'
        )
        dice_explainer = dice_ml.Dice(dice_data, dice_model)
        dice_results = dice_explainer.generate_counterfactuals(
            query_instances = x_chosen,
            features_to_vary = dice_features,
            desired_class=1 - FACTUAL_CLASS,
            total_CFs=1,
        )

        # Iterate through each result and append to the DataFrame
        dice_df_list = []
        for cf in dice_results.cf_examples_list:
            # Convert to DataFrame and append
            cf_df = cf.final_cfs_df
            dice_df_list.append(cf_df)

        df_counterfactual = (
            pd.concat(dice_df_list).reset_index(drop=True).drop(self.target_name, axis=1)
        )
        if SHUFFLE_COUNTERFACTUAL:
            df_counterfactual = df_counterfactual.sample(frac=1).reset_index(drop=True)

        x_counterfactual = df_counterfactual
        print(f'---- x_counterfactual has already been generated ----')
        y_counterfactual = self.ml_model.predict(x_counterfactual)
        print(f'---- y_counterfactual has already been generated ----')

        return x_chosen.values, y_factual, x_counterfactual.values, y_counterfactual
















































# # 具体实现类 KNNCounterfactualExplainer
class KNNCounterfactualExplainer(CounterFactualExplainer):
    def __init__(self, model, data):
        super().__init__(model, data)

    def ce(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算反事实解释
        
        参数:
        data: DataFrame类型的数据变量
        
        返回:
        DataFrame类型的反事实解释结果
        """
        counterfactuals = data.copy()
        for col in counterfactuals.columns:
            counterfactuals[col] = counterfactuals[col].apply(lambda x: x - np.random.uniform(-1, 1))
        return counterfactuals

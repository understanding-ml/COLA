"""
COLA Data Module - 统一的数据接口

支持 Pandas DataFrame 和 NumPy array 输入
自动处理 target column 的管理
支持数据预处理和逆变换
"""

from typing import List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class COLAData:
    """
    COLA 统一数据接口
    
    支持同时管理 factual 和 counterfactual 数据
    自动验证数据一致性
    支持数据预处理（transform）和逆变换
    
    Parameters:
    -----------
    factual_data : Union[pd.DataFrame, np.ndarray]
        事实数据，必须包含 label column
        如果是 DataFrame：检查 label_column 是否存在
        如果是 numpy：需要提供所有列名（包含 label_column）
    
    label_column : str
        标签列名称，默认应在最后一列
    
    counterfactual_data : Optional[Union[pd.DataFrame, np.ndarray]]
        反事实数据（可选）
        如果是 DataFrame：检查与 factual 的列是否一致
        如果是 numpy：使用 factual 的列名
        
    column_names : Optional[List[str]]
        仅当 factual_data 是 numpy 时必需
        提供所有列名（包括 label_column），顺序要匹配
    
    transform : Optional[str], default=None
        数据预处理方法：
        - "ohe-zscore": categorical 特征独热编码，numerical 特征标准化（z-score）
        - "ohe-min-max": categorical 特征独热编码，numerical 特征归一化（min-max）
        - None: 不进行预处理（数据已经是处理好的格式）
    
    numerical_features : Optional[List[str]], default=None
        数值特征列表。如果为 None，默认所有特征都是 numerical。
        其他特征自动推断为 categorical。
    """
    
    def __init__(
        self,
        factual_data: Union[pd.DataFrame, np.ndarray],
        label_column: str,
        counterfactual_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        column_names: Optional[List[str]] = None,
        transform: Optional[str] = None,
        numerical_features: Optional[List[str]] = None
    ):
        # 验证并设置 label column
        self.label_column = label_column
        
        # 处理 factual data
        self.factual_df = self._process_input_data(
            factual_data, 
            data_type='factual',
            column_names=column_names,
            reference_df=None
        )
        
        # 设置 transformation 参数
        self.transform_method = transform
        self.numerical_features = numerical_features if numerical_features is not None else []
        
        # 初始化 transformer（如果指定了 transform）
        self._transformer = None
        if self.transform_method is not None:
            self._transformer = self._init_transformer()
        
        # 处理 counterfactual data（如果提供）
        self.counterfactual_df = None
        if counterfactual_data is not None:
            self.add_counterfactuals(counterfactual_data)
    
    def _process_input_data(
        self, 
        data: Union[pd.DataFrame, np.ndarray], 
        data_type: str,
        column_names: Optional[List[str]] = None,
        reference_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        处理输入数据，转换为 DataFrame
        
        Parameters:
        -----------
        data : Union[pd.DataFrame, np.ndarray]
            输入数据
        data_type : str
            'factual' or 'counterfactual'
        column_names : Optional[List[str]]
            列名（仅 numpy 需要）
        reference_df : Optional[pd.DataFrame]
            参考 DataFrame（用于验证 counterfactual）
        
        Returns:
        --------
        pd.DataFrame
            处理后的 DataFrame
        """
        if isinstance(data, pd.DataFrame):
            df = data.copy()
            
            # 如果是 factual，验证 label column 存在
            if data_type == 'factual':
                if self.label_column not in df.columns:
                    raise ValueError(
                        f"Label column '{self.label_column}' not found in factual data. "
                        f"Available columns: {df.columns.tolist()}"
                    )
            
            # 如果是 counterfactual，验证列一致性
            elif data_type == 'counterfactual' and reference_df is not None:
                expected_cols = reference_df.columns.tolist()
                actual_cols = df.columns.tolist()
                if expected_cols != actual_cols:
                    raise ValueError(
                        f"Counterfactual columns must match factual columns.\n"
                        f"Expected: {expected_cols}\n"
                        f"Got: {actual_cols}"
                    )
            
            return df
            
        elif isinstance(data, np.ndarray):
            if column_names is None and reference_df is None:
                raise ValueError(
                    "When providing numpy array, you must either:\n"
                    "1. Provide column_names parameter (for factual)\n"
                    "2. Provide counterfactual using add_counterfactuals() (uses factual columns)"
                )
            
            # 使用提供的列名或参考 DataFrame 的列名
            if column_names is not None:
                if len(column_names) != data.shape[1]:
                    raise ValueError(
                        f"Number of column_names ({len(column_names)}) doesn't match "
                        f"data shape ({data.shape[1]} columns)"
                    )
                columns = column_names
            elif reference_df is not None:
                columns = reference_df.columns.tolist()
                if len(columns) != data.shape[1]:
                    raise ValueError(
                        f"Counterfactual shape ({data.shape[1]} columns) doesn't match "
                        f"factual shape ({len(columns)} columns)"
                    )
            else:
                raise ValueError("Must provide column_names")
            
            return pd.DataFrame(data, columns=columns)
        
        else:
            raise TypeError(
                f"Unsupported data type: {type(data)}. "
                f"Supported types: pd.DataFrame, np.ndarray"
            )
    
    def _init_transformer(self):
        """
        初始化数据转换器
        
        Returns:
        --------
        ColumnTransformer
            配置好的数据转换器
        """
        if self.transform_method is None:
            return None
        
        # 获取特征列
        feature_columns = self.get_feature_columns()
        
        # 推断 categorical features（所有非 numerical 的特征）
        categorical_features = [col for col in feature_columns if col not in self.numerical_features]
        numerical_features = [col for col in feature_columns if col in self.numerical_features]
        
        transformers = []
        
        # 添加 categorical 特征处理器（OneHotEncoder）
        if categorical_features:
            transformers.append(
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
                 categorical_features)
            )
        
        # 添加 numerical 特征处理器
        if numerical_features:
            if self.transform_method == 'ohe-zscore':
                transformers.append(
                    ('num', StandardScaler(), numerical_features)
                )
            elif self.transform_method == 'ohe-min-max':
                transformers.append(
                    ('num', MinMaxScaler(), numerical_features)
                )
        
        if not transformers:
            return None
        
        transformer = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough',
            verbose_feature_names_out=False
        )
        
        # Fit transformer on factual data
        factual_features = self.get_factual_features()
        transformer.fit(factual_features)
        
        return transformer
    
    def _transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        对数据进行转换
        
        Parameters:
        -----------
        data : pd.DataFrame
            输入数据（只包含特征列，不含 target）
        
        Returns:
        --------
        pd.DataFrame
            转换后的数据
        """
        if self._transformer is None:
            return data.copy()
        
        # 转换数据
        transformed_array = self._transformer.transform(data)
        
        # 获取转换后的列名
        try:
            transformed_columns = self._transformer.get_feature_names_out()
        except AttributeError:
            # 对于老版本的 sklearn，使用不同的方法
            transformed_columns = self._transformer.get_feature_names(data.columns)
        
        # 创建 DataFrame
        transformed_df = pd.DataFrame(
            transformed_array,
            columns=transformed_columns,
            index=data.index
        )
        
        return transformed_df
    
    def _inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        对转换后的数据进行逆变换，还原为原始格式
        
        Parameters:
        -----------
        data : pd.DataFrame
            转换后的数据
        
        Returns:
        --------
        pd.DataFrame
            还原后的数据
        """
        if self._transformer is None:
            return data.copy()
        
        # 逆变换
        try:
            inverse_array = self._transformer.inverse_transform(data)
        except Exception as e:
            raise ValueError(
                f"Inverse transformation failed: {e}. "
                f"This may occur if the transformed data has a different number of features."
            )
        
        # 还原列名（使用原始特征列名）
        original_columns = self.get_feature_columns()
        
        # 如果逆变换返回的列数与原始不同，说明可能有 one-hot 编码
        if inverse_array.shape[1] != len(original_columns):
            # 这种情况比较复杂，需要更智能的处理
            # 暂时返回一个警告并尝试截取
            import warnings
            warnings.warn(
                f"Number of features after inverse transform ({inverse_array.shape[1]}) "
                f"doesn't match original ({len(original_columns)}). "
                f"Attempting to truncate/pad."
            )
            if inverse_array.shape[1] > len(original_columns):
                inverse_array = inverse_array[:, :len(original_columns)]
            else:
                # Pad with zeros
                pad_width = ((0, 0), (0, len(original_columns) - inverse_array.shape[1]))
                inverse_array = np.pad(inverse_array, pad_width, 'constant')
        
        inverse_df = pd.DataFrame(
            inverse_array,
            columns=original_columns,
            index=data.index
        )
        
        return inverse_df
    
    def add_counterfactuals(
        self, 
        counterfactual_data: Union[pd.DataFrame, np.ndarray],
        with_target_column: bool = True
    ):
        """
        添加或更新反事实数据
        
        Parameters:
        -----------
        counterfactual_data : Union[pd.DataFrame, np.ndarray]
            反事实数据
            如果是 DataFrame：检查与 factual 的列是否一致（取决于 with_target_column）
            如果是 numpy：使用 factual 的列名（取决于 with_target_column）
        with_target_column : bool, default=False
            如果 True：counterfactual_data 包含 target column，列数与 factual 相同
            如果 False：counterfactual_data 不包含 target column，只有特征列
                      此时会自动从 factual 的 target column 反转值（0->1, 1->0）并添加
        
        Raises:
        -------
        ValueError
            如果 with_target_column=False 时，factual 和 counterfactual 行数不一致
        """
        if with_target_column:
            # Counterfactual 包含 target column，处理逻辑和之前一样
            self.counterfactual_df = self._process_input_data(
                counterfactual_data,
                data_type='counterfactual',
                reference_df=self.factual_df
            )
        else:
            # Counterfactual 不包含 target column
            # 首先处理特征数据
            if isinstance(counterfactual_data, pd.DataFrame):
                cf_features_df = counterfactual_data.copy()
                # 检查是否意外包含了 target column
                if self.label_column in cf_features_df.columns:
                    raise ValueError(
                        f"Counterfactual data contains target column '{self.label_column}', "
                        f"but with_target_column=False. "
                        f"Either remove the target column from counterfactual data "
                        f"or set with_target_column=True."
                    )
            elif isinstance(counterfactual_data, np.ndarray):
                # numpy array，应该是特征数据
                feature_columns = self.get_feature_columns()
                if counterfactual_data.shape[1] != len(feature_columns):
                    # 检查是否包含了 target column
                    if counterfactual_data.shape[1] == len(self.get_all_columns()):
                        raise ValueError(
                            f"Counterfactual numpy array has {counterfactual_data.shape[1]} columns, "
                            f"which matches all columns (including target). "
                            f"Set with_target_column=True if counterfactual includes target column, "
                            f"or provide only {len(feature_columns)} feature columns."
                        )
                    else:
                        raise ValueError(
                            f"Counterfactual shape ({counterfactual_data.shape[1]} columns) doesn't match "
                            f"expected feature columns ({len(feature_columns)})."
                        )
                cf_features_df = pd.DataFrame(counterfactual_data, columns=feature_columns)
            else:
                raise TypeError(
                    f"Unsupported data type: {type(counterfactual_data)}. "
                    f"Supported types: pd.DataFrame, np.ndarray"
                )
            
            # 验证行数一致
            if len(cf_features_df) != len(self.factual_df):
                raise ValueError(
                    f"Factual and counterfactual must have the same number of rows. "
                    f"Factual: {len(self.factual_df)} rows, "
                    f"Counterfactual: {len(cf_features_df)} rows."
                )
            
            # 获取 factual 的 target column 值并反转（0->1, 1->0）
            factual_labels = self.get_factual_labels()
            reversed_labels = 1 - factual_labels  # 反转：0->1, 1->0
            
            # 创建完整的 counterfactual DataFrame（包含 target column）
            self.counterfactual_df = cf_features_df.copy()
            self.counterfactual_df[self.label_column] = reversed_labels.values
            
            # 确保列的顺序与 factual 一致
            self.counterfactual_df = self.counterfactual_df[self.get_all_columns()]
    
    # ========== 列名相关方法 ==========
    
    def get_all_columns(self) -> List[str]:
        """
        获取所有列名（包含 label column）
        
        Returns:
        --------
        List[str]
            所有列名的列表
        """
        return self.factual_df.columns.tolist()
    
    def get_feature_columns(self) -> List[str]:
        """
        获取特征列名（不包含 label column）
        
        Returns:
        --------
        List[str]
            特征列名的列表
        """
        return [col for col in self.factual_df.columns if col != self.label_column]
    
    def get_label_column(self) -> str:
        """
        获取标签列名
        
        Returns:
        --------
        str
            标签列名
        """
        return self.label_column
    
    # ========== Factual 数据方法 ==========
    
    def get_factual_all(self) -> pd.DataFrame:
        """
        获取包含 label column 的完整 factual DataFrame
        
        Returns:
        --------
        pd.DataFrame
            完整的 factual 数据（包含 label column）
        """
        return self.factual_df.copy()
    
    def get_factual_features(self) -> pd.DataFrame:
        """
        获取不包含 label column 的 factual 特征数据
        
        Returns:
        --------
        pd.DataFrame
            Factual 特征数据（不含 label column）
        """
        return self.factual_df.drop(columns=[self.label_column]).copy()
    
    def get_factual_labels(self) -> pd.Series:
        """
        获取 factual 的标签列
        
        Returns:
        --------
        pd.Series
            Factual 标签列
        """
        return self.factual_df[self.label_column].copy()
    
    # ========== Counterfactual 数据方法 ==========
    
    def get_counterfactual_all(self) -> pd.DataFrame:
        """
        获取包含 label column 的完整 counterfactual DataFrame
        
        Returns:
        --------
        pd.DataFrame
            完整的 counterfactual 数据（包含 label column）
            
        Raises:
        -------
        ValueError
            如果 counterfactual 数据未设置
        """
        if self.counterfactual_df is None:
            raise ValueError("Counterfactual data has not been set. Use add_counterfactuals() first.")
        return self.counterfactual_df.copy()
    
    def get_counterfactual_features(self) -> pd.DataFrame:
        """
        获取不包含 label column 的 counterfactual 特征数据
        
        Returns:
        --------
        pd.DataFrame
            Counterfactual 特征数据（不含 label column）
            
        Raises:
        -------
        ValueError
            如果 counterfactual 数据未设置
        """
        if self.counterfactual_df is None:
            raise ValueError("Counterfactual data has not been set. Use add_counterfactuals() first.")
        return self.counterfactual_df.drop(columns=[self.label_column]).copy()
    
    def get_counterfactual_labels(self) -> pd.Series:
        """
        获取 counterfactual 的标签列
        
        Returns:
        --------
        pd.Series
            Counterfactual 标签列
            
        Raises:
        -------
        ValueError
            如果 counterfactual 数据未设置
        """
        if self.counterfactual_df is None:
            raise ValueError("Counterfactual data has not been set. Use add_counterfactuals() first.")
        return self.counterfactual_df[self.label_column].copy()
    
    # ========== 便捷方法 ==========
    
    def has_counterfactual(self) -> bool:
        """
        检查是否已设置 counterfactual 数据
        
        Returns:
        --------
        bool
            如果有 counterfactual 数据返回 True
        """
        return self.counterfactual_df is not None
    
    def get_feature_count(self) -> int:
        """
        获取特征数量（不包含 label column）
        
        Returns:
        --------
        int
            特征数量
        """
        return len(self.get_feature_columns())
    
    def get_sample_count(self) -> int:
        """
        获取样本数量
        
        Returns:
        --------
        int
            样本数量
        """
        return len(self.factual_df)
    
    # ========== NumPy 转换方法 ==========
    
    def to_numpy_factual_features(self) -> np.ndarray:
        """
        将 factual 特征转换为 NumPy array
        
        Returns:
        --------
        np.ndarray
            Factual 特征矩阵
        """
        return self.get_factual_features().values
    
    def to_numpy_counterfactual_features(self) -> np.ndarray:
        """
        将 counterfactual 特征转换为 NumPy array
        
        Returns:
        --------
        np.ndarray
            Counterfactual 特征矩阵
            
        Raises:
        -------
        ValueError
            如果 counterfactual 数据未设置
        """
        return self.get_counterfactual_features().values
    
    def to_numpy_labels(self) -> np.ndarray:
        """
        将标签转换为 NumPy array
        
        Returns:
        --------
        np.ndarray
            标签数组
        """
        return self.get_factual_labels().values
    
    # ========== 信息方法 ==========
    
    def __repr__(self) -> str:
        """字符串表示"""
        cf_info = f", counterfactual: {len(self.counterfactual_df)} rows" if self.counterfactual_df is not None else ", no counterfactual"
        return (
            f"COLAData(factual: {len(self.factual_df)} rows, "
            f"features: {self.get_feature_count()}, "
            f"label: {self.label_column}{cf_info})"
        )
    
    def __str__(self) -> str:
        return self.__repr__()
    
    def summary(self) -> dict:
        """
        获取数据摘要信息
        
        Returns:
        --------
        dict
            包含数据摘要的字典
        """
        info = {
            'factual_samples': len(self.factual_df),
            'feature_count': self.get_feature_count(),
            'label_column': self.label_column,
            'all_columns': self.get_all_columns(),
            'has_counterfactual': self.has_counterfactual()
        }
        
        if self.counterfactual_df is not None:
            info['counterfactual_samples'] = len(self.counterfactual_df)
        
        return info


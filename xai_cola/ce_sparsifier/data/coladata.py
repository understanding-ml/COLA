"""
COLA Data Module - 统一的数据接口

支持 Pandas DataFrame 和 NumPy array 输入
自动处理 target column 的管理
"""

from typing import List, Optional, Union
import pandas as pd
import numpy as np

try:
    from sklearn.compose import ColumnTransformer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class COLAData:
    """
    COLA 统一数据接口

    支持同时管理 factual 和 counterfactual 数据
    自动验证数据一致性

    Parameters:
    -----------
    factual_data : Union[pd.DataFrame, np.ndarray]
        事实数据（原始数据），必须包含 label column
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

    numerical_features : Optional[List[str]], default=None
        数值特征列表。用于记录哪些特征是连续数值型。
        如果为 None，默认所有特征都是 numerical。
        其他特征自动推断为 categorical。
        注意：这个参数仅用于记录特征类型信息，不进行数据转换。

    transform_method : Optional[object], default=None
        数据预处理器（如 sklearn 的 StandardScaler, ColumnTransformer 等）
        必须有 transform() 和 inverse_transform() 方法
        用于在生成反事实前后进行数据转换

    preprocessor : Optional[object], default=None
        transform_method 的别名，两者选其一即可
    """

    def __init__(
        self,
        factual_data: Union[pd.DataFrame, np.ndarray],
        label_column: str,
        counterfactual_data: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        column_names: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        transform_method: Optional[object] = None,
        preprocessor: Optional[object] = None
    ):
        # 验证并设置 label column
        self.label_column = label_column

        # 设置数据预处理器（transform_method 和 preprocessor 是别名，取其一）
        if transform_method is not None and preprocessor is not None:
            raise ValueError("Cannot specify both 'transform_method' and 'preprocessor'. Use one or the other.")

        self.transform_method = transform_method if transform_method is not None else preprocessor

        # 验证 transform_method 有必要的方法
        if self.transform_method is not None:
            if not hasattr(self.transform_method, 'transform'):
                raise ValueError("transform_method must have a 'transform()' method")

            # 对于 ColumnTransformer，我们在内部实现了自定义的 inverse_transform
            # 所以不需要在这里检查 inverse_transform 方法
            # 对于其他转换器，只在实际调用时才会检查
            if not (SKLEARN_AVAILABLE and isinstance(self.transform_method, ColumnTransformer)):
                # 非 ColumnTransformer 需要有 inverse_transform
                if not hasattr(self.transform_method, 'inverse_transform'):
                    raise ValueError(
                        "transform_method must have an 'inverse_transform()' method. "
                        "ColumnTransformer is supported with custom inverse_transform logic."
                    )

        # 处理 factual data
        self.factual_df = self._process_input_data(
            factual_data,
            data_type='factual',
            column_names=column_names,
            reference_df=None
        )

        # 设置 numerical_features（仅用于记录特征类型信息）
        self.numerical_features = numerical_features if numerical_features is not None else []

        # 如果明确给出了 numerical_features，则把剩余的特征当作 categorical 并
        # 将它们的值转换为字符串类型，避免后续在交互（例如 DiCE 生成反事实）中
        # 因类型不匹配导致的警告或错误。不要转换 label column。
        if self.numerical_features:
            try:
                categorical_cols = [
                    col for col in self.get_feature_columns() if col not in self.numerical_features
                ]
                for col in categorical_cols:
                    if col in self.factual_df.columns:
                        # 转为字符串以避免 int/str 混合导致的 pandas 警告
                        try:
                            self.factual_df[col] = self.factual_df[col].astype(str)
                        except Exception:
                            # 如果转换失败（非常罕见），则跳过该列
                            pass
            except Exception:
                # 容错：任何异常都不应阻塞 COLAData 的构造
                pass

        # 处理 counterfactual data（如果提供）
        self.counterfactual_df = None
        if counterfactual_data is not None:
            self.add_counterfactuals(counterfactual_data)

        # ========== 转换后的数据存储 ==========
        # 如果设置了 transform_method，自动计算并存储转换后的数据
        self.transformed_factual_df = None
        self.transformed_counterfactual_df = None
        self.transformed_column_order = None  # 转换后的列顺序（ColumnTransformer 会改变列顺序）

        if self.transform_method is not None:
            # 转换 factual 数据
            factual_features = self.get_factual_features()
            self.transformed_factual_df = self._transform(factual_features)

            # 记录转换后的列顺序
            self.transformed_column_order = self.transformed_factual_df.columns.tolist()

            # 如果已经有 counterfactual 数据，也进行转换
            if self.counterfactual_df is not None:
                counterfactual_features = self.get_counterfactual_features()
                self.transformed_counterfactual_df = self._transform(counterfactual_features)
    
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
            # 如果指定了 numerical_features，同样把 counterfactual 的类别特征转换为字符串
            if self.numerical_features:
                try:
                    categorical_cols = [
                        col for col in self.get_feature_columns() if col not in self.numerical_features
                    ]
                    for col in categorical_cols:
                        if col in self.counterfactual_df.columns:
                            try:
                                self.counterfactual_df[col] = self.counterfactual_df[col].astype(str)
                            except Exception:
                                pass
                except Exception:
                    pass
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

        # 如果设置了 transform_method，同时转换 counterfactual 数据
        if self.transform_method is not None:
            counterfactual_features = self.get_counterfactual_features()
            self.transformed_counterfactual_df = self._transform(counterfactual_features)
    
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

    def get_numerical_features(self) -> List[str]:
        """
        获取数值特征列表

        Returns:
        --------
        List[str]
            数值特征列名的列表
        """
        return self.numerical_features.copy() if self.numerical_features else []

    def get_categorical_features(self) -> List[str]:
        """
        获取类别特征列表（所有非数值特征）

        Returns:
        --------
        List[str]
            类别特征列名的列表
        """
        feature_columns = self.get_feature_columns()
        if not self.numerical_features:
            # 如果没有指定 numerical_features，则假设所有特征都是数值型，返回空列表
            return []
        return [col for col in feature_columns if col not in self.numerical_features]

    def get_transformed_feature_columns(self) -> Optional[List[str]]:
        """
        获取转换后的特征列名

        对于 ColumnTransformer，列顺序会变为 [numerical_features, categorical_features]
        对于其他转换器，列顺序保持不变

        Returns:
        --------
        Optional[List[str]]
            转换后的特征列名列表，如果未设置 transform_method 则返回 None
        """
        if self.transform_method is None:
            return None
        return self.transformed_column_order
    
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

    def get_transformed_factual_features(self) -> Optional[pd.DataFrame]:
        """
        获取转换后的 factual 特征数据

        Returns:
        --------
        Optional[pd.DataFrame]
            转换后的 factual 特征数据，如果未设置 transform_method 则返回 None

        Example:
        --------
        >>> data = COLAData(df, label_column='Risk', transform_method=scaler)
        >>> transformed = data.get_transformed_factual_features()
        >>> # 用于计算 Shapley values 或其他基于转换后数据的计算
        """
        if self.transform_method is None:
            return None
        return self.transformed_factual_df.copy()

    def has_transformed_data(self) -> bool:
        """
        检查是否有转换后的数据

        Returns:
        --------
        bool
            如果设置了 transform_method 并有转换后的数据返回 True
        """
        return self.transformed_factual_df is not None
    
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

    def get_transformed_counterfactual_features(self) -> Optional[pd.DataFrame]:
        """
        获取转换后的 counterfactual 特征数据

        Returns:
        --------
        Optional[pd.DataFrame]
            转换后的 counterfactual 特征数据，如果未设置 transform_method 或 counterfactual 则返回 None

        Raises:
        -------
        ValueError
            如果设置了 transform_method 但 counterfactual 数据未设置

        Example:
        --------
        >>> data = COLAData(df, label_column='Risk', transform_method=scaler)
        >>> data.add_counterfactuals(cf_df)
        >>> transformed_cf = data.get_transformed_counterfactual_features()
        >>> # 用于在转换空间中计算 matching 或 Q
        """
        if self.transform_method is None:
            return None
        if self.counterfactual_df is None:
            raise ValueError("Counterfactual data has not been set. Use add_counterfactuals() first.")
        return self.transformed_counterfactual_df.copy()
    
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

    def to_numpy_transformed_factual_features(self) -> Optional[np.ndarray]:
        """
        将转换后的 factual 特征转换为 NumPy array

        Returns:
        --------
        Optional[np.ndarray]
            转换后的 factual 特征矩阵，如果未设置 transform_method 则返回 None

        Example:
        --------
        >>> data = COLAData(df, label_column='Risk', transform_method=scaler)
        >>> X_transformed = data.to_numpy_transformed_factual_features()
        >>> # 在转换空间中计算 Shapley values
        """
        if self.transform_method is None:
            return None
        return self.transformed_factual_df.values

    def to_numpy_transformed_counterfactual_features(self) -> Optional[np.ndarray]:
        """
        将转换后的 counterfactual 特征转换为 NumPy array

        Returns:
        --------
        Optional[np.ndarray]
            转换后的 counterfactual 特征矩阵，如果未设置 transform_method 或 counterfactual 则返回 None

        Raises:
        -------
        ValueError
            如果设置了 transform_method 但 counterfactual 数据未设置

        Example:
        --------
        >>> data = COLAData(df, label_column='Risk', transform_method=scaler)
        >>> data.add_counterfactuals(cf_df)
        >>> CF_transformed = data.to_numpy_transformed_counterfactual_features()
        >>> # 在转换空间中计算 matching 距离
        """
        if self.transform_method is None:
            return None
        if self.counterfactual_df is None:
            raise ValueError("Counterfactual data has not been set. Use add_counterfactuals() first.")
        return self.transformed_counterfactual_df.values
    
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
            'has_counterfactual': self.has_counterfactual(),
            'has_transform_method': self.transform_method is not None,
            'has_transformed_data': self.has_transformed_data()
        }

        if self.counterfactual_df is not None:
            info['counterfactual_samples'] = len(self.counterfactual_df)

        if self.has_transformed_data():
            info['transformed_feature_columns'] = self.get_transformed_feature_columns()
            info['has_transformed_counterfactual'] = self.transformed_counterfactual_df is not None

        return info

    # ========== 数据转换方法 ==========

    def _transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        应用数据转换（使用 transform_method）

        Parameters:
        -----------
        data : pd.DataFrame
            需要转换的数据（只包含特征列，不含 label column）

        Returns:
        --------
        pd.DataFrame
            转换后的数据

        Raises:
        -------
        ValueError
            如果未设置 transform_method

        Example:
        --------
        >>> from sklearn.preprocessing import StandardScaler
        >>> scaler = StandardScaler()
        >>> scaler.fit(X_train)
        >>> data = COLAData(
        ...     factual_data=df,
        ...     label_column='Risk',
        ...     transform_method=scaler
        ... )
        >>> transformed = data._transform(data.get_factual_features())
        """
        if self.transform_method is None:
            raise ValueError("No transform_method is set. Cannot transform data.")

        # 应用转换
        # 对于 ColumnTransformer，需要传入 DataFrame 以便根据列名选择特征
        if SKLEARN_AVAILABLE and isinstance(self.transform_method, ColumnTransformer):
            transformed_values = self.transform_method.transform(data)  # 传入 DataFrame

            # ColumnTransformer 的输出顺序是 [numerical_features, categorical_features]
            # 而不是原始数据的列顺序
            transformed_column_order = self.numerical_features + self.get_categorical_features()

            transformed_df = pd.DataFrame(
                transformed_values,
                columns=transformed_column_order,  # 使用转换后的列顺序
                index=data.index
            )
        else:
            transformed_values = self.transform_method.transform(data.values)  # 传入 numpy array

            # 创建 DataFrame，保持原始列名和索引
            transformed_df = pd.DataFrame(
                transformed_values,
                columns=data.columns,
                index=data.index
            )

        return transformed_df

    def _inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        应用逆转换（使用 transform_method 的 inverse_transform）

        Parameters:
        -----------
        data : pd.DataFrame
            需要逆转换的数据（只包含特征列，不含 label column）

        Returns:
        --------
        pd.DataFrame
            逆转换后的数据

        Raises:
        -------
        ValueError
            如果未设置 transform_method

        Example:
        --------
        >>> # 假设数据经过了标准化
        >>> original = data._inverse_transform(transformed_data)

        Notes:
        ------
        对于 ColumnTransformer（包含 OrdinalEncoder 等分类编码器），此方法会智能处理：
        - 自动检测 ColumnTransformer 并分离数值和分类特征
        - 对分类特征进行四舍五入后再逆转换，避免浮点数错误
        - 正确重组特征顺序
        """
        if self.transform_method is None:
            raise ValueError("No transform_method is set. Cannot inverse transform data.")

        # 检查是否是 ColumnTransformer
        if SKLEARN_AVAILABLE and isinstance(self.transform_method, ColumnTransformer):
            return self._inverse_transform_column_transformer(data)
        else:
            # 标准逆转换（适用于 StandardScaler 等简单转换器）
            inverse_transformed_values = self.transform_method.inverse_transform(data.values)

            # 创建 DataFrame，保持列名和索引
            inverse_transformed_df = pd.DataFrame(
                inverse_transformed_values,
                columns=data.columns,
                index=data.index
            )

            return inverse_transformed_df

    def _inverse_transform_column_transformer(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        专门处理 ColumnTransformer 的逆转换

        此方法解决了 ColumnTransformer 逆转换分类特征时的浮点数问题：
        - 分离数值和分类特征
        - 对分类特征进行四舍五入（如果需要）
        - 分别逆转换后重组

        支持：
        - 简单转换器（StandardScaler, OrdinalEncoder）
        - Pipeline 转换器（如 OrdinalEncoder + StandardScaler）

        Parameters:
        -----------
        data : pd.DataFrame
            转换后的数据

        Returns:
        --------
        pd.DataFrame
            逆转换后的数据
        """
        transformer = self.transform_method

        # 获取各个子转换器
        num_transformer = transformer.named_transformers_.get('num')
        cat_transformer = transformer.named_transformers_.get('cat')

        n_num = len(self.numerical_features)
        n_cat = len(self.get_categorical_features())

        # 将数据转为 numpy array
        X_transformed = data.values

        # 分离数值和分类特征
        results = []
        feature_names = []

        # 处理数值特征
        if n_num > 0 and num_transformer is not None:
            X_num_scaled = X_transformed[:, :n_num]
            X_num_original = num_transformer.inverse_transform(X_num_scaled)
            results.append(X_num_original)
            feature_names.extend(self.numerical_features)

        # 处理分类特征
        if n_cat > 0 and cat_transformer is not None:
            X_cat_encoded = X_transformed[:, n_num:n_num + n_cat]

            # 检查 cat_transformer 是否是 Pipeline
            is_pipeline = hasattr(cat_transformer, 'named_steps')

            if is_pipeline:
                # Pipeline 情况：通常是 OrdinalEncoder -> StandardScaler
                # 我们需要手动逆转换，不能使用 Pipeline.inverse_transform()
                # 因为 Pipeline.inverse_transform() 会自动执行所有步骤，
                # 导致直接返回字符串，无法插入 round/clip 逻辑
                #
                # 正确步骤：
                # 1. StandardScaler.inverse_transform() → 得到浮点数编码
                # 2. round + clip → 得到有效的整数编码
                # 3. OrdinalEncoder.inverse_transform() → 得到原始字符串

                # 获取 Pipeline 中的各个步骤
                ordinal_encoder = None
                scaler = None

                for step_name, step_transformer in cat_transformer.named_steps.items():
                    if hasattr(step_transformer, 'categories_'):
                        ordinal_encoder = step_transformer
                    elif hasattr(step_transformer, 'mean_'):  # StandardScaler 有 mean_ 属性
                        scaler = step_transformer

                # 第一步：StandardScaler 逆转换（从标准化空间 → 编码空间）
                if scaler is not None:
                    X_cat_after_scaler = scaler.inverse_transform(X_cat_encoded)
                else:
                    # 如果没有 scaler，直接使用原始数据
                    X_cat_after_scaler = X_cat_encoded

                # 第二步：四舍五入并限制在有效范围内
                X_cat_rounded = np.round(X_cat_after_scaler)

                if ordinal_encoder is not None:
                    # 限制编码值在有效范围内
                    for i in range(X_cat_rounded.shape[1]):
                        n_categories = len(ordinal_encoder.categories_[i])
                        X_cat_rounded[:, i] = np.clip(X_cat_rounded[:, i], 0, n_categories - 1)

                    # 第三步：OrdinalEncoder 逆转换（从编码 → 原始字符串）
                    X_cat_original = ordinal_encoder.inverse_transform(X_cat_rounded)
                else:
                    # 如果没有 OrdinalEncoder，直接使用 rounded 结果
                    X_cat_original = X_cat_rounded
            else:
                # 非 Pipeline 情况：直接是 OrdinalEncoder
                # 对分类特征取整，避免浮点数导致的逆转换错误
                X_cat_encoded_rounded = np.round(X_cat_encoded)

                # 限制编码值在有效范围内（OrdinalEncoder 的编码范围是 0 到 n_categories-1）
                if hasattr(cat_transformer, 'categories_'):
                    for i in range(X_cat_encoded_rounded.shape[1]):
                        n_categories = len(cat_transformer.categories_[i])
                        X_cat_encoded_rounded[:, i] = np.clip(X_cat_encoded_rounded[:, i], 0, n_categories - 1)

                # 逆转换分类特征
                X_cat_original = cat_transformer.inverse_transform(X_cat_encoded_rounded)

            results.append(X_cat_original)
            feature_names.extend(self.get_categorical_features())

        # 合并结果
        if len(results) == 0:
            raise ValueError("No features to inverse transform in ColumnTransformer")

        X_original = np.hstack(results) if len(results) > 1 else results[0]

        # 创建 DataFrame
        # 注意：由于 _transform() 已经确保了列顺序是 [numerical_features, categorical_features]
        # 这里 feature_names 的顺序也是 [numerical_features, categorical_features]
        inverse_transformed_df = pd.DataFrame(
            X_original,
            columns=feature_names,
            index=data.index
        )

        # 重新排列列顺序，使其与原始数据的列顺序一致
        original_feature_columns = self.get_feature_columns()
        inverse_transformed_df = inverse_transformed_df[original_feature_columns]

        return inverse_transformed_df


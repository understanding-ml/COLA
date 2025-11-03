"""Tests for data interface module - Updated for COLAData."""

import pytest
import numpy as np
import pandas as pd
from xai_cola.data import COLAData


class TestCOLAData:
    """Tests for COLAData class."""
    
    def test_init_with_dataframe(self):
        """Test initialization with pandas DataFrame."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        data = COLAData(factual_data=df, label_column='target')
        
        assert data is not None
        assert isinstance(data, COLAData)
        assert data.has_counterfactual() == False
    
    def test_get_all_columns(self):
        """Test get_all_columns returns all column names."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        data = COLAData(factual_data=df, label_column='target')
        
        columns = data.get_all_columns()
        assert isinstance(columns, list)
        assert 'feature1' in columns
        assert 'feature2' in columns
        assert 'target' in columns
    
    def test_get_feature_columns(self):
        """Test get_feature_columns returns feature columns."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        data = COLAData(factual_data=df, label_column='target')
        
        features = data.get_feature_columns()
        assert isinstance(features, list)
        assert 'feature1' in features
        assert 'feature2' in features
        assert 'target' not in features
    
    def test_get_factual_features(self):
        """Test get_factual_features returns features without target."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        data = COLAData(factual_data=df, label_column='target')
        
        features = data.get_factual_features()
        assert isinstance(features, pd.DataFrame)
        assert 'target' not in features.columns
        assert 'feature1' in features.columns
        assert 'feature2' in features.columns
    
    def test_get_factual_labels(self):
        """Test get_factual_labels returns target column."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        data = COLAData(factual_data=df, label_column='target')
        
        labels = data.get_factual_labels()
        assert isinstance(labels, pd.Series)
        assert len(labels) == 3
        assert list(labels) == [0, 1, 0]
    
    def test_get_factual_all(self):
        """Test get_factual_all returns complete data."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        data = COLAData(factual_data=df, label_column='target')
        
        all_data = data.get_factual_all()
        assert isinstance(all_data, pd.DataFrame)
        assert 'target' in all_data.columns
        assert len(all_data) == 3
    
    def test_label_column_not_found(self):
        """Test that missing label column raises error."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        with pytest.raises(ValueError, match="not found"):
            COLAData(factual_data=df, label_column='target')
    
    def test_init_with_numpy(self):
        """Test initialization with numpy array."""
        X = np.array([
            [1, 2, 0],
            [3, 4, 1],
            [5, 6, 0]
        ])
        columns = ['feature1', 'feature2', 'target']
        
        data = COLAData(
            factual_data=X, 
            label_column='target',
            column_names=columns
        )
        
        assert data is not None
        assert isinstance(data, COLAData)
    
    def test_get_label_column(self):
        """Test get_label_column returns label column name."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        data = COLAData(factual_data=df, label_column='target')
        
        assert data.get_label_column() == 'target'
    
    def test_add_counterfactual(self):
        """Test adding counterfactual data."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        data = COLAData(factual_data=df, label_column='target')
        
        assert data.has_counterfactual() == False
        
        cf_df = pd.DataFrame({
            'feature1': [10, 20, 30],
            'feature2': [40, 50, 60],
            'target': [1, 0, 1]
        })
        
        data.add_counterfactuals(cf_df)
        assert data.has_counterfactual() == True
        
        cf_data = data.get_counterfactual_all()
        assert isinstance(cf_data, pd.DataFrame)
        assert len(cf_data) == 3
    
    def test_counterfactual_column_mismatch(self):
        """Test that counterfactual with mismatched columns raises error."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        data = COLAData(factual_data=df, label_column='target')
        
        wrong_cf = pd.DataFrame({
            'wrong_col': [1, 2, 3]
        })
        
        with pytest.raises(ValueError, match="must match"):
            data.add_counterfactuals(wrong_cf)
    
    def test_to_numpy_factual_features(self):
        """Test converting features to numpy array."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        data = COLAData(factual_data=df, label_column='target')
        
        np_array = data.to_numpy_factual_features()
        assert isinstance(np_array, np.ndarray)
        assert np_array.shape == (3, 2)  # 3 samples, 2 features
    
    def test_get_feature_count(self):
        """Test get_feature_count."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        data = COLAData(factual_data=df, label_column='target')
        
        assert data.get_feature_count() == 2
    
    def test_get_sample_count(self):
        """Test get_sample_count."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        data = COLAData(factual_data=df, label_column='target')
        
        assert data.get_sample_count() == 3
    
    def test_summary(self):
        """Test summary method."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        data = COLAData(factual_data=df, label_column='target')
        
        summary = data.summary()
        assert isinstance(summary, dict)
        assert 'factual_samples' in summary
        assert 'feature_count' in summary
        assert 'label_column' in summary
        assert summary['factual_samples'] == 3
        assert summary['feature_count'] == 2
    
    def test_str_repr(self):
        """Test string representation."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        data = COLAData(factual_data=df, label_column='target')
        
        str_repr = str(data)
        assert 'COLAData' in str_repr
        assert 'target' in str_repr


class TestCOLADataWithCounterfactual:
    """Tests for COLAData with counterfactual data."""
    
    def test_counterfactual_features(self):
        """Test getting counterfactual features."""
        factual = pd.DataFrame({
            'feature1': [1, 2],
            'feature2': [3, 4],
            'target': [0, 1]
        })
        counterfactual = pd.DataFrame({
            'feature1': [10, 20],
            'feature2': [30, 40],
            'target': [1, 0]
        })
        
        data = COLAData(
            factual_data=factual,
            label_column='target',
            counterfactual_data=counterfactual
        )
        
        cf_features = data.get_counterfactual_features()
        assert isinstance(cf_features, pd.DataFrame)
        assert 'target' not in cf_features.columns
        assert len(cf_features) == 2
    
    def test_counterfactual_labels(self):
        """Test getting counterfactual labels."""
        factual = pd.DataFrame({
            'feature1': [1, 2],
            'target': [0, 1]
        })
        counterfactual = pd.DataFrame({
            'feature1': [10, 20],
            'target': [1, 0]
        })
        
        data = COLAData(
            factual_data=factual,
            label_column='target',
            counterfactual_data=counterfactual
        )
        
        cf_labels = data.get_counterfactual_labels()
        assert isinstance(cf_labels, pd.Series)
        assert list(cf_labels) == [1, 0]

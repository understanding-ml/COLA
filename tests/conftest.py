"""Pytest configuration and fixtures."""

import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y
    
    return df

@pytest.fixture
def sample_model(sample_data):
    """Create a sample model for testing."""
    from sklearn.ensemble import RandomForestClassifier
    
    X = sample_data.drop(columns=['target'])
    y = sample_data['target']
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    return model

@pytest.fixture
def factual_instances(sample_data):
    """Create factual instances for testing."""
    df = sample_data[sample_data['target'] == 1].head(5)
    return df.drop(columns=['target']).values

@pytest.fixture
def cola_data(sample_data):
    """Setup COLA data interface."""
    from xai_cola.data import COLAData
    
    # COLAData expects data WITH target column
    data = COLAData(factual_data=sample_data, label_column='target')
    return data


"""Tests for model interface module."""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xai_cola.ml_model_interface import Model

class TestModelInterface:
    """Tests for Model interface."""
    
    def test_sklearn_model_init(self):
        """Test initialization with sklearn model."""
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        model = Model(model=clf, backend="sklearn")
        
        assert model is not None
        assert model.backend == "sklearn"
    
    def test_model_predict(self, sample_model, sample_data):
        """Test model prediction."""
        from sklearn.linear_model import LogisticRegression
        
        X = sample_data.drop(columns=['target'])
        y = sample_data['target']
        
        clf = LogisticRegression(random_state=42)
        clf.fit(X, y)
        
        model = Model(model=clf, backend="sklearn")
        
        predictions = model.predict(X.values)
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_model_predict_proba(self, sample_model, sample_data):
        """Test model prediction probabilities."""
        from sklearn.linear_model import LogisticRegression
        
        X = sample_data.drop(columns=['target'])
        y = sample_data['target']
        
        clf = LogisticRegression(random_state=42)
        clf.fit(X, y)
        
        model = Model(model=clf, backend="sklearn")
        
        proba = model.predict_proba(X.values)
        assert len(proba) == len(X)
        assert all(0 <= p <= 1 for p in proba)


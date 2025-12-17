"""
Model Training Module
Trains regression, classification, clustering, and PCA models for stock prediction.
Now includes stationary features to minimize data drift.
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingRegressor, VotingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import json


class ModelTrainer:
    """Trains and saves ML models for stock prediction."""
    
    def __init__(self, output_dir: str = "models", metrics_dir: str = "reports"):
        """
        Initialize ModelTrainer.
        
        Args:
            output_dir: Directory to save trained models
            metrics_dir: Directory to save evaluation metrics
        """
        self.output_dir = output_dir
        self.metrics_dir = metrics_dir
        self.metrics = {}
        
        # Ensure directories exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Feature columns for training - includes stationary features for drift minimization
        # Primary features: technical indicators
        self.base_feature_cols = ['sma_20', 'sma_50', 'rsi', 'macd']
        
        # Stationary features: reduce drift by using normalized/relative values
        self.stationary_feature_cols = ['log_return', 'pct_return', 'volatility_20', 'price_zscore']
        
        # Combined feature set
        self.feature_cols = self.base_feature_cols + self.stationary_feature_cols
    
    def _get_available_features(self, df: pd.DataFrame) -> list:
        """Get only features that exist in the DataFrame."""
        available = [col for col in self.feature_cols if col in df.columns]
        missing = [col for col in self.feature_cols if col not in df.columns]
        
        if missing:
            print(f"   ⚠️ Missing features (using available): {missing}")
        
        # Fallback to base features if no stationary features
        if len(available) == 0:
            available = [col for col in self.base_feature_cols if col in df.columns]
        
        print(f"   ✅ Using features: {available}")
        return available
    
    def train_regression(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Train regression model to predict next day's closing price."""
        print("Training Regression Model...")
        
        # Get available features (handles missing stationary features gracefully)
        features = self._get_available_features(train_df)
        
        # Prepare features and targets
        X_train = train_df[features].values
        y_train = train_df['target_price'].values
        
        X_test = test_df[features].values
        y_test = test_df['target_price'].values
        
        # Create ensemble regressor
        regressor = VotingRegressor([
            ('lr', LinearRegression()),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))
        ])
        
        # Train
        regressor.fit(X_train, y_train)
        
        # Evaluate
        y_pred = regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        self.metrics['regression'] = {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse)
        }
        
        print(f"Regression - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        # Save model
        model_path = os.path.join(self.output_dir, "regression_model.pkl")
        joblib.dump(regressor, model_path)
        print(f"Regression model saved to {model_path}")
    
    def train_classification(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Train classification model to predict next day's direction (UP/DOWN)."""
        print("Training Classification Model...")
        
        # Get available features (handles missing stationary features gracefully)
        features = self._get_available_features(train_df)
        
        # Prepare features and targets
        X_train = train_df[features].values
        y_train = train_df['target_direction'].values
        
        X_test = test_df[features].values
        y_test = test_df['target_direction'].values
        
        # Create ensemble classifier with soft voting
        classifier = VotingClassifier([
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('svc', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
        ], voting='soft')
        
        # Train
        classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.metrics['classification'] = {
            'accuracy': float(accuracy),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        print(f"Classification - Accuracy: {accuracy:.4f}")
        
        # Save model
        model_path = os.path.join(self.output_dir, "classification_model.pkl")
        joblib.dump(classifier, model_path)
        print(f"Classification model saved to {model_path}")
    
    def train_clustering(self, df: pd.DataFrame):
        """Train K-Means clustering model for market regime detection."""
        print("Training Clustering Model...")
        
        # Use volatility and RSI for clustering
        # Calculate volatility if not present
        if 'volatility' not in df.columns:
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            df = df.dropna()
        
        # Prepare features for clustering
        cluster_features = df[['volatility', 'rsi']].values
        
        # Train K-Means with 3 clusters
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(cluster_features)
        
        # Save model
        model_path = os.path.join(self.output_dir, "clustering_model.pkl")
        joblib.dump(kmeans, model_path)
        print(f"Clustering model saved to {model_path}")
    
    def train_pca(self, df: pd.DataFrame):
        """Train PCA model for dimensionality reduction and visualization."""
        print("Training PCA Model...")
        
        # Prepare features
        pca_features = df[self.feature_cols].values
        
        # Train PCA to reduce to 2 components
        pca = PCA(n_components=2, random_state=42)
        pca.fit(pca_features)
        
        # Save model
        model_path = os.path.join(self.output_dir, "pca_model.pkl")
        joblib.dump(pca, model_path)
        print(f"PCA model saved to {model_path}")
        
        # Store explained variance
        self.metrics['pca'] = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'total_variance_explained': float(sum(pca.explained_variance_ratio_))
        }
    
    def save_metrics(self):
        """Save evaluation metrics to JSON file."""
        metrics_path = os.path.join(self.metrics_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {metrics_path}")

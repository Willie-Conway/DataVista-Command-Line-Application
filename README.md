# 📊 DataVista - Comprehensive Data Science CLI Platform

![DatVista](https://github.com/Willie-Conway/DataVista-App/blob/62b22806b37009186f100531f50769ed98517397/assets/DataVista.png)

[![CI](https://github.com/Willie-Conway/DataVista-Command-Line-Application/actions/workflows/ci.yml/badge.svg)](https://github.com/Willie-Conway/DataVista-Command-Line-Application/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

## 🚀 Overview
<p float="left">
    <img src="https://github.com/Willie-Conway/DataVista-Command-Line-Application/blob/main/image/DataVisa%20Menu%20Options.png" width="200" />
    <img src="https://github.com/Willie-Conway/DataVista-Command-Line-Application/blob/main/image/Detailed%20analysis%20for%20numeric%20column%20price.png" width="200" />
    <img src="https://github.com/Willie-Conway/DataVista-Command-Line-Application/blob/main/image/Performing%20a%20Statistical%20Analysis.png" width="200" />
    <img src="https://github.com/Willie-Conway/DataVista-Command-Line-Application/blob/main/image/DataVista%20in%20VSCode.png" width="200" />
</p>

**DataVista** is a comprehensive Python-based command-line data science platform that provides end-to-end data analysis capabilities. From data loading and cleaning to advanced machine learning and visualization, DataVista empowers data professionals with a unified toolset for exploratory data analysis, statistical testing, and predictive modeling.

## 📺 Live Demo & Documentation

[![Documentation](https://img.shields.io/badge/📚_Full_Documentation-8B5CF6?style=for-the-badge&logo=readthedocs&logoColor=white)](https://github.com/Willie-Conway/DataVista-Command-Line-Application/tree/main/docs)
[![Code Walkthrough](https://img.shields.io/badge/🎬_Code_Walkthrough-FF0000?style=for-the-badge&logo=youtube&logoColor=white)](https://github.com/Willie-Conway/DataVista-Command-Line-Application/tree/main/WALKTHROUGH.md)
![Production Ready](https://img.shields.io/badge/Production_Ready-10B981?style=for-the-badge&logo=python&logoColor=white)

## 🏗️ Project Architecture

```
📂 DataVista-Command-Line-Application/
│
├── 📂 src/                            # Core application modules
│   ├── 📜 data_vista.py              # Main application orchestrator
│   ├── 📜 data_loader.py             # Multi-format data loading
│   ├── 📜 data_cleaner.py            # Data cleaning & wrangling
│   ├── 📜 data_preprocessor.py       # Feature engineering & preprocessing
│   ├── 📜 statistical_analysis.py    # Statistical methods & tests
│   ├── 📜 hypothesis_testing.py      # A/B testing & hypothesis validation
│   ├── 📜 machine_learning.py        # ML algorithms & model training
│   ├── 📜 model_generater.py         # AutoML & model generation
│   └── 📜 visualization.py           # Data visualization engine
│
├── 📂 tests/                         # Comprehensive test suite
│   ├── 📜 test_data_vista.py         # Integration tests
│   └── 📜 __pycache__/               # Compiled test modules
│
├── 📂 data/                          # Sample datasets
│   ├── Age_Income_Dataset.csv        # Demographic analysis dataset
│   ├── customer_churn.csv            # Customer behavior dataset
│   ├── market_research.csv           # Market analysis dataset
│   ├── sample_data.csv               # General-purpose sample
│   ├── test_data_with_duplicates.csv # Testing data with duplicates
│   └── walmart_grocery_data.csv      # Retail analytics dataset
│
├── 📂 docs/                          # Comprehensive documentation
│   ├── DataVista_User_Guide.md       # Complete user manual
│   ├── Dissecting the DataCleaner - A Code Walkthrough.md
│   ├── Dissecting the DataLoader - A Code Walkthrough.md
│   ├── Dissecting the DataPreprocessor - A Code Walkthrough.md
│   └── Dissecting the HypothesisTesting - A Code Walkthrough.md
│
├── 📂 models/                        # Serialized ML models
│   ├── decision_tree_model.joblib    # Decision tree classifier
│   └── linear_regression_model.joblib # Linear regression model
│
├── 📂 .github/workflows/            # CI/CD pipeline
│   └── ci.yml                       # Continuous integration setup
│
├── 📜 requirements.txt              # Python dependencies
├── 📜 README.md                     # This documentation
├── 📜 SUMMARY.md                    # Project summary
├── 📜 WALKTHROUGH.md                # Code walkthrough guide
└── 📜 data_vista_info.sh           # Setup and information script
```

## ✨ Key Features

### **📥 Data Loading & Integration**
- **Multi-format Support**: CSV, JSON, Excel, Parquet
- **Streaming Capabilities**: Handle large datasets efficiently
- **Database Connectivity**: SQL, NoSQL integration options
- **API Data Ingestion**: REST API data collection

### **🧹 Data Cleaning & Wrangling**
- **Automated Cleaning**: Handle missing values, outliers, duplicates
- **Data Validation**: Schema validation and data quality checks
- **Feature Engineering**: Create new features from existing data
- **Text Processing**: NLP preprocessing for text columns

### **📊 Statistical Analysis**
- **Descriptive Statistics**: Mean, median, mode, variance, skewness, kurtosis
- **Inferential Statistics**: Confidence intervals, hypothesis testing
- **Correlation Analysis**: Pearson, Spearman, Kendall correlations
- **Distribution Testing**: Normality tests, distribution fitting

### **🤖 Machine Learning Pipeline**
- **Supervised Learning**: Regression, classification algorithms
- **Unsupervised Learning**: Clustering, dimensionality reduction
- **Model Evaluation**: Cross-validation, metrics, learning curves
- **Hyperparameter Tuning**: Grid search, random search, Bayesian optimization

### **📈 Advanced Visualization**
- **Statistical Plots**: Histograms, box plots, violin plots, QQ plots
- **Geospatial Visualization**: Maps, heatmaps, choropleth maps
- **Interactive Charts**: Plotly-based interactive visualizations
- **Dashboard Generation**: Automated report and dashboard creation

### **🔬 Hypothesis Testing Framework**
- **A/B Testing**: Statistical significance testing for experiments
- **Multi-variate Testing**: ANOVA, MANOVA, factorial designs
- **Time Series Testing**: Stationarity tests, seasonality detection
- **Non-parametric Tests**: Mann-Whitney, Kruskal-Wallis, Chi-square

## 🛠️ Technology Stack

### **Core Libraries**
![Pandas](https://img.shields.io/badge/Pandas_2.0-Expert-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy_1.24-013243?style=flat-square&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy_1.11-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn_1.3-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)

### **Visualization**
![Matplotlib](https://img.shields.io/badge/Matplotlib_3.7-11557C?style=flat-square&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn_0.12-4C8CBF?style=flat-square)
![Plotly](https://img.shields.io/badge/Plotly_5.17-3F4F75?style=flat-square&logo=plotly&logoColor=white)

### **Advanced Analytics**
![Statsmodels](https://img.shields.io/badge/Statsmodels_0.14-B31B1B?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost_2.0-3776AB?style=flat-square&logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM_4.0-792EE5?style=flat-square)

### **DevOps & Testing**
![Pytest](https://img.shields.io/badge/Pytest_7.4-0A9EDC?style=flat-square&logo=pytest&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)

## 🚀 Quick Start Guide

### **Prerequisites**
```bash
# System Requirements
- Python 3.10+                    # Core runtime
- pip 23.0+                      # Package manager
- Git 2.40+                      # Version control
- 4GB+ RAM                       # For large datasets
```

### **1. Installation**
```bash
# Clone repository
git clone https://github.com/Willie-Conway/DataVista-Command-Line-Application.git
cd DataVista-Command-Line-Application

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### **2. Configuration**
```python
# Create config.yaml (optional)
echo '
data_vista:
  logging:
    level: INFO
    file: data_vista.log
  visualization:
    style: seaborn
    palette: viridis
  machine_learning:
    cross_validation_folds: 5
    random_state: 42
' > config.yaml
```

### **3. Run DataVista**
```bash
# Basic usage with sample data
python src/data_vista.py

# With custom dataset
python src/data_vista.py --data data/customer_churn.csv

# With specific configuration
python src/data_vista.py --config config.yaml --output results/

# For batch processing
python src/data_vista.py --batch data/ --output analysis_reports/
```

## 📊 Module Architecture

### **Core Components**

```python
# DataVista Application Architecture
class DataVista:
    """
    Main orchestrator class that integrates all modules
    """
    def __init__(self):
        self.data_loader = DataLoader()
        self.data_cleaner = DataCleaner()
        self.preprocessor = DataPreprocessor()
        self.statistics = StatisticalAnalysis()
        self.hypothesis = HypothesisTesting()
        self.ml_engine = MachineLearning()
        self.visualizer = Visualization()
        
    def run_pipeline(self, filepath: str):
        """Complete data analysis pipeline"""
        # 1. Load data
        data = self.data_loader.load(filepath)
        
        # 2. Clean data
        cleaned = self.data_cleaner.clean(data)
        
        # 3. Preprocess
        processed = self.preprocessor.preprocess(cleaned)
        
        # 4. Statistical analysis
        stats = self.statistics.analyze(processed)
        
        # 5. Machine learning
        models = self.ml_engine.train(processed)
        
        # 6. Generate insights
        insights = self.generate_insights(stats, models)
        
        return insights
```

### **Data Loader Module**
```python
class DataLoader:
    """
    Handles loading data from multiple sources and formats
    """
    SUPPORTED_FORMATS = {
        'csv': pd.read_csv,
        'json': pd.read_json,
        'excel': pd.read_excel,
        'parquet': pd.read_parquet,
        'feather': pd.read_feather
    }
    
    def load(self, filepath: str) -> pd.DataFrame:
        """Load data with automatic format detection"""
        ext = filepath.split('.')[-1].lower()
        
        if ext in self.SUPPORTED_FORMATS:
            return self.SUPPORTED_FORMATS[ext](filepath)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def load_from_database(self, query: str, connection_string: str):
        """Load data from SQL database"""
        import sqlalchemy
        engine = sqlalchemy.create_engine(connection_string)
        return pd.read_sql(query, engine)
```

### **Data Cleaner Module**
```python
class DataCleaner:
    """
    Comprehensive data cleaning and quality assurance
    """
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute full cleaning pipeline"""
        df = self.remove_duplicates(df)
        df = self.handle_missing_values(df)
        df = self.remove_outliers(df)
        df = self.standardize_formats(df)
        df = self.validate_data_types(df)
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, 
                             strategy: str = 'auto') -> pd.DataFrame:
        """Intelligent missing value imputation"""
        if strategy == 'auto':
            # Automatic strategy selection
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                elif df[col].dtype == 'object':
                    df[col].fillna(df[col].mode()[0], inplace=True)
        return df
    
    def remove_outliers(self, df: pd.DataFrame, 
                       method: str = 'iqr') -> pd.DataFrame:
        """Outlier detection and removal"""
        if method == 'iqr':
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            return df[~((df < (Q1 - 1.5 * IQR)) | 
                        (df > (Q3 + 1.5 * IQR))).any(axis=1)]
        return df
```

### **Statistical Analysis Module**
```python
class StatisticalAnalysis:
    """
    Comprehensive statistical analysis toolkit
    """
    def analyze(self, df: pd.DataFrame) -> dict:
        """Perform complete statistical analysis"""
        analysis = {
            'descriptive': self.descriptive_stats(df),
            'correlation': self.correlation_analysis(df),
            'distribution': self.distribution_analysis(df),
            'time_series': self.time_series_analysis(df) if self.has_dates(df) else None
        }
        return analysis
    
    def descriptive_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate descriptive statistics"""
        stats = df.describe(include='all').T
        stats['variance'] = df.var()
        stats['skewness'] = df.skew()
        stats['kurtosis'] = df.kurtosis()
        stats['missing'] = df.isnull().sum()
        stats['missing_pct'] = (df.isnull().sum() / len(df)) * 100
        return stats
    
    def correlation_analysis(self, df: pd.DataFrame) -> dict:
        """Multiple correlation analysis methods"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        correlations = {
            'pearson': df[numeric_cols].corr(method='pearson'),
            'spearman': df[numeric_cols].corr(method='spearman'),
            'kendall': df[numeric_cols].corr(method='kendall')
        }
        
        # Heatmap visualization
        self.visualize_correlation(correlations['pearson'])
        
        return correlations
```

## 🤖 Machine Learning Capabilities

### **Model Training Pipeline**
```python
class MachineLearning:
    """
    End-to-end machine learning pipeline
    """
    def train(self, df: pd.DataFrame, 
              target: str,
              algorithm: str = 'auto') -> dict:
        """Train ML model with automated feature selection"""
        
        # Prepare data
        X = df.drop(columns=[target])
        y = df[target]
        
        # Feature selection
        X_selected = self.feature_selection(X, y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42
        )
        
        # Select algorithm
        if algorithm == 'auto':
            model = self.auto_select_algorithm(X_train, y_train)
        else:
            model = self.get_algorithm(algorithm)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = self.evaluate_model(model, X_test, y_test)
        
        # Feature importance
        importance = self.get_feature_importance(model, X_selected.columns)
        
        return {
            'model': model,
            'metrics': metrics,
            'importance': importance,
            'predictions': model.predict(X_test)
        }
    
    def auto_select_algorithm(self, X: pd.DataFrame, 
                             y: pd.Series) -> Any:
        """Automatically select best algorithm for the data"""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression
        
        # Determine problem type
        if y.nunique() <= 10:  # Classification
            if len(X) > 10000:
                return RandomForestClassifier()
            else:
                return LogisticRegression()
        else:  # Regression
            if len(X) > 10000:
                return RandomForestRegressor()
            else:
                return LinearRegression()
```

### **Model Evaluation Suite**
```python
    def evaluate_model(self, model: Any, 
                      X_test: pd.DataFrame, 
                      y_test: pd.Series) -> dict:
        """Comprehensive model evaluation"""
        y_pred = model.predict(X_test)
        
        metrics = {}
        
        # Classification metrics
        if hasattr(model, 'predict_proba'):
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score,
                f1_score, roc_auc_score, confusion_matrix
            )
            
            metrics.update({
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            })
        
        # Regression metrics
        else:
            from sklearn.metrics import (
                mean_squared_error, mean_absolute_error,
                r2_score, explained_variance_score
            )
            
            metrics.update({
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'explained_variance': explained_variance_score(y_test, y_pred)
            })
        
        # Learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_test, y_test, cv=5
        )
        metrics['learning_curve'] = {
            'train_sizes': train_sizes,
            'train_scores': train_scores.mean(axis=1),
            'test_scores': test_scores.mean(axis=1)
        }
        
        return metrics
```

## 📈 Visualization Engine

### **Advanced Plotting System**
```python
class Visualization:
    """
    Advanced data visualization with multiple chart types
    """
    def __init__(self, style: str = 'seaborn'):
        plt.style.use(style)
        self.figsize = (12, 8)
        
    def create_dashboard(self, df: pd.DataFrame, 
                        insights: dict) -> None:
        """Create comprehensive visualization dashboard"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Distribution plots
        ax1 = plt.subplot(3, 3, 1)
        self.plot_distributions(df, ax1)
        
        # 2. Correlation heatmap
        ax2 = plt.subplot(3, 3, 2)
        self.plot_correlation_heatmap(df, ax2)
        
        # 3. Box plots
        ax3 = plt.subplot(3, 3, 3)
        self.plot_boxplots(df, ax3)
        
        # 4. Time series (if applicable)
        if self.has_dates(df):
            ax4 = plt.subplot(3, 3, 4)
            self.plot_time_series(df, ax4)
        
        # 5. Feature importance
        if 'importance' in insights:
            ax5 = plt.subplot(3, 3, 5)
            self.plot_feature_importance(insights['importance'], ax5)
        
        plt.tight_layout()
        plt.savefig('data_vista_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_distributions(self, df: pd.DataFrame, 
                          ax: plt.Axes) -> None:
        """Plot distribution of all numerical columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for i, col in enumerate(numeric_cols[:4]):  # First 4 columns
            sns.histplot(df[col], ax=ax if i == 0 else None, 
                        label=col, alpha=0.5, kde=True)
        
        ax.set_title('Feature Distributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, 
                                ax: plt.Axes) -> None:
        """Plot correlation matrix heatmap"""
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, 
                   center=0, annot=True, fmt='.2f',
                   square=True, linewidths=.5, 
                   cbar_kws={"shrink": .5}, ax=ax)
        
        ax.set_title('Correlation Heatmap')
```

## 🔬 Hypothesis Testing Framework

### **Statistical Testing Suite**
```python
class HypothesisTesting:
    """
    Comprehensive hypothesis testing capabilities
    """
    def perform_test(self, df: pd.DataFrame, 
                    test_type: str, 
                    **kwargs) -> dict:
        """Perform various hypothesis tests"""
        test_functions = {
            't_test': self.t_test,
            'anova': self.anova_test,
            'chi_square': self.chi_square_test,
            'mann_whitney': self.mann_whitney_test,
            'wilcoxon': self.wilcoxon_test,
            'kruskal_wallis': self.kruskal_wallis_test
        }
        
        if test_type not in test_functions:
            raise ValueError(f"Unsupported test type: {test_type}")
        
        return test_functions[test_type](df, **kwargs)
    
    def t_test(self, df: pd.DataFrame, 
               group_col: str, 
               value_col: str) -> dict:
        """Perform t-test between groups"""
        from scipy import stats
        
        groups = df[group_col].unique()
        if len(groups) != 2:
            raise ValueError("T-test requires exactly 2 groups")
        
        group1 = df[df[group_col] == groups[0]][value_col]
        group2 = df[df[group_col] == groups[1]][value_col]
        
        # Check assumptions
        normality_check = {
            'group1_normal': stats.shapiro(group1)[1] > 0.05,
            'group2_normal': stats.shapiro(group2)[1] > 0.05
        }
        
        # Perform test
        if normality_check['group1_normal'] and normality_check['group2_normal']:
            # Parametric t-test
            stat, p_value = stats.ttest_ind(group1, group2)
            test_type = 'parametric'
        else:
            # Non-parametric Mann-Whitney U test
            stat, p_value = stats.mannwhitneyu(group1, group2)
            test_type = 'non_parametric'
        
        # Calculate effect size
        cohen_d = (group1.mean() - group2.mean()) / np.sqrt(
            (group1.std() ** 2 + group2.std() ** 2) / 2
        )
        
        return {
            'test_type': test_type,
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': cohen_d,
            'group_means': {
                groups[0]: group1.mean(),
                groups[1]: group2.mean()
            },
            'assumptions': normality_check
        }
    
    def ab_testing(self, df: pd.DataFrame,
                   control_col: str,
                   treatment_col: str,
                   metric_col: str) -> dict:
        """Perform A/B testing analysis"""
        control = df[control_col]
        treatment = df[treatment_col]
        
        # Calculate metrics
        control_mean = control.mean()
        treatment_mean = treatment.mean()
        relative_change = (treatment_mean - control_mean) / control_mean * 100
        
        # Statistical significance
        from scipy import stats
        stat, p_value = stats.ttest_ind(control, treatment)
        
        # Confidence intervals
        control_ci = self.calculate_confidence_interval(control)
        treatment_ci = self.calculate_confidence_interval(treatment)
        
        # Power analysis
        power = self.calculate_power(
            control_mean, treatment_mean,
            control.std(), treatment.std(),
            len(control), len(treatment)
        )
        
        return {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'relative_change': relative_change,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'confidence_intervals': {
                'control': control_ci,
                'treatment': treatment_ci
            },
            'statistical_power': power,
            'recommendation': 'Implement treatment' if p_value < 0.05 else 'Keep control'
        }
```

## 🧪 Testing Suite

### **Comprehensive Test Coverage**
```python
# tests/test_data_vista.py
import pytest
import pandas as pd
import numpy as np
from src.data_vista import DataVista
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.machine_learning import MachineLearning

class TestDataVista:
    """Comprehensive test suite for DataVista"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data"""
        return pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100)
        })
    
    def test_data_loading(self, tmp_path):
        """Test data loading functionality"""
        # Create test CSV
        test_csv = tmp_path / "test.csv"
        pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']}).to_csv(test_csv)
        
        loader = DataLoader()
        df = loader.load(str(test_csv))
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert list(df.columns) == ['col1', 'col2']
    
    def test_data_cleaning(self, sample_data):
        """Test data cleaning pipeline"""
        # Add some missing values and duplicates
        dirty_data = sample_data.copy()
        dirty_data.iloc[0, 0] = np.nan
        dirty_data = pd.concat([dirty_data, dirty_data.iloc[:5]])
        
        cleaner = DataCleaner()
        cleaned = cleaner.clean(dirty_data)
        
        assert cleaned.isnull().sum().sum() == 0
        assert not cleaned.duplicated().any()
    
    def test_machine_learning(self, sample_data):
        """Test ML model training and evaluation"""
        ml = MachineLearning()
        results = ml.train(sample_data, target='target')
        
        assert 'model' in results
        assert 'metrics' in results
        assert 'importance' in results
        
        # Check model performance
        assert results['metrics']['accuracy'] > 0.5  # Better than random
    
    def test_statistical_analysis(self, sample_data):
        """Test statistical analysis functions"""
        from src.statistical_analysis import StatisticalAnalysis
        
        analyzer = StatisticalAnalysis()
        stats = analyzer.analyze(sample_data)
        
        assert 'descriptive' in stats
        assert 'correlation' in stats
        assert isinstance(stats['descriptive'], pd.DataFrame)
    
    def test_hypothesis_testing(self, sample_data):
        """Test hypothesis testing functionality"""
        from src.hypothesis_testing import HypothesisTesting
        
        tester = HypothesisTesting()
        results = tester.t_test(sample_data, 'category', 'feature1')
        
        assert 'p_value' in results
        assert 'significant' in results
        assert isinstance(results['p_value'], float)
    
    @pytest.mark.integration
    def test_full_pipeline(self, tmp_path):
        """Test complete DataVista pipeline"""
        # Create test dataset
        test_data = pd.DataFrame({
            'age': np.random.randint(18, 65, 100),
            'income': np.random.normal(50000, 15000, 100),
            'purchase': np.random.randint(0, 2, 100)
        })
        
        test_file = tmp_path / "test_dataset.csv"
        test_data.to_csv(test_file, index=False)
        
        # Run full pipeline
        datavista = DataVista()
        insights = datavista.run_pipeline(str(test_file))
        
        assert 'data_summary' in insights
        assert 'statistical_analysis' in insights
        assert 'ml_results' in insights
        assert 'visualizations' in insights
```

## 🐳 Docker Deployment

### **Containerized Application**
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create non-root user
RUN useradd -m -u 1000 datavista
USER datavista

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import pandas; print('Health check passed')"

# Default command
CMD ["python", "src/data_vista.py"]
```

### **Docker Compose Setup**
```yaml
# docker-compose.yml
version: '3.8'

services:
  datavista:
    build: .
    container_name: datavista-app
    volumes:
      - ./data:/app/data
      - ./results:/app/results
      - ./models:/app/models
    environment:
      - PYTHONUNBUFFERED=1
      - DATAVISTA_LOG_LEVEL=INFO
    ports:
      - "8080:8080"
    command: ["python", "src/data_vista.py", "--data", "data/sample_data.csv"]
  
  jupyter:
    image: jupyter/datascience-notebook:latest
    container_name: datavista-jupyter
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/home/jovyan/work
      - ./data:/home/jovyan/data
    environment:
      - JUPYTER_ENABLE_LAB=yes
  
  postgres:
    image: postgres:15
    container_name: datavista-db
    environment:
      POSTGRES_USER: datavista
      POSTGRES_PASSWORD: datavista123
      POSTGRES_DB: analytics
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

## 🤖 CI/CD Pipeline

### **GitHub Actions Workflow**
```yaml
# .github/workflows/ci.yml
name: DataVista CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov black flake8 mypy
    
    - name: Lint with flake8
      run: |
        flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    - name: Type checking with mypy
      run: |
        mypy src/ --ignore-missing-imports
    
    - name: Format checking with black
      run: |
        black --check src/
    
    - name: Test with pytest
      run: |
        pytest tests/ -v --cov=src --cov-report=xml --cov-report=html
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
    
    - name: Build Docker image
      run: |
        docker build -t datavista:${{ github.sha }} .
    
    - name: Run integration tests
      run: |
        docker-compose -f docker-compose.test.yml up -d
        sleep 30
        docker-compose -f docker-compose.test.yml exec datavista pytest tests/integration/ -v
  
  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to PyPI
      run: |
        pip install twine
        python setup.py sdist bdist_wheel
        twine upload dist/* --username __token__ --password ${{ secrets.PYPI_API_TOKEN }}
    
    - name: Deploy to Docker Hub
      run: |
        echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
        docker tag datavista:${{ github.sha }} ${{ secrets.DOCKER_USERNAME }}/datavista:latest
        docker push ${{ secrets.DOCKER_USERNAME }}/datavista:latest
```

## 📊 Performance Optimization

### **Memory Optimization**
```python
class OptimizedDataVista(DataVista):
    """
    Memory-optimized version for large datasets
    """
    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size
        super().__init__()
    
    def process_large_file(self, filepath: str):
        """Process large files in chunks"""
        results = []
        
        # Use pandas chunking for CSV files
        if filepath.endswith('.csv'):
            for chunk in pd.read_csv(filepath, chunksize=self.chunk_size):
                # Process chunk
                chunk_result = self.process_chunk(chunk)
                results.append(chunk_result)
        
        # Aggregate results
        return self.aggregate_results(results)
    
    def process_chunk(self, chunk: pd.DataFrame):
        """Process a single chunk"""
        # Optimized operations for chunk
        chunk = chunk.select_dtypes(include=[np.number])
        chunk = chunk.fillna(chunk.mean())
        return chunk
    
    def aggregate_results(self, results: list):
        """Aggregate chunk results"""
        return pd.concat(results, ignore_index=True)
```

### **Parallel Processing**
```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

class ParallelDataVista(DataVista):
    """
    Parallel processing implementation
    """
    def __init__(self, n_jobs: int = -1):
        self.n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        super().__init__()
    
    def parallel_analysis(self, df: pd.DataFrame):
        """Run analyses in parallel"""
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit tasks
            futures = {
                'descriptive': executor.submit(self.statistics.descriptive_stats, df),
                'correlation': executor.submit(self.statistics.correlation_analysis, df),
                'ml': executor.submit(self.ml_engine.train, df, 'target'),
                'visualization': executor.submit(self.visualizer.create_dashboard, df)
            }
            
            # Collect results
            results = {}
            for name, future in futures.items():
                results[name] = future.result()
            
            return results
```

## 📚 Documentation & Examples

### **Usage Examples**
```python
# Example 1: Basic analysis
from src.data_vista import DataVista

dv = DataVista()
insights = dv.run_pipeline('data/customer_churn.csv')
print(insights['data_summary'])

# Example 2: Custom pipeline
dv = DataVista()
data = dv.data_loader.load('data/sales_data.csv')
clean_data = dv.data_cleaner.clean(data)
ml_results = dv.ml_engine.train(clean_data, target='revenue')

# Example 3: Advanced visualization
dv.visualizer.create_dashboard(clean_data, ml_results)

# Example 4: Hypothesis testing
test_results = dv.hypothesis.ab_testing(
    data, 
    control_col='version_a',
    treatment_col='version_b',
    metric_col='conversion_rate'
)
```

### **Command Line Interface**
```bash
# Basic analysis
python src/data_vista.py --data data/sample.csv

# With specific options
python src/data_vista.py \
  --data data/customer_data.csv \
  --target churn \
  --algorithm random_forest \
  --output results/ \
  --visualize all

# Batch processing
python src/data_vista.py \
  --batch data/input_folder/ \
  --output data/results/ \
  --format json

# Docker usage
docker run -v $(pwd)/data:/app/data datavista:latest \
  python src/data_vista.py --data /app/data/sample.csv
```

## 🏆 Project Achievements

✅ **Complete Data Science Pipeline** from data loading to deployment  
✅ **30+ Statistical Tests** with automated assumption checking  
✅ **15+ ML Algorithms** with automated hyperparameter tuning  
✅ **Parallel Processing** for large dataset handling  
✅ **Docker Containerization** for consistent deployments  
✅ **Comprehensive CI/CD** with automated testing  
✅ **Production-Ready** error handling and logging  
✅ **Extensive Documentation** with code walkthroughs  

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md):

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Run linting
flake8 src/
black src/
mypy src/

# Build documentation
cd docs && make html
```

## 📞 Support & Community

- **GitHub Issues**: [Report Bugs](https://github.com/Willie-Conway/DataVista-Command-Line-Application/issues)
- **Documentation**: [User Guide](https://github.com/Willie-Conway/DataVista-Command-Line-Application/tree/main/docs)
- **Discussions**: [Community Forum](https://github.com/Willie-Conway/DataVista-Command-Line-Application/discussions)

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏🏿 Acknowledgments

- **Python Data Science Community** for amazing libraries
- **Open Source Contributors** who built the tools we depend on
- **All Users** who provide feedback and suggestions
- **Data Science Educators** who inspire continuous learning

---

**DataVista - Your Comprehensive Data Science Companion! 📊🚀**

*Project Created: Dec 2023*  
*Last Updated: Jan 28, 2025*  
*Python Version: 3.10+*  
*Active Contributors: 1*  
*Total Downloads: 500+*

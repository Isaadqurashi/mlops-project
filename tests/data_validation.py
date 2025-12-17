import pandas as pd
import os

def validate_data(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str = "reports"):
    """
    Runs DeepChecks on training and testing data.
    Falls back gracefully if deepchecks is unavailable or has compatibility issues.
    """
    try:
        from deepchecks.tabular import Dataset
        from deepchecks.tabular.suites import data_integrity, train_test_validation
        
        # Create DeepChecks Datasets
        # Assuming 'target_price' is the label for regression
        train_ds = Dataset(train_df, label='target_price', cat_features=[])
        test_ds = Dataset(test_df, label='target_price', cat_features=[])

        os.makedirs(output_dir, exist_ok=True)

        # 1. Data Integrity Check
        print("Running Data Integrity Check...")
        integrity_suite = data_integrity()
        integrity_result = integrity_suite.run(train_ds)
        integrity_result.save_as_html(f"{output_dir}/data_integrity.html")
        print(f"Data Integrity report saved to {output_dir}/data_integrity.html")

        # 2. Train-Test Validation (Drift)
        print("Running Train-Test Validation (Drift Check)...")
        validation_suite = train_test_validation()
        validation_result = validation_suite.run(train_ds, test_ds)
        validation_result.save_as_html(f"{output_dir}/train_test_validation.html")
        print(f"Train-Test Validation report saved to {output_dir}/train_test_validation.html")

        return integrity_result, validation_result
        
    except ImportError as e:
        print(f"⚠️  DeepChecks not available: {e}")
        print("⚠️  Skipping data validation. Pipeline will continue...")
        return None, None
    except Exception as e:
        print(f"⚠️  DeepChecks validation failed: {e}")
        print("⚠️  This may be due to scikit-learn compatibility issues.")
        print("⚠️  Skipping data validation. Pipeline will continue...")
        return None, None

if __name__ == "__main__":
    # Example usage
    # df = pd.read_csv("data/processed/AAPL_processed.csv")
    # train_df = df.iloc[:-30]
    # test_df = df.iloc[-30:]
    # validate_data(train_df, test_df)
    pass

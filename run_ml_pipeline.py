#! /usr/bin/env python3

import os
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from snowflake.ml.model import ModelVersion
from snowflake.ml.registry import Registry
from snowflake.snowpark import Session


class Datasets(NamedTuple):
    x: pd.DataFrame
    y: pd.Series
    train_x: pd.DataFrame
    test_x: pd.DataFrame
    train_y: pd.Series
    test_y: pd.Series


@dataclass
class MLPipeline:
    session: Session
    table: str

    @cached_property
    def df_raw(self) -> pd.DataFrame:
        return self.session.table(self.table).to_pandas().dropna()

    @cached_property
    def data(self) -> Datasets:
        print("Loading data, creating train/test datasets...")
        df = self.df_raw.copy()  # Preserve raw for post-inference join

        # 2. Encode 'GENDER' using LabelEncoder
        le_gender = LabelEncoder()
        df["GENDER"] = le_gender.fit_transform(df["GENDER"])

        # 3. Define features and target
        target_col = "CHURN"
        id_col = "CUSTOMERID"

        df_x = df.drop(columns=[target_col, id_col])
        df_y = df[target_col]
        splits = train_test_split(df_x, df_y, test_size=0.2, stratify=df_y, random_state=42)

        return Datasets(df_x, df_y, splits[0], splits[1], splits[2], splits[3])

    @property
    def ml_pipeline(self):
        # 5. Define feature types
        categorical_ohe = ["SUBSCRIPTION_TYPE"]
        categorical_ord = ["CONTRACT_LENGTH"]
        numeric_features = [col for col in self.data.x.columns if col not in categorical_ohe + categorical_ord + ["GENDER"]]

        preprocessor = ColumnTransformer(
            [
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_ohe),
                ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_ord),
                ("gender", "passthrough", ["GENDER"]),
                ("numeric", "passthrough", numeric_features),
            ]
        )

        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)),
            ]
        )

    @cached_property
    def reg(self) -> Registry:
        return Registry(session=self.session)

    @cached_property
    def model(self) -> ModelVersion:
        """
        create and log the best performing model obtained by tuning hyperparameters
        """
        MODEL_NAME = "CUSTOMER_CHURN_MODEL"

        try:
            model = self.reg.get_model(MODEL_NAME)
            print(f"Model '{MODEL_NAME}' already registered, skipping creating a new version")
            return model.version("LAST")
        except ValueError:
            pass

        print("Create a new model using training data...")
        # 8. Hyperparameter tuning
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [10, 20, None],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2],
            "classifier__max_features": ["sqrt", "log2"],
        }

        search = RandomizedSearchCV(
            self.ml_pipeline,
            param_distributions=param_grid,
            n_iter=10,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )

        # 9. Train and select best model
        search.fit(self.data.train_x, self.data.train_y)
        best_model = search.best_estimator_

        mv = self.reg.log_model(
            model_name=MODEL_NAME,
            model=best_model,
            sample_input_data=self.data.train_x.head(100),
            target_platforms=["WAREHOUSE"],
            options={"enable_explainability": True, "relax_version": True},
        )
        self.reg.show_models()

        return mv

    @cached_property
    def inferred(self) -> pd.DataFrame:
        """
        Run inference on test data using model registry, and combine the results with raw data
        """
        print("Running predict! on test data...")
        pred_df = self.model.run(self.data.test_x, function_name="predict").rename(
            columns={"output_feature_0": "CHURN_PREDICTION"}
        )
        prob_df = self.model.run(self.data.test_x, function_name="predict_proba")[["output_feature_1"]].rename(
            columns={"output_feature_1": "CHURN_PREDICTION_PROB"}
        )
        explain_df = self.model.run(self.data.test_x, function_name="explain")

        # Get full original rows corresponding to test set
        raw_test_df = self.df_raw.copy().loc[self.data.test_x.index].reset_index(drop=True)  # Includes CUSTOMERID and raw GENDER

        # Combine everything into final_df
        final_df = pd.concat(
            [raw_test_df, pred_df.reset_index(drop=True), prob_df.reset_index(drop=True), explain_df.reset_index(drop=True)],
            axis=1,
        )

        # Add actual label
        final_df["CHURN"] = self.data.test_y.reset_index(drop=True)

        print("Running model evaluation...")
        print(f"ROC AUC on test data: {roc_auc_score(final_df['CHURN'], final_df['CHURN_PREDICTION_PROB']):.4f}")
        print("Model Classification Report:\n", classification_report(final_df["CHURN"], final_df["CHURN_PREDICTION"]))

        return final_df

    def run(self):
        """
        write inferred data to Snowflake table
        """
        self.session.create_dataframe(self.inferred).write.mode("overwrite").save_as_table("CUSTOMER_CHURN_EXPLAINATION")


def get_session() -> Session:
    session = Session.get_active_session()
    if session is None:
        connection_name = os.getenv("SNOWFLAKE_DEFAULT_CONNECTION_NAME", "default")
        try:
            print(f"Creating new connection using connection name: '{connection_name}'")
            session = Session.SessionBuilder().configs({"connection_name": connection_name, "warehouse": "ADHOC2"}).create()
        except Exception as e:
            raise SystemExit(f"Connection using name '{connection_name}' failed: {str(e)}")

    session.use_database("POC")
    session.use_schema("CHURN")

    return session


def main():
    MLPipeline(get_session(), "CUSTOMER_CHURN").run()


if __name__ == "__main__":
    main()

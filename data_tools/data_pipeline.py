import pandas as pd
import uuid

class DataPipeline:
    def __init__(self):
        self.datasets = {}

    def ingest_csv(self, csv_path: str = None, json_path: str = None):
        if csv_path is not None:
            df = pd.read_csv(csv_path)
        elif json_path is not None:
            df = pd.read_json(json_path)
        else:
            raise ValueError("Must provide either csv or json path")

        if df.empty:
            raise ValueError("Dataset is empty")

        dataset_id = str(uuid.uuid4())

        df = self.normalize_columns(df)
        df = self.infer_types(df)
        df = self.clean_data(df)
        df = self.feature_engineering(df)
        schema = self.extract_schema(df)
        metadata = self.generate_metadata(df)

        self.datasets[dataset_id] = {
            "dataframe": df,
            "schema": schema,
            "metadata": metadata
        }

        return dataset_id

    def normalize_columns(self, df: pd.DataFrame):
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(r"[^\w]+", "_", regex=True)
        )

        return df

    def infer_types(self, df: pd.DataFrame):
        for col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], format="mixed")
                continue
            except:
                pass

            try:
                df[col] = pd.to_numeric(df[col])
                continue
            except:
                pass

        return df

    def clean_data(self, df: pd.DataFrame):
        df = df.dropna(how="all")
        df = df.drop_duplicates()

        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()

        return df

    def feature_engineering(self, df: pd.DataFrame):
        for col in df.select_dtypes(include="datetime").columns:
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month

        return df

    def extract_schema(self, df: pd.DataFrame):
        schema = {}
        for col in df.columns:
            dtype = str(df[col].dtype)

            if "int" in dtype or "float" in dtype:
                schema[col] = "numeric"
            elif "datetime" in dtype:
                schema[col] = "datetime"
            else:
                schema[col] = "string"

        return schema

    def generate_metadata(self, df: pd.DataFrame):
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "missing_values": df.isnull().sum().to_dict(),
        }

    def get_dataset_id(self, dataset_id: str):
        return self.datasets.get(dataset_id)
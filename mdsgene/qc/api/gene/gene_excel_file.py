import os

import pandas as pd

from mdsgene.qc.logging_config import logger


def categorize_column(column_data):
    """Categorize a column based on its data type."""
    if pd.api.types.is_string_dtype(column_data):
        return "Nominal"  # Nominal if contains strings
    elif pd.api.types.is_numeric_dtype(column_data):
        unique_values = column_data.nunique()
        if unique_values < 10:
            return "Ordinal"  # Ordinal if few unique values
        elif unique_values >= 10:
            return "Interval" if column_data.min() > 0 else "Ratio"  # Interval or Ratio
    return "Unknown"


def get_columns(file_path: str):
    try:
        # Load Excel file into DataFrame
        df = pd.read_excel(file_path)

        # Return the same format as save_file
        columns_info = [
            {"name": col, "category": categorize_column(df[col])}
            for col in df.columns
        ]

        return columns_info
    except Exception as e:
        logger.error(f"Error in get_columns: {str(e)}")
        return []


def save_file(file):
    file_path = os.path.join('excel', file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    columns_info = get_columns(file_path)

    # Return column information with categories
    return columns_info


def delete_columns(file_path: str, columns_to_delete: list[str]) -> bool:
    try:
        # Read Excel file
        df = pd.read_excel(file_path)

        # Check if columns exist
        existing_columns = set(df.columns)
        # Convert column names from objects to strings
        columns_to_delete_names = set(columns_to_delete)

        if not columns_to_delete_names.issubset(existing_columns):
            logger.error(f"Some columns don't exist: {columns_to_delete_names - existing_columns}")
            return False

        # Delete columns
        df.drop(columns=list(columns_to_delete_names), inplace=True)

        # Save file
        df.to_excel(file_path, index=False)
        return True
    except Exception as e:
        logger.error(f"Error in delete_columns: {str(e)}")
        return False

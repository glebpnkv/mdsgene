import os
from typing import List
from qc.logging_config import logger

import pandas as pd

def save_file(file):
    file_path = os.path.join('excel', file.filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())

    # Загружаем Excel файл в DataFrame
    df = pd.read_excel(file_path)

    # Определение категории для каждого столбца
    def categorize_column(column_data):
        if pd.api.types.is_string_dtype(column_data):
            return "Nominal"  # Номинальный, если содержит строки
        elif pd.api.types.is_numeric_dtype(column_data):
            unique_values = column_data.nunique()
            if unique_values < 10:
                return "Ordinal"  # Порядковый, если немного уникальных значений
            elif unique_values >= 10:
                return "Interval" if column_data.min() > 0 else "Ratio"  # Интервальный или рациональный
        return "Unknown"

    # Составляем список с информацией о столбцах
    columns_info = [
        {"name": col, "category": categorize_column(df[col])}
        for col in df.columns
    ]

    # Возвращаем информацию о столбцах с категориями
    return columns_info


def delete_columns(file_path: str, columns_to_delete: List[str]) -> bool:
    try:
        # Читаем Excel файл
        df = pd.read_excel(file_path)

        # Проверяем, существуют ли колонки
        existing_columns = set(df.columns)
        # Преобразуем имена колонок из объектов в строки
        columns_to_delete_names = set(columns_to_delete)

        if not columns_to_delete_names.issubset(existing_columns):
            logger.error(f"Some columns don't exist: {columns_to_delete_names - existing_columns}")
            return False

        # Удаляем колонки
        df.drop(columns=list(columns_to_delete_names), inplace=True)

        # Сохраняем файл
        df.to_excel(file_path, index=False)

        return True
    except Exception as e:
        logger.error(f"Error in delete_columns: {str(e)}")
        return False

def get_columns(file_path: str):
    try:
        # Загружаем Excel файл в DataFrame
        df = pd.read_excel(file_path)

        # Используем ту же логику категоризации, что и в save_file
        def categorize_column(column_data):
            if pd.api.types.is_string_dtype(column_data):
                return "Nominal"
            elif pd.api.types.is_numeric_dtype(column_data):
                unique_values = column_data.nunique()
                if unique_values < 10:
                    return "Ordinal"
                elif unique_values >= 10:
                    return "Interval" if column_data.min() > 0 else "Ratio"
            return "Unknown"

        # Возвращаем тот же формат, что и save_file
        columns_info = [
            {"name": col, "category": categorize_column(df[col])}
            for col in df.columns
        ]

        return columns_info
    except Exception as e:
        logger.error(f"Error in get_columns: {str(e)}")
        return []
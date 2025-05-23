import os
from typing import List, Optional
from typing import Dict, Union
import json

from fastapi import APIRouter, HTTPException, Form, UploadFile, File
from pydantic import BaseModel
from starlette.responses import JSONResponse

from qc.api.gene.merge_symptoms import merge_symptoms, MergeSymptomRequest
from qc.api.symptoms_service.service import update_symptom_order, get_categories_with_nested_keys
from qc.logging_config import logger
from qc.api.files import get_excel_files_list
from qc.api.gene import gene_excel_file, delete_excel_file, update_symptom_categories
from qc.api.gene.list_genes import get_symptom_gene_mapping_files  # Оставляем только get_unique_genes
from qc.api.gene.update_excel_file import update_excel_file_content
from qc.api.gene.update_symptom_categories import update_symptom_categories
from qc.config import properties_directory, version_folder, excel_folder
import shutil
import pandas as pd
from collections import OrderedDict

router = APIRouter()

@router.get("/list_excel_files", response_model=List[str])
async def get_excel_files():
    try:
        # Получаем список файлов с помощью функции из отдельного модуля
        excel_files = get_excel_files_list()

        return JSONResponse(content=excel_files)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/list_diseases_genes", response_model=List[str])  # Переименованный эндпоинт
async def list_diseases_genes():
    try:
        items = list(get_symptom_gene_mapping_files())
        #remove empty strings and numbers
        items = [item for item in items if isinstance(item, str) and item]
        return JSONResponse(content=items)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/update_excel_file")
async def update_excel_file(fileId: str = Form(...), newFile: UploadFile = File(...)):
    if not fileId.endswith(".xlsx"):
        return JSONResponse(status_code=400, content={"error": "Invalid file extension. Only .xlsx files are allowed."})

    try:
        await update_excel_file_content(fileId, newFile)
        return JSONResponse(content={"message": "File updated successfully"})
    except FileNotFoundError:
        return JSONResponse(status_code=404, content={"error": "File not found"})
    except Exception as e:
        logger.error(f"❌ Error updating file {fileId}: {str(e)}")
        logger.exception("Detailed error:")
        return JSONResponse(status_code=500, content={"error": str(e)})

# Определим модель для данных о симптомах, которую клиент отправит на сервер
class SymptomOrder(BaseModel):
    geneName: str
    symptomName: str
    categoryName: str
    order: int

@router.post("/set_symptom_order")
async def set_symptoms_order(symptoms: List[SymptomOrder]):
    # Добавляем логирование входящих данных
    logger.info(f"Received set_symptom_order request with data: {symptoms}")
    try:
        updated = update_symptom_order(symptoms)
        # Логируем результат вызова update_symptom_order
        logger.info(f"update_symptom_order returned: {updated}")

        if not updated:
            raise HTTPException(status_code=400, detail="Failed to update symptoms order")
        return {"status": "success", "message": "Symptoms order updated successfully"}
    except Exception as e:
        logger.error(f"Error in set_symptoms_order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add_new_gene")
async def upload_gene_excel_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No selected file")

    try:
        # Сохраняем файл и получаем имена колонок
        file_path = os.path.join("excel", file.filename)
        column_names = gene_excel_file.save_file(file)

        # Обновляем категории симптомов для загруженного файла
        try:
            update_symptom_categories(file_path)
        except Exception as e:
            logger.error(f"Error updating symptom categories: {str(e)}")
            # Не прерываем выполнение, если обновление категорий не удалось
            # Просто логируем ошибку

        return JSONResponse(content={
            "message": "File uploaded successfully and symptoms categorized",
            "columns": column_names
        })
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        logger.exception("Detailed error:")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/delete_excel_file")
async def web_delete_excel_file(file_id : str):
    logger.debug(f"Received file_id: {file_id}")
    file_path = f"excel/{file_id}"
    if delete_excel_file.delete(file_path):
        return {"message": "File deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="File not found")

# Обновим модель для запроса
class DeleteColumnsRequest(BaseModel):
    file_name: str
    columns: List[str]

@router.post("/delete_columns")
async def delete_columns(request: DeleteColumnsRequest):
    try:
        # Проверяем существование файла
        file_path = f"excel/{request.file_name}"
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"File {request.file_name} not found"
            )

        # Удаляем колонки из файла
        success = gene_excel_file.delete_columns(file_path, request.columns)

        if success:
            # Получаем обновленный список колонок
            remaining_columns = gene_excel_file.get_columns(file_path)
            return JSONResponse(content={
                "message": "Columns deleted successfully",
                "remaining_columns": remaining_columns
            })
        else:
            raise HTTPException(status_code=400, detail="Failed to delete columns")

    except Exception as e:
        logger.error(f"Error deleting columns: {str(e)}")
        logger.exception("Detailed error:")
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/get_columns")
async def get_columns(file_name: str):
    try:
        file_path = f"excel/{file_name}"
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail=f"File {file_name} not found"
            )

        columns = gene_excel_file.get_columns(file_path)
        return JSONResponse(content={
            "columns": columns
        })
    except Exception as e:
        logger.error(f"Error getting columns: {str(e)}")
        logger.exception("Detailed error:")
        return JSONResponse(status_code=500, content={"error": str(e)})

# FastAPI endpoint
@router.post("/merge_symptoms")
async def merge_symptoms_endpoint(data: MergeSymptomRequest):
    success = merge_symptoms(data)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to merge symptoms")
    return {"status": "success"}


# Убедимся, что папка для версий существует
os.makedirs(version_folder, exist_ok=True)


def save_version(file_path, version_folder, prefix):
    prefix_folder = os.path.join(version_folder, prefix)
    os.makedirs(prefix_folder, exist_ok=True)

    base_name = os.path.basename(file_path)

    # Удаление последующих версий
    existing_versions = sorted(os.listdir(prefix_folder))
    current_version_number = len([v for v in existing_versions if v.split('.')[-1].isdigit()]) + 1
    for version in existing_versions:
        if version.split('.')[-1].isdigit() and int(version.split('.')[-1]) >= current_version_number:
            os.remove(os.path.join(prefix_folder, version))

    version_path = os.path.join(prefix_folder, f"{base_name}.{current_version_number}")
    shutil.copy(file_path, version_path)

    # Запись файла с номером текущей версии
    current_version_file = os.path.join(prefix_folder, "current_version")
    with open(current_version_file, 'w') as f:
        f.write(str(current_version_number))


def get_current_version(prefix_folder):
    current_version_file = os.path.join(prefix_folder, "current_version")
    versions = sorted([int(v.split('.')[-1]) for v in os.listdir(prefix_folder) if v.split('.')[-1].isdigit()])

    if os.path.exists(current_version_file):
        with open(current_version_file, 'r') as f:
            try:
                current_version = int(f.read().strip())
                if current_version in versions:
                    return current_version
            except ValueError:
                pass

    if versions:
        with open(current_version_file, 'w') as f:
            f.write(str(versions[-1]))
        return versions[-1]

    return None


def restore_version(file_path, version_folder, prefix, direction):
    prefix_folder = os.path.join(version_folder, prefix)
    base_name = os.path.basename(file_path)
    versions = sorted([int(v.split('.')[-1]) for v in os.listdir(prefix_folder) if v.split('.')[-1].isdigit()])
    current_version = get_current_version(prefix_folder)

    if direction == "next":
        next_version = min([v for v in versions if v > current_version], default=None)
    elif direction == "previous":
        next_version = max([v for v in versions if v < current_version], default=None)
    else:
        raise ValueError("Direction must be 'next' or 'previous'.")

    if next_version is not None:
        version_path = os.path.join(prefix_folder, f"{base_name}.{next_version}")
        if os.path.exists(version_path):
            shutil.copy(version_path, file_path)

            with open(os.path.join(prefix_folder, "current_version"), 'w') as f:
                f.write(str(next_version))
            return next_version

    return None

class RenameSymptomRequest(BaseModel):
    json_file: str
    old_name: str
    new_name: str

@router.post("/rename-symptom")
def rename_symptom_endpoint(request: RenameSymptomRequest):
    json_file = request.json_file
    old_name = request.old_name
    new_name = request.new_name
    try:
        json_file_path = os.path.join(properties_directory, json_file)
        json_file_name = os.path.basename(json_file_path)
        components = json_file_name.split("_")
        if len(components) >= 3:
            prefix = f"{components[-1].replace('.json', '')}-{components[2]}"
        else:
            raise ValueError("Invalid JSON file name format.")

        save_version(json_file_path, version_folder, prefix)

        with open(json_file_path, 'r') as file:
            data = json.load(file)

        for category, symptoms in data.items():
            if new_name in symptoms:
                raise ValueError(f"Symptom '{new_name}' already exists.")

        updated = False
        for category, symptoms in data.items():
            if isinstance(symptoms, dict):
                symptoms = OrderedDict(symptoms)

            if old_name in symptoms:
                # Создаем новый OrderedDict, чтобы порядок сохранялся
                new_symptoms = OrderedDict()
                for key, value in symptoms.items():
                    if key == old_name:
                        new_symptoms[new_name] = value
                        updated = True
                    else:
                        new_symptoms[key] = value

                data[category] = new_symptoms

        if updated:
            with open(json_file_path, 'w') as file:
                json.dump(data, file, indent=2)
        else:
            raise HTTPException(status_code=404, detail=f"Symptom '{old_name}' not found.")

        excel_file_name = find_excel_file_by_prefix(prefix, excel_folder)
        if not excel_file_name:
            raise HTTPException(status_code=404, detail="Excel file not found.")

        excel_file_path = os.path.join(excel_folder, excel_file_name)
        save_version(excel_file_path, version_folder, prefix)

        df = pd.read_excel(excel_file_path)
        if new_name in df.columns or new_name in df.values:
            raise ValueError(f"Symptom '{new_name}' already exists in Excel.")

        df.columns = [col if col.lower() != old_name.lower() else new_name for col in df.columns]
        for col in df.columns:
            if col.lower() in ["initial_sympt1", "initial_sympt2", "initial_sympt3"]:
                df[col].replace({old_name: new_name}, inplace=True, regex=True)

        df.to_excel(excel_file_path, index=False)
        return {"message": f"Symptom '{old_name}' renamed to '{new_name}' successfully."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/step")
def step_version(json_file: str, direction: str):
    try:
        json_file_path = os.path.join(properties_directory, json_file)
        json_file_name = os.path.basename(json_file_path)
        components = json_file_name.split("_")
        if len(components) >= 3:
            prefix = f"{components[-1].replace('.json', '')}-{components[1]}"
        else:
            raise ValueError("Invalid JSON file name format.")

        next_version = restore_version(json_file_path, version_folder, prefix, direction)
        if next_version is None:
            raise HTTPException(status_code=404, detail="No more versions available in this direction.")

        return {"message": f"Moved to {direction} version {next_version}."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list_symptoms")
def get_current_json(json_file: str):
    try:
        full_file_path = os.path.join(properties_directory, json_file)
        symptoms = get_categories_with_nested_keys(full_file_path)
        return JSONResponse(content=symptoms)
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        return JSONResponse(status_code=404, content={"error": "File not found"})
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format: {str(e)}")
        return JSONResponse(status_code=400, content={"error": "Invalid file format"})
    except Exception as e:
        logger.error(f"Error in web_list_symptoms: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# @router.get("/list_symptoms")
# async def web_list_symptoms(
#     file_name: str  # Теперь принимаем имя файла напрямую
# ):
#     try:
#         # Формируем полный путь к файлу, используя properties_directory
#         full_file_path = os.path.join(properties_directory, file_name)
#         symptoms = get_categories_with_nested_keys(full_file_path)
#         return JSONResponse(content=symptoms)
#     except FileNotFoundError as e:
#         logger.error(f"File not found: {str(e)}")
#         return JSONResponse(status_code=404, content={"error": "File not found"})
#     except json.JSONDecodeError as e:
#         logger.error(f"Invalid JSON format: {str(e)}")
#         return JSONResponse(status_code=400, content={"error": "Invalid file format"})
#     except Exception as e:
#         logger.error(f"Error in web_list_symptoms: {str(e)}")
#         return JSONResponse(status_code=500, content={"error": str(e)})



@router.get("/current-excel")
def get_current_excel(json_file: str):
    try:
        json_file_name = os.path.basename(json_file)
        components = json_file_name.split("_")
        if len(components) >= 3:
            prefix = f"{components[-1].replace('.json', '')}-{components[1]}"
        else:
            raise ValueError("Invalid JSON file name format.")

        excel_file_name = find_excel_file_by_prefix(prefix, excel_folder)
        if not excel_file_name:
            raise HTTPException(status_code=404, detail="Excel file not found.")

        excel_file_path = os.path.join(excel_folder, excel_file_name)
        df = pd.read_excel(excel_file_path)
        return df.to_dict()

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def find_excel_file_by_prefix(prefix, folder):
    prefix_lower = prefix.lower()
    for file_name in os.listdir(folder):
        if file_name.lower().startswith(prefix_lower) and file_name.endswith(".xlsx"):
            return file_name
    return None

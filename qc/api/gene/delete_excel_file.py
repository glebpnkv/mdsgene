import os

def delete(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False
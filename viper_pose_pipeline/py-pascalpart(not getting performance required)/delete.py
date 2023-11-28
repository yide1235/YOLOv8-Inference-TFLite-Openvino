import os

def delete_empty_txt_files(folder_path):
    # List all files in the specified folder
    files = os.listdir(folder_path)
    
    # Iterate through the files and delete if they are empty .txt files
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(folder_path, file)
            # Check if the file is empty
            if os.path.getsize(file_path) == 0:
                try:
                    os.remove(file_path)  # Delete the empty file
                    print(f"Deleted empty file: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

# Specify the folder path where you want to delete .txt files
folder_path = "./datasets/labels/train2017"

# Call the function to delete empty .txt files in the specified folder
delete_empty_txt_files(folder_path)

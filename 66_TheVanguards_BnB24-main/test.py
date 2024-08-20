import os

def rename_files(folder_path):
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return
    
    files = os.listdir(folder_path)
    files.sort()
    
    # Check if there are files in the directory
    if len(files) == 0:
        print("No files found in the directory.")
        return
    
    # Determine the maximum number of digits required for renaming
    max_digits = len(str(len(files)))
    
    for index, file in enumerate(files, 2383):
        # Create the new file name with leading zeros
        new_name = f"{index:0{max_digits}d}{os.path.splitext(file)[1]}"
        
        # Rename the file
        os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_name))
        
        print(f"Renamed {file} to {new_name}.")

# Specify the folder path here
folder_path = "data/TEACHER"
rename_files(folder_path)

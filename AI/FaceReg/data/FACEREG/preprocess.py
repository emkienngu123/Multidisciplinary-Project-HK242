import os

def rename_images_in_folders(base_path):
    # List all folders in the base path
    folders = ["KIEN", "KIET", "LONG", "MINH"]
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        
        if not os.path.exists(folder_path):
            print(f"Folder '{folder}' does not exist. Skipping...")
            continue
        
        # List all files in the folder
        files = sorted(os.listdir(folder_path))  # Sort to ensure consistent renaming order
        
        for i, file_name in enumerate(files):
            # Get the file extension
            _, ext = os.path.splitext(file_name)
            
            # Construct the new file name
            new_name = f"{i}{ext}"
            
            # Rename the file
            old_path = os.path.join(folder_path, file_name)
            new_path = os.path.join(folder_path, new_name)
            os.rename(old_path, new_path)
        
        print(f"Renamed files in folder '{folder}'.")

# Base path where the folders are located
base_path = r"c:\Users\dangv\Desktop\Multidisciplinary-Project-HK242\AI\FaceReg\data\FACEREG"

# Call the function
rename_images_in_folders(base_path)
import os
from modelscope.hub.api import HubApi
from modelscope.utils.constant import DEFAULT_MODEL_REVISION
import argparse
from tqdm import tqdm

def upload_folder_to_modelscope(local_path, model_id, access_token, bulk_upload=False):
    """
    Upload an entire folder to ModelScope dataset.
    
    Args:
        local_path (str): Local folder path to upload
        model_id (str): ModelScope model ID in format 'username/model_name'
        access_token (str): ModelScope access token
        bulk_upload (bool): Whether to upload the entire folder at once
    """
    # Initialize ModelScope API
    api = HubApi()
    api.login(access_token)
    
    # Check if repository exists, create if not
    try:
        api.get_model(model_id)
        print(f"Repository {model_id} exists.")
    except Exception as e:
        print(f"Creating repository {model_id}...")
        try:
            api.create_model(model_id)
        except Exception as create_err:
            print(f"Error creating repository: {str(create_err)}")
            return
    
    if bulk_upload:
        # Upload entire folder at once
        try:
            print(f"Uploading entire folder {local_path} to {model_id}...")
            api.upload_folder(
                repo_id=model_id,
                folder_path=local_path,
                path_in_repo="",  # Upload to root of the repository
                commit_message=f"Bulk upload from {os.path.basename(local_path)}"
            )
            print("Folder upload completed successfully!")
        except Exception as e:
            print(f"Failed to upload folder: {str(e)}")
    else:
        # Get all files in the folder recursively
        all_files = []
        for root, _, files in os.walk(local_path):
            for file in files:
                # Get full local path
                local_file = os.path.join(root, file)
                # Get relative path for ModelScope
                rel_path = os.path.relpath(local_file, local_path)
                all_files.append((local_file, rel_path))
        
        print(f"Found {len(all_files)} files to upload")
        
        # Upload each file
        for local_file, rel_path in tqdm(all_files, desc="Uploading files"):
            try:
                # Use upload_file method
                api.upload_file(
                    path_or_fileobj=local_file,
                    path_in_repo=rel_path,
                    repo_id=model_id,
                    commit_message=f"Upload {rel_path}"
                )
                print(f"Successfully uploaded: {rel_path}")
            except Exception as e:
                print(f"Failed to upload {rel_path}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload folder to ModelScope")
    parser.add_argument("--local_path", type=str, required=True,
                      help="Local folder path to upload")
    parser.add_argument("--model_id", type=str, required=True,
                      help="ModelScope model ID (xDAN2099/model_name)")
    parser.add_argument("--access_token", type=str, 
                      default="721ec736-ec61-45ad-a1ec-b3b339ef016d",
                      help="ModelScope access token")
    parser.add_argument("--bulk", action="store_true",
                      help="Upload entire folder at once instead of file by file")
    
    args = parser.parse_args()
    
    upload_folder_to_modelscope(
        local_path=args.local_path,
        model_id=args.model_id,
        access_token=args.access_token,
        bulk_upload=args.bulk
    )

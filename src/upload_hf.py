from huggingface_hub import HfApi, HfFolder, upload_folder
import os

# Replace with your Hugging Face username
username = 'XShadow'

# Repository details
repo_name = 'EarthNets-GAMUS'
repo_id = f'{username}/{repo_name}'

# Local dataset directory to upload
local_dir = 'gamus_dataset_val/'

# Check if the local directory exists
if not os.path.isdir(local_dir):
    raise FileNotFoundError(f"The directory '{local_dir}' does not exist.")

# Get the Hugging Face authentication token
token = HfFolder.get_token()
if token is None:
    raise ValueError("You are not logged in. Please run 'huggingface-cli login' to log in to your Hugging Face account.")

# Initialize the API
api = HfApi()

# Create the dataset repository if it doesn't exist
api.create_repo(
    repo_id=repo_id,
    repo_type='dataset',
    token=token,
    exist_ok=True  # Set to True to avoid errors if the repo already exists
)

# Upload the local directory to the repository
upload_folder(
    repo_id=repo_id,
    repo_type='dataset',
    folder_path=local_dir,
    path_in_repo='',  # Upload to the root of the repository
    token=token,
    commit_message='Upload gamus_dataset_train'
)

print(f"Dataset '{local_dir}' has been uploaded to '{repo_id}' on Hugging Face Hub.")

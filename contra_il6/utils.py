import zipfile
import os

def unzip_file(zip_file_path, extract_dir):
    """
    Unzips a .zip file to the specified directory.
    
    Parameters:
    zip_file_path (str): Path to the .zip file.
    extract_dir (str): Directory where the contents will be extracted.
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f'Extracted {zip_file_path} to {extract_dir}')
    
    # Remove the .zip file after extraction
    os.remove(zip_file_path)
    
def download_ckpts(url, output_dir):
    """
    Downloads a .zip file from the specified URL and extracts it to the output directory.
    
    Parameters:
    url (str): URL of the .zip file to download.
    output_dir (str): Directory where the .zip file will be saved and extracted.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    zip_file_path = os.path.join(output_dir, 'ckpts.zip')
    
    # Download the file
    import requests
    response = requests.get(url)
    with open(zip_file_path, 'wb') as f:
        f.write(response.content)
    
    print(f'Downloaded {url} to {zip_file_path}')
    
    # Unzip the file
    unzip_file(zip_file_path, output_dir)
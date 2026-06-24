import requests
import os

url = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"
filename = "3dshapes.h5"

if os.path.exists(filename):
    print(f"{filename} already exists. Skipping download.")
else:
    print(f"Downloading {filename} from {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Download failed: {e}")

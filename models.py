import requests

def download_model_from_url(url, output_path):
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Write the content of the response to a file
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully.")
    else:
        print("Failed to download model. Status code:", response.status_code)

download_model_from_url("https://storage.googleapis.com/sandeep_personal/efficientnet_b3_custom.pt", "leaf.pt")
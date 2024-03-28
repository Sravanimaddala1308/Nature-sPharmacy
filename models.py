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

# Example usage:
# models = ["https://storage.googleapis.com/sandeep_personal/Bone.pt","https://storage.googleapis.com/sandeep_personal/xray.pt","https://storage.googleapis.com/sandeep_personal/retina%20(1).pt","https://storage.googleapis.com/sandeep_personal/lung_model_torchpt.pt"]
# output_path = ['Bone.pt',"xray.pt","retina.pt","lung.pt" ]
download_model_from_url("https://storage.googleapis.com/sandeep_personal/efficientnet_b3_custom.pt", "leaf.pt")
# for i in range(len(models)):
#     download_model_from_url(models[i], output_path[i])
import requests

def get_model_information(model_id):
    """
    Retrieve model information from Civitai AI.

    Parameters:
    model_id (str): The ID of the model to retrieve information for.

    Returns:
    dict: A dictionary containing the model information, or None if not found.
    """
    base_url = "https://api.civitai.ai/v1/models/"
    url = f"{base_url}{model_id}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to retrieve model information. Status code: {response.status_code}")
        return None


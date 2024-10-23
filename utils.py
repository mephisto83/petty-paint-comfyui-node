import os

class PPApi:
    @classmethod
    def get_image_file_paths(cls, directory):
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.svg', '.webp', '.heic')
        return PPApi.get_file_paths(directory, image_extensions)

    @classmethod
    def get_xl_models(cls, directory):
        image_extensions = ('.safetensors')
        return PPApi.get_file_names(directory, image_extensions)

    @classmethod
    def get_file_names(cls, directory, image_extensions):
        # List to hold image file paths
        file_names = []

        # Walk through the directory
        for root, _, files in os.walk(directory):
            for file in files:
                # Check if the file ends with one of the image extensions
                if file.lower().endswith(image_extensions):
                    # Create the full file path and add it to the list
                    file_names.append(file)

        return file_names

    @classmethod
    def get_file_paths(cls, directory, image_extensions):
        # List to hold image file paths
        image_file_paths = []

        # Walk through the directory
        for root, _, files in os.walk(directory):
            for file in files:
                # Check if the file ends with one of the image extensions
                if file.lower().endswith(image_extensions):
                    # Create the full file path and add it to the list
                    image_file_paths.append(os.path.join(root, file))

        return image_file_paths


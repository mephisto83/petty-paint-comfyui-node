import http.server
import socketserver
import json
import threading
import hashlib
from threading import Lock
import os
import requests
import traceback
from PIL import Image, ImageOps
import logging
from urllib.parse import urlparse, unquote
import urllib.parse
import folder_paths
import comfy.sd
import comfy.utils
import torch
import numpy as np
import socketserver
import threading
import base64
import hashlib
import struct

# Create a global dictionary to hold locks for files
download_locks = {}
PROJECT_FILE_NAME = "model_dict.json"
SOURCE_IMAGE_FOLDER = "source_images"
RENDERED_IMAGE_FOLDER = "images"
def get_lock_for_filename(filename):
    """Retrieve a unique lock for each filename."""
    if filename not in download_locks:
        download_locks[filename] = Lock()
    return download_locks[filename]
def find_image_file_path(setup_json):
    try:
        setups = json.loads(setup_json)
        image_path = setups.get('source_images', {}).get("last_image_path", None)
        
        if not image_path:  # Return False early if 'models' is empty or not present
            return None
        return image_path
        
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except IndexError as e:
        print(f"Index error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None
def download_image(url, folder):
    try:
        # Make a request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Extract the filename from the URL
        filename = url.split('/')[-1].split('?')[0]  # Simple split method to get the filename before any parameters
        save_path = os.path.join(folder, filename)  # Construct the full path
        
        # Create the folder if it does not exist
        os.makedirs(folder, exist_ok=True)
        
        # Write the image to the specified folder
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        print(f"File downloaded successfully: {save_path}")
        return filename, save_path
    except requests.RequestException as e:
        print(f"Failed to download the file: {e}")
        return None, None
    except OSError as e:
        print(f"Failed to save the file: {e}")
        return None, None

def extract_filename_from_url(url):
    # Parse the URL into components
    parsed_url = urllib.parse.urlparse(url)
    # Decode the URL path to handle encoded characters
    decoded_path = urllib.parse.unquote(parsed_url.path)
    # Extract the filename part after the last '/'
    filename = decoded_path.split('/')[-1]
    return filename

def write_string_to_file(file_path, content):
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        print("Content written successfully.")
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")

def read_string_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return None
    except IOError as e:
        print(f"An error occurred while reading the file: {e}")
        return None

def download_file(url, save_dir):
    try:
        result = None
        # Send a HTTP request to the URL
        response = requests.get(url, allow_redirects=True)
        response.raise_for_status()  # Will raise an exception for HTTP errors

        # Try to extract the filename from the Content-Disposition header
        if "Content-Disposition" in response.headers:
            content_disposition = response.headers['Content-Disposition']
            filename = content_disposition.split('filename=')[-1].strip('\"')
            result = filename 
            print(f"download {filename}")
        else:
            print(f"no Content-Disposition, guess the filename or throw an error")
            # If no Content-Disposition, guess the filename or throw an error
            filename = os.path.basename(unquote(urlparse(url).path))
            if not filename:
                raise ValueError("Filename could not be auto-detected from the URL or headers. Please specify a filename.")
            result = filename 

        # Combine the directory with the filename
        save_path = os.path.join(save_dir, filename)

        # Acquire lock for the specific filename
        file_lock = get_lock_for_filename(filename)
        with file_lock:
            # Check if the file already exists to avoid re-downloading
            if os.path.exists(save_path):
                print(f"File already exists: {save_path}. Skipping download.")
                return result

            # Save the file content
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"File downloaded and saved as: {save_path}")
        return result
    except Exception as e:
        logging.warning(f"[ERROR] An error occurred while downloading")
        logging.warning(traceback.format_exc())

def fetch_model_file_name(setup_json, models_index):
    try:
        setups = json.loads(setup_json)
        models = setups.get('models', [])
        
        if not models:  # Return False early if 'models' is empty or not present
            return False
        
        setups_list = setups.get('setups', [])
        if not setups_list:  # Return False early if 'setups' is empty or not present
            return False
        
        # Using modulo to safely handle index out of range and avoid crashing
        setup_index = models_index % len(setups_list)
        specific_setup = setups_list[setup_index]
        
        # Direct access to nested dictionary elements if they exist
        model_url = specific_setup.get('model', {}).get('url', None)
        fileName = specific_setup.get('model', {}).get('fileName', None)
        if fileName:
            return fileName
        if model_url and model_url in models:
            return models[model_url]
        
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except IndexError as e:
        print(f"Index error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return None

def fetch_model_info(setup_json, models_index):
    try:
        setups = json.loads(setup_json)
        models = setups.get('models', [])
        
        if not models:  # Return False early if 'models' is empty or not present
            return False
        
        setups_list = setups.get('setups', [])
        if not setups_list:  # Return False early if 'setups' is empty or not present
            return False
        
        # Using modulo to safely handle index out of range and avoid crashing
        setup_index = models_index % len(setups_list)
        specific_setup = setups_list[setup_index]
        
        # Direct access to nested dictionary elements if they exist
        steps = specific_setup.get('steps', 10)
        denoise = specific_setup.get('denoise', .7)
        return steps, denoise
        
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except IndexError as e:
        print(f"Index error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return 10, .5
    
class PettyPaintConnector:
    class CustomHTTPHandler(http.server.SimpleHTTPRequestHandler):
        connector = None  # Reference to the PettyPaintConnector that owns the handler
        model_folder = None
        lora_folder = None
        root_folder = None
        civitai_api_key = None
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')  # Allows all domains
            self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()

        def do_OPTIONS(self):
            self.send_response(200)
            self.end_headers()

        def do_GET(self):
            if self.path.startswith('/images'):
                self.list_images()
            elif self.path.startswith('/serve_image'):
                self.serve_image()
            elif self.path.startswith('/delete_image'):
                self.delete_image()
            elif self.path.startswith('/latest'):
                self.latests()
            elif self.path.startswith('/state'):
                self.getstate()
            else:
                self.send_error(404, "File not found")

        def handshake_websocket(self):
            key = self.headers['Sec-WebSocket-Key']
            accept_key = base64.b64encode(hashlib.sha1((key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').encode()).digest()).decode()
            self.send_response(101, "Switching Protocols")
            self.send_header('Upgrade', 'websocket')
            self.send_header('Connection', 'Upgrade')
            self.send_header('Sec-WebSocket-Accept', accept_key)
            self.end_headers()
            self.handle_websocket()

        def handle_websocket(self):
            """Handle the incoming WebSocket data."""
            try:
                while True:
                    data = self.read_message()
                    print("Received message:", data)  # Debug print
                    self.send_message(data)  # Echo back the received message
            except Exception as e:
                print("WebSocket connection closed", e)

        def read_message(self):
            """Read a single WebSocket frame and return the data."""
            byte1, byte2 = self.rfile.read(2)
            opcode = byte1 & 0x0F
            is_fin = byte1 & 0x80
            masked = byte2 & 0x80
            payload_len = byte2 & 0x7F

            if masked != 128:  # Clients must send masked data
                raise ValueError("Client must mask data")

            if payload_len == 126:
                payload_len = struct.unpack(">H", self.rfile.read(2))[0]
            elif payload_len == 127:
                payload_len = struct.unpack(">Q", self.rfile.read(8))[0]

            # Read masking key
            masking_key = self.rfile.read(4)

            # Read the payload and unmask it
            data = bytearray(self.rfile.read(payload_len))
            for i in range(len(data)):
                data[i] ^= masking_key[i % 4]

            return data.decode('utf-8')

        def send_message(self, message):
            """Send a message to the client over WebSocket."""
            response = bytearray()
            response.append(129)  # Text frame and FIN bit set
            length = len(message)
            if length <= 125:
                response.append(length)
            elif length <= 65535:
                response.append(126)
                response.extend(struct.pack(">H", length))
            else:
                response.append(127)
                response.extend(struct.pack(">Q", length))
            
            response.extend(message.encode('utf-8'))
            self.wfile.write(response)
        def latests(self):
            folder_path = os.path.join(self.connector.root_folder, RENDERED_IMAGE_FOLDER)
            try:
                # Assuming folder_path is defined and points to the directory containing the images
                filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                # Sort files by modification time, newest first
                filenames.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)

                # Get the newest file
                newest_image = filenames[0] if filenames else None
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(newest_image).encode('utf-8'))
            except Exception as e:
                self.send_error(500, str(e))

        def getstate(self):
            try:
                project_file = os.path.join(self.connector.root_folder, PROJECT_FILE_NAME)
                # Assuming folder_path is defined and points to the directory containing the images
                fileContents = read_string_from_file(project_file)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({ "data": "fileContents" }).encode('utf-8'))
            except Exception as e:
                self.send_error(500, str(e))
            
        def list_images(self):
            folder_path = os.path.join(self.connector.root_folder, RENDERED_IMAGE_FOLDER)
            try:
                # Assuming folder_path is defined and points to the directory containing the images
                filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
                # Sort files by modification time, newest first
                filenames.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(filenames).encode('utf-8'))
            except Exception as e:
                self.send_error(500, str(e))

        def delete_image(self):
            # Extract filename from URL
            filename = self.path.split('/')[-1]
            image_path = os.path.join(self.connector.root_folder, RENDERED_IMAGE_FOLDER, filename)
            if os.path.isfile(image_path):
                os.remove(image_path)
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')  # Assuming JPEG, adjust if different
                self.end_headers()
                self.wfile.write(json.dumps({}).encode('utf-8'))
            else:
                self.send_error(404, 'Image not found')   

        def serve_image(self):
            # Extract filename from URL
            filename = self.path.split('/')[-1]
            image_path = os.path.join(self.connector.root_folder, RENDERED_IMAGE_FOLDER, filename)
            if os.path.isfile(image_path):
                self.send_response(200)
                self.send_header('Content-type', 'image/jpeg')  # Assuming JPEG, adjust if different
                self.end_headers()
                with open(image_path, 'rb') as file:
                    self.wfile.write(file.read())
            else:
                self.send_error(404, 'Image not found')               

        def do_POST(self):
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            if self.path == '/draw_images':
                self.handle_draw_images(post_data)
            elif self.path == '/render_targets':
                self.handle_render_targets(post_data)
            else:
                response = {
                    "status": "success",
                    "message": "Data received at /draw_images"
                }
                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')  # Allows all domains
                self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
        def handle_draw_images(self, data):
            data_dict = json.loads(data.decode('utf-8'))
            response = {
                "status": "success",
                "message": "Data received at /draw_images"
            }
            # Optionally process the data with connector methods
            if self.connector:
                self.connector.process_draw_images(data_dict)

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))


        def handle_render_targets(self, data):
            data_dict = json.loads(data.decode('utf-8'))
            response = {
                "status": "success",
                "message": "Data received at /render_targets",
                "data": data_dict
            }
            # Optionally process the data with connector methods
            if self.connector:
                self.connector.process_render_targets(data_dict)

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode('utf-8'))

    def __init__(self, port=5005):
        try:
            self.port = int(port)
        except ValueError:
            raise ValueError("Port must be an integer value within the range 1-65535")

        self.server = socketserver.TCPServer(("", self.port), self.CustomHTTPHandler)
        self.CustomHTTPHandler.connector = self  # Pass instance to handler
    
    def setPort(self, port):
        if self.server:
            self.stop()
        self.port = int(port)
        self.server = socketserver.TCPServer(("", self.port), self.CustomHTTPHandler)
        self.CustomHTTPHandler.connector = self  # Pass instance to handler

    def start(self):
        print(f"Serving at port {self.port}")
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()
        print("Server stopped.")

    def process_draw_images(self, data):
        # Process data received from /draw_images
        print("Processing data at /draw_images:", data)
        res = self.load_project_file()
        if data["eventType"] == "child_changed" and data["key"] == "url":
            res["source_images"]["last_image"] = data["data"]
            filename, path = download_image(res["source_images"]["last_image"], os.path.join(self.root_folder, SOURCE_IMAGE_FOLDER))
            res["source_images"]["last_image_path"] = path
            res["source_images"]["last_image_filename"] = filename
            write_string_to_file(os.path.join(self.root_folder, PROJECT_FILE_NAME), json.dumps(res))

    def load_project_file(self):
        print(os.path.join(self.root_folder, PROJECT_FILE_NAME))
        res = read_string_from_file(os.path.join(self.root_folder, PROJECT_FILE_NAME))
        if res != None:
            res = json.loads(res)
        else:
            res = {}
            res["models"] = {}
            res["loras"] = {}
            res["setups"] = {}
            write_string_to_file(os.path.join(self.root_folder, PROJECT_FILE_NAME), json.dumps(res))
        
        if not "source_images" in res:
            res["source_images"] = {}
        
        return res
        
    def process_render_targets(self, data):
        # Process data received from /render_targets
        print("Processing data at /render_targets:", data)
        if data["eventType"] == "child_changed" and data["key"] == "setups":
            print("downloading resources")
            res = self.load_project_file()
            setup_datas = data["data"]
            setups = json.loads(setup_datas)
            res["setups"] = setups
            write_string_to_file(os.path.join(self.root_folder, PROJECT_FILE_NAME), json.dumps(res))
            for data_dict in setups:
                if does_directory_exist(self.lora_folder):
                    for loraObj in data_dict["loras"]:
                        print("downloading lora")
                        skip = False
                        if "url" in loraObj:
                            url = loraObj["url"]
                            if url in res["loras"]:
                                if res["loras"][url] and os.path.exists(os.path.join(self.lora_folder, res["loras"][url])):
                                    skip = True
                        if not skip:
                            if 'downloadUrl' in loraObj and loraObj['downloadUrl']: 
                                filename = download_file(f"{loraObj['downloadUrl']}?token={self.civitai_api_key}", self.lora_folder)
                                res["loras"][url] = filename
                                write_string_to_file(os.path.join(self.root_folder, PROJECT_FILE_NAME), json.dumps(res))
                                print("downloaded lora")
                        else:
                            print("already downloaded")
                if does_directory_exist(self.model_folder):
                    if "model" in data_dict:
                        print("downloading model")
                        model = data_dict['model']
                        skip = False
                        if "url" in model:
                            url = model["url"]
                            if url in res["models"]:
                                if res["models"][url] and os.path.exists(os.path.join(self.model_folder, res["models"][url])):
                                    skip = True
                        if not skip:
                            if "downloadUrl" in model and model['downloadUrl']:
                                filename = download_file(f"{model['downloadUrl']}?token={self.civitai_api_key}", self.model_folder)
                                res["models"][url] = filename
                                write_string_to_file(os.path.join(self.root_folder, PROJECT_FILE_NAME), json.dumps(res))
                            else:
                                print("missing download url")

            print("write model_dict.json")
            write_string_to_file(os.path.join(self.root_folder, PROJECT_FILE_NAME), json.dumps(res))

    def set_civitai_api_key(self, api_key):
        self.civitai_api_key = api_key

    def set_model_folder(self, folder):
        self.model_folder = folder

    def set_lora_folder(self, folder):
        self.lora_folder = folder

    def set_root_folder(self, folder):
        self.root_folder = folder


def does_directory_exist(dirpath):
    return os.path.exists(dirpath)

def read_string_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        return None
    except IOError as e:
        print(f"An error occurred while reading the file: {e}")
        return None

class PettyPaintComponent:
    ConnectorServer = None  # Static property declaration
    def __init__(self) -> None:
        self.connector = None
        self.setups = ""

    @classmethod
    def VALIDATE_INPUTS(cls, civitai_api_key, port, root_folder, model_folder, lora_folder, settings_file, **_):
        try:
            index = 0
            if not does_directory_exist(os.path.join(root_folder)):
                return "Expecting root folder at " + root_folder + " didnt exist"
            if not does_directory_exist(os.path.join(model_folder)):
                return "Expecting model folder at " + model_folder + " didnt exist"
            if not does_directory_exist(os.path.join(lora_folder)):
                return "Expecting lora folder at " + lora_folder + " didnt exist"
            port_num = int(port)
            if cls.ConnectorServer is not None and cls.ConnectorServer.port != port_num:
                cls.ConnectorServer.stop()
                cls.ConnectorServer = None
            if cls.ConnectorServer is None:
                cls.ConnectorServer = PettyPaintConnector(port_num)
                cls.ConnectorServer.start()
            cls.ConnectorServer.set_civitai_api_key(civitai_api_key)
            cls.ConnectorServer.set_model_folder(model_folder)
            cls.ConnectorServer.set_lora_folder(lora_folder)
            cls.ConnectorServer.set_root_folder(root_folder)
            if os.path.exists(os.path.join(root_folder, PROJECT_FILE_NAME)):
                setups = read_string_from_file(os.path.join(root_folder, PROJECT_FILE_NAME))
                model_file = fetch_model_file_name(setups, index)
                print(f"model_file: {model_file}")
                if not model_file:
                    return "model file not available"

            if 1 <= port_num <= 65535:
                return True
            else:
                return "Port number is out of the allowable range (1-65535)."
        except ValueError:
            return "Provided port is not a valid integer."
    
    @classmethod
    def IS_CHANGED(s, civitai_api_key, port, root_folder, model_folder, lora_folder, settings_file):
        project_file_path = os.path.join(root_folder, PROJECT_FILE_NAME)
        image_path = ""
        setups = ""
        hex = None
        if os.path.exists(project_file_path):
            setups = read_string_from_file((project_file_path))
            image_path = find_image_file_path(image_path)
            m = hashlib.sha256()
            with open(image_path, 'rb') as f:
                m.update(f.read())
            hex = m.digest().hex()
        res = read_string_from_file(os.path.join(root_folder, PROJECT_FILE_NAME))
        if res == None:
            res = {}
        last_image = res.get('source_images', {}).get('last_image', None)
        last_image_path = res.get('source_images', {}).get("last_image_path", None)
        last_image_filename = res.get('source_images', {}).get("last_image_filename", None)
        temp = f"{last_image}-{last_image_filename}-{last_image_path}-{civitai_api_key}-{port}-{root_folder}-{model_folder}-{lora_folder}-{hex}-{setups}-{settings_file}"
        print(temp)
        return temp
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
                "civitai_api_key": ("STRING", {"default": '', "multiline": False}),
                "port": ("STRING", {"default": '5005', "multiline": False}),
                "root_folder": ("STRING", {"default": '', "multiline": False}),
                "model_folder": ("STRING", {"default": '', "multiline": False}),
                "lora_folder": ("STRING", {"default": '', "multiline": False}),
                "settings_file": ("STRING", {"default": '', "forceInput": True, "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "MODEL", "CLIP", "VAE", "IMAGE", "LORA_STACK", "STRING", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("civitai_api_key", "localUrl", "root", "model", "lora", "setups", "MODEL", "CLIP", "VAE", "IMAGE", "LORA_STACK", "Prompt", "Negative Prompt", "steps", "denoise")
    FUNCTION = "doStuff"
    CATEGORY = "APorter"




    def fetch_lora_stack_file_name(self, setup_json, models_index):
        lora_list = list()
        try:
            setups = json.loads(setup_json)
            loras = setups.get('loras', [])
            
            if not loras:  # Return False early if 'models' is empty or not present
                return list()
            
            setups_list = setups.get('setups', [])
            if not setups_list:  # Return False early if 'setups' is empty or not present
                return list()
            
            # Using modulo to safely handle index out of range and avoid crashing
            setup_index = models_index % len(setups_list)
            specific_setup = setups_list[setup_index]
            
            # Direct access to nested dictionary elements if they exist
            loras = specific_setup.get('loras', [])
            for lora in loras:
                lora_url = lora.get('url', None)
                strength = lora.get('strength', 0)
                clip = lora.get('clip', 0)
                if lora_url and lora_url in loras:
                    lora_name = loras[lora_url]
                    lora_list.extend([(lora_name, strength, clip)])
            
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
        except IndexError as e:
            print(f"Index error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        return lora_list

    def fetch_image_file_path(self, setup_json):
        try:
            setups = json.loads(setup_json)
            image_path = setups.get('source_images', {}).get("last_image_path", None)
            
            if not image_path:  # Return False early if 'models' is empty or not present
                return None
            return image_path
            
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
        except IndexError as e:
            print(f"Index error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        return None
    def fetch_prompts(self, setup_json, index):
        try:
            setupjson = json.loads(setup_json)
            setups = setupjson.get('setups', {})
            prompts = setups[index].get("prompts", [])
            pos_prompt = ""
            neg_prompt = ""
            for prompt in prompts:
                if "prompt" in prompt and "positive" in prompt and prompt["positive"]:
                    pos_prompt += prompt["prompt"] + " "
                elif "prompt" in prompt and  "positive" in prompt and prompt["positive"] == False:
                    neg_prompt += prompt["prompt"] + " "
            
            return pos_prompt, neg_prompt
            
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
        except IndexError as e:
            print(f"Index error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        return "", ""
    

    def load_checkpoint(self, setups, index):
        print(setups)
        if setups:
            model_file = fetch_model_file_name(setups, index)
            steps, denoise = fetch_model_info(setups, index)
            print(f"model_file: {model_file}")
            if not model_file:
                return (None, None, None, image, lora_stack, "", "") #out[:3]
            lora_stack = self.fetch_lora_stack_file_name(setups, index)
            print(f"lora_stack: {lora_stack}")
            image_file = self.fetch_image_file_path(setups)
            print(f"image_file: {image_file}")
            pos_prompt, neg_prompt = self.fetch_prompts(setups, index)
            i = Image.open(image_file)
            i = ImageOps.exif_transpose(i)
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            ckpt_path = folder_paths.get_full_path("checkpoints", model_file)
            print("ckpt_path")
            if ckpt_path == None:
                ckpt_path = os.path.join(PettyPaintComponent.ConnectorServer.model_folder, model_file)
            print(ckpt_path)
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )
            return (out[0], out[1], out[2], image, lora_stack, pos_prompt, neg_prompt, steps, denoise) #out[:3]
        else:
            image = None
            model_file = None
            lora_stack = list()
            return (None, None, None, image, lora_stack, "", "") #out[:3]

    def doStuff(self, civitai_api_key, port, root_folder, model_folder, lora_folder, settings_file, output_vae=True, output_clip=True, **kwargs):
        if PettyPaintComponent.ConnectorServer != None and PettyPaintComponent.ConnectorServer.port != port:
            PettyPaintComponent.ConnectorServer.setPort(port)
            PettyPaintComponent.ConnectorServer.start()
        PettyPaintComponent.ConnectorServer.set_civitai_api_key(civitai_api_key)
        PettyPaintComponent.ConnectorServer.set_model_folder(model_folder)
        PettyPaintComponent.ConnectorServer.set_lora_folder(lora_folder)
        PettyPaintComponent.ConnectorServer.set_root_folder(root_folder)
        print("started petty paint connector")
        setups = ""
        if os.path.exists(os.path.join(root_folder, PROJECT_FILE_NAME)):
            setups = read_string_from_file(os.path.join(root_folder, PROJECT_FILE_NAME))
        others = self.load_checkpoint(setups, 0)
        # Compile output data into a tuple for easy access
        output_data = (civitai_api_key, f"http://localhost:{port}", root_folder, model_folder, lora_folder, setups) + tuple(others)
        
        return output_data

class PettyPaintSDTurboScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "steps": ("INT", {"default": 1, "min": 1, "max": 10, "forceInput": True }),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.01, "forceInput": True }),
                      }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "APorter"
    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, steps, denoise):
        start_step = 10 - int(10 * denoise)
        timesteps = torch.flip(torch.arange(1, 11) * 100 - 1, (0,))[start_step:start_step + steps]
        comfy.model_management.load_models_gpu([model])
        sigmas = model.model.model_sampling.sigma(timesteps)
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        return (sigmas, )
    
def validate_setups_inputs(setup_json, models_index):
    """
    Validates if the provided JSON string contains the required 'models' and a specific model setup.

    Args:
    setup_json (str): JSON string containing setup and model details.
    models_index (int): Index to identify the specific setup in the setups list.

    Returns:
    bool: True if the required URL exists in 'models', otherwise False.
    """
    try:
        print(setup_json)
        setups = json.loads(setup_json)
        models = setups.get('models', [])
        
        if not models:  # Return False early if 'models' is empty or not present
            print("no models in setups")
            return False
        
        setups_list = setups.get('setups', [])
        if not setups_list:  # Return False early if 'setups' is empty or not present
            print("no setups in setups")
            return False
        
        # Using modulo to safely handle index out of range and avoid crashing
        setup_index = models_index % len(setups_list)
        specific_setup = setups_list[setup_index]
        
        # Direct access to nested dictionary elements if they exist
        model_url = specific_setup.get('model', {}).get('url', None)
        if model_url and model_url in models:
            return True
        print("no model_url in models")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except IndexError as e:
        print(f"Index error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return False

def validate_image_file_path(setup_json):
    try:
        setups = json.loads(setup_json)
        image_path = setups.get('source_images', {}).get("last_image_path", None)
        
        if not image_path:  # Return False early if 'models' is empty or not present
            return False
        if not os.path.exists(image_path):
            return "Expecting layer.png at " + image_path
        return True
        
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except IndexError as e:
        print(f"Index error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    return False

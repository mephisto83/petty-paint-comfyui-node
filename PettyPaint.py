import subprocess
import pkg_resources

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = {
    "Flask": "2.1.2"
}

# Check for missing packages and install them
for package, version in required_packages.items():
    try:
        pkg_resources.get_distribution(f"{package}=={version}")
    except pkg_resources.DistributionNotFound:
        install_package(f"{package}=={version}")


from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/draw_images', methods=['POST'])
def draw_images():
    # Get the JSON data sent with the request
    data = request.get_json()
    # Here you can process the data and do whatever is needed
    print("Received at /draw_images:", data)
    # Respond back with a success message
    return jsonify({"status": "success", "message": "Data received at /draw_images"}), 200

@app.route('/render_targets', methods=['POST'])
def render_targets():
    # Get the JSON data sent with the request
    data = request.get_json()
    # Here you can process the data and do whatever is needed
    print("Received at /render_targets:", data)
    # Respond back with a success message
    return jsonify({"status": "success", "message": "Data received at /render_targets"}), 200

if __name__ == '__main__':
    # Run the Flask app on all available interfaces on port 5000
    app.run(host='0.0.0.0', port=5000)


class PettyPaintConnector:
    def start():
        pass

import requests
from pathlib import Path
import json

API_URL = 'http://localhost:5000'

print("Testing YOLO Inference API")
print("=" * 50)

response = requests.get(f'{API_URL}/health')
print(f"Health: {response.json()}")

response = requests.get(f'{API_URL}/model-info')
model_info = response.json()
print(f"Model Info: {json.dumps(model_info, indent=2)}")

test_image_path = Path(__file__).parent.parent / 'data' / 'images' / 'test' / 'test_0000.jpg'

if test_image_path.exists():
    print(f"\nTesting inference with: {test_image_path.name}")
    
    with open(test_image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(f'{API_URL}/predict', files=files)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Detections: {result['num_detections']}")
        print(f"Classes found: {[d['class'] for d in result['detections']]}")
    else:
        print(f"Error: {response.json()}")
else:
    print(f"Test image not found: {test_image_path}")

print("=" * 50)
print("API Test Completed")

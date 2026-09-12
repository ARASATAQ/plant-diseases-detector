"""
End-to-end test: creates a synthetic diseased leaf image and hits /api/predict
"""
import cv2
import numpy as np
import urllib.request
import json
import os

# -- Create synthetic leaf image -----------------------------------------
img = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.ellipse(img, (200, 200), (180, 160), 0, 0, 360, (34, 139, 34), -1)  # green leaf
cv2.ellipse(img, (260, 180), (60, 50),   0, 0, 360, (30,  80, 120), -1)  # brown spot 1
cv2.ellipse(img, (150, 230), (40, 35),   0, 0, 360, (20,  60, 100), -1)  # brown spot 2
cv2.imwrite('test_leaf.jpg', img)
print('Test image created: test_leaf.jpg')

# -- Upload via multipart/form-data --------------------------------------
boundary = '----PythonTestBoundary12345'
with open('test_leaf.jpg', 'rb') as f:
    img_bytes = f.read()

body = (
    f'--{boundary}\r\n'.encode() +
    b'Content-Disposition: form-data; name="file"; filename="test_leaf.jpg"\r\n' +
    b'Content-Type: image/jpeg\r\n\r\n' +
    img_bytes +
    f'\r\n--{boundary}--\r\n'.encode()
)

req = urllib.request.Request(
    'http://localhost:8000/api/predict',
    data=body,
    method='POST',
)
req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')

try:
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read().decode())
    print('\n=== PREDICT RESULT ===')
    print(f"severity_grade : {data['severity_grade']}")
    print(f"severity_score : {data['severity_score']}")
    print(f"POI            : {data['poi']} %")
    print(f"ROI            : {data['roi']} px")
    print(f"disease_class  : {data['disease_class']}")
    print(f"confidence     : {data['confidence']} %")
    print(f"fine_tuned     : {data['using_fine_tuned_model']}")
    print(f"recommendation : {data['recommendation'][:90]}...")
    print(f"overlay_image  : [base64 PNG, {len(data['overlay_image'])} chars]")
    print('\nTEST PASSED')
except Exception as e:
    print(f'TEST FAILED: {e}')
finally:
    if os.path.exists('test_leaf.jpg'):
        os.remove('test_leaf.jpg')

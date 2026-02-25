import requests

with open("dashboard/db/uploads/IMG_9229.jpeg", "rb") as f:
    files = {"file": ("test.jpg", f, "image/jpeg")}
    data = {"platform_id": "test", "caption": "Hi", "uploader_id": "user"}
    r = requests.post("http://localhost:8000/v2/analyze", files=files, data=data)
    print(r.json())

import requests

with open("dashboard/db/uploads/077244d3-75bb-4598-b800-fd0bf6cdfab1.png", "rb") as f:
    files = {"file": ("test.png", f, "image/png")}
    data = {"platform_id": "test", "caption": "Gandhi Walking", "uploader_id": "user"}
    r = requests.post("http://localhost:8000/v2/analyze", files=files, data=data)
    print(r.json())

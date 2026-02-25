import requests

with open("dummy.jpg", "rb") as f:
    files = {"file": ("test.jpg", f, "image/jpeg")}
    data = {"platform_id": "test", "caption": "Hi", "uploader_id": "user"}
    r = requests.post("http://localhost:8000/v2/analyze", files=files, data=data)
    if r.status_code != 200:
        print("Status code:", r.status_code)
        print("Response text:", r.text)
    else:
        print(r.json())

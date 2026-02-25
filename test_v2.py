import requests
import json

def test():
    url = "http://localhost:8000/v2/analyze"
    
    # Test 1: Fraudulent Account
    with open("dummy4.jpg", "rb") as f:
        files = {"file": f}
        data = {
            "caption": "Click link below to double your crypto!",
            "uploader_id": "user_new_account",
            "platform_id": "test_script"
        }
        res = requests.post(url, files=files, data=data)
    # Test 2: Foreign Language Fraud (Hindi)
    with open("dummy4.jpg", "rb") as f:
        files = {"file": f}
        data = {
            "caption": "पैसे कमाएं और इस लिंक पर क्लिक करें",
            "uploader_id": "user_new_account",
            "platform_id": "test_script"
        }
        res2 = requests.post(url, files=files, data=data)
        print("\n=== FOREIGN LANGUAGE FRAUD TEST (HINDI) ===")
        print(json.dumps(res2.json(), indent=2))

    # Test 3: Satire Account
    with open("dummy4.jpg", "rb") as f:
        files = {"file": f}
        data = {
            "caption": "This is a parody generated with Midjourney",
            "uploader_id": "user_satire_bot",
            "platform_id": "test_script"
        }
        res2 = requests.post(url, files=files, data=data)
        print("\n=== SATIRE TEST ===")
        print(json.dumps(res2.json(), indent=2))

if __name__ == "__main__":
    test()

import requests


if __name__ == "__main__":
    files = {"file": open("file.txt", "rb")}
    r = requests.post("http://localhost:8000/upload_file", files=files)
    print(r.content)
    print(r.status_code)

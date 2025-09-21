# test_api.py
import requests
BASE = "http://127.0.0.1:5000"

def main():
    # 1. register (if exists it will return 400, that's fine)
    r = requests.post(BASE + "/auth/register", json={"username":"demo_user","password":"pass123"})
    print("register:", r.status_code, r.json())

    # 2. login
    r = requests.post(BASE + "/auth/login", json={"username":"demo_user","password":"pass123"})
    print("login:", r.status_code, r.json())
    if r.status_code != 200:
        print("Login failed — cannot continue.")
        return
    token = r.json().get("access_token")

    # 3. upload resume file — use exact filename in your folder
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": open("shaik_Mohammed_Nubaid_SDE.pdf", "rb")}# change exact name if differs
    r2 = requests.post(BASE + "/resume/upload", headers=headers, files=files)
    print("upload:", r2.status_code, r2.json())
    if not (r2.status_code == 200 or r2.status_code == 201):
        print("Upload failed — cannot continue.")
        return
    resume_id = r2.json().get("resume_id")

    # 4. match
    data = {
        "resume_id": resume_id,
        "job_description": "Looking for backend Python developer with Flask, REST API, SQL, Docker. Minimum 2 years experience."
    }
    r3 = requests.post(BASE + "/match", headers=headers, data=data)
    print("match:", r3.status_code, r3.json())

if __name__ == "__main__":
    main()

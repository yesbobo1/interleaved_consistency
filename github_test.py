import requests, base64, os

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OWNER = "yesbobo1"
REPO = "any2any-entertainment"
BRANCH = "main"

print(GITHUB_TOKEN)

url = "https://api.github.com/user"
r = requests.get(url, headers={"Authorization": f"token {GITHUB_TOKEN}"})
print(r.status_code, r.json())

file_bytes = b"hello github"  # 测试文件
path_in_repo = "pointclouds/test_upload.txt"
content_b64 = base64.b64encode(file_bytes).decode("utf-8")
url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{path_in_repo}"

# 检查文件是否存在
r = requests.get(url, headers={"Authorization": f"token {GITHUB_TOKEN}"})
if r.status_code == 200:
    sha = r.json()["sha"]
    data = {"message": "test upload", "content": content_b64, "sha": sha, "branch": BRANCH}
else:
    data = {"message": "test upload", "content": content_b64, "branch": BRANCH}

resp = requests.put(url, headers={"Authorization": f"token {GITHUB_TOKEN}",
                                  "Accept": "application/vnd.github+json"}, json=data)
print(resp.status_code, resp.text)

import os
import urllib.request

DATA_DIR = "../data"
os.makedirs(DATA_DIR, exist_ok=True)

import zipfile

def download(url, filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, path)
        print(f"Saved to {path}")
        # zip 파일이면 압축 해제 및 파일 목록 출력
        if filename.lower().endswith('.zip'):
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
                print(f"Extracted {filename} to {DATA_DIR}")
                print("Contained files:")
                for info in zip_ref.infolist():
                    print(" -", info.filename)
    else:
        print(f"{filename} already exists. Skipping download.")

# 1. UCI Adult
download(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    "adult.data"
)
download(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
    "adult.test"
)

# 2. Communities & Crime
download(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data",
    "communities.data"
)


# # 3. Law School Admission (LSAC)
# # UCI에는 없고, folktables에서 제공하는 링크 사용
# download(
#     "https://raw.githubusercontent.com/folktables/fairness-datasets/main/data/lsac.csv",
#     "lsac.csv"
# )

# # 4. Student Performance
# download(
#     "https://archive.ics.uci.edu/static/public/320/student+performance.zip",
#     "student-performance.zip"
# )


print("All datasets downloaded.")
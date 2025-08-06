import requests
import os

def download_file(url, save):
    query_parameters = {"downloadformat": "csv"}
    response = requests.get(url, params=query_parameters)
    if os.path.exists(save):
        print( "already exists. Skipping download.")
    else:
        with open(save, 'wb') as f:
            f.write(response.content)

if __name__ == "__main__":
    download_file("https://raw.githubusercontent.com/emotion-analysis-project/SemEval2025-Task11/refs/heads/main/task-dataset/semeval-2025-task11-dataset/track_a/test/eng.csv", "../data/test1.csv")
    download_file("https://raw.githubusercontent.com/emotion-analysis-project/SemEval2025-Task11/refs/heads/main/task-dataset/semeval-2025-task11-dataset/track_b/test/eng.csv", "../data/test2.csv")
    download_file("https://raw.githubusercontent.com/emotion-analysis-project/SemEval2025-Task11/refs/heads/main/task-dataset/semeval-2025-task11-dataset/track_a/train/eng.csv", "../data/train1.csv")
    download_file("https://raw.githubusercontent.com/emotion-analysis-project/SemEval2025-Task11/refs/heads/main/task-dataset/semeval-2025-task11-dataset/track_b/train/eng.csv", "../data/train2.csv")
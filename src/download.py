import requests
import os

# Function to download data
def download_file(url, save):
    response = requests.get(url)
    if os.path.exists(save):
        print( "already exists. Skipping download.")
    else:
        # Make data folder
        os.makedirs(os.path.dirname(save), exist_ok=True)
        # Save the data in the specified folder
        with open(save, 'wb') as f:
            f.write(response.content)

if __name__ == "__main__":
    download_file("https://raw.githubusercontent.com/emotion-analysis-project/SemEval2025-Task11/refs/heads/main/task-dataset/semeval-2025-task11-dataset/track_a/test/eng.csv", "../data/test1.csv")
    download_file("https://raw.githubusercontent.com/emotion-analysis-project/SemEval2025-Task11/refs/heads/main/task-dataset/semeval-2025-task11-dataset/track_b/test/eng.csv", "../data/test2.csv")
    download_file("https://raw.githubusercontent.com/emotion-analysis-project/SemEval2025-Task11/refs/heads/main/task-dataset/semeval-2025-task11-dataset/track_a/train/eng.csv", "../data/train1.csv")
    download_file("https://raw.githubusercontent.com/emotion-analysis-project/SemEval2025-Task11/refs/heads/main/task-dataset/semeval-2025-task11-dataset/track_b/train/eng.csv", "../data/train2.csv")
import os


def get_latest_csv(directory):
  files = [f for f in os.listdir(directory) if f.endswith('.csv')]
  files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
  latest_file = files[0]
  return os.path.join(directory, latest_file)

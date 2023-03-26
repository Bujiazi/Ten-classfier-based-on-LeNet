import requests
import os

def download_data(url, directory):
    os.makedirs(directory, exist_ok=True)
    filename = url.split('/')[-1]
    filepath = os.path.join(directory, filename)
    response = requests.get(url)
    with open(filepath, 'wb') as f:
        f.write(response.content)

    
if __name__ == '__main__':
    url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    directory =os.getcwd() + '/dataset'
    download_data(url, directory)

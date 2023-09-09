import requests
import os
import helpers
import json

def get_mp3_file(url):
    # It redirects the url before you get the actual file
    redirect_url = requests.get(url).url
    file = requests.get(redirect_url)
    return file

def save_mp3_file(file, file_path):
    with open(file_path, 'wb') as f:
        f.write(file.content)

if __name__ == '__main__':
    print("\n--- Downloading episodes... ---\n")
    # Obtener metadata del episodio
    base_dir = helpers.get_base_dir()
    with open(f'{base_dir}/podcast_metadata.json', 'r') as f:
        episode = json.load(f)
        
    url = episode['url']
    file_path = episode['download_episode_path']
    # Obtener el archivo de audio
    file = get_mp3_file(url)
    # Guardar el archivo de audio
    save_mp3_file(file, file_path)
    print(file_path, "saved")

            
            


            
            

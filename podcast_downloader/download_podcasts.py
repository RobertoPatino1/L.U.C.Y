from podcast import Podcast
import dateutil.parser
import requests
import re
import json
import os


def get_episodes_metadata(podcast_items):
    episode_urls = [podcast.find('enclosure')['url'] for podcast in podcast_items]
    episode_titles = [podcast.find('title').text for podcast in podcast_items]
    episode_release_dates = [parse_date(podcast.find('pubDate').text) for podcast in podcast_items]
    episode_descriptions = [podcast.find('description').text for podcast in podcast_items]
    return list(zip(episode_urls, episode_titles, episode_release_dates, episode_descriptions))

def parse_date(date):
    return dateutil.parser.parse(date).strftime('%b-%d-%Y')

def get_mp3_file(url):
    # It redirects the url before you get the actual file
    redirect_url = requests.get(url).url
    file = requests.get(redirect_url)
    return file

def save_mp3_file(file, file_path):
    with open(file_path, 'wb') as f:
        f.write(file.content)

def simplify_title(title):
    file_name = re.sub(r'[%/&!@#\*\$\?\+\^\\.\\\\]', '', title)[:100].replace(' ', '-')
    return file_name

def get_podcast_list(raw_podcast_list):
    podcast_list = []

    for raw_podcast in raw_podcast_list:
        podcast_list += [Podcast(raw_podcast['name'], raw_podcast['rss_feed_url'])]
    
    return podcast_list

def load_json(file_path):
    with open(file_path) as json_file:
        dictionary = json.load(json_file)
    return dictionary


if __name__ == '__main__':
    print("\n--- Downloading podcasts... ---\n")

    # Obtener el podcast_list
    base_dir = './Podcast-Downloader'
    podcast_list_dir = f'{base_dir}/podcast_list.json'

    raw_podcast_list = load_json(podcast_list_dir)['podcast_list']
    podcast_list = get_podcast_list(raw_podcast_list)
    
    search = 'Me sentí demasiado cansado, quisiera que no me vuelva a pasar'
    for podcast in podcast_list:
        podcast_items = podcast.search_items(search, top_limit=2)
        episodes_metadata = get_episodes_metadata(podcast_items)
        for episode in episodes_metadata:
            # Obtener el file_path del episodio en mp3
            url, title, release_date, description = episode
            simple_title = simplify_title(title)
            file_path = f'{podcast.download_directory}/{simple_title}.mp3'

            file = get_mp3_file(url)
            save_mp3_file(file, file_path)
            print(file_path, "saved")

            
            


            
            

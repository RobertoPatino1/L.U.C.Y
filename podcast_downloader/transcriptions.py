import os
import json
import time
import requests

import sys
sys.path.append('./')

from podcast_downloader.podcast import Podcast

with open('podcast.json', 'r') as f:
		podcast_data = json.load(f)

base_dir = './podcast_downloader'
assembly_ai_key = podcast_data['assembly_key']

def create_transcripts(podcast_list, **kwargs):
	all_transcription_metadata = {}
	for podcast in podcast_list:
		podcast_metadata = {}
		downloads = os.listdir(podcast.download_directory)
		for download in downloads:
			print("Uploading", download)
			file_path = f'{podcast.download_directory}/{download}'
			content_url = upload_to_assembly_ai(file_path)
			os.remove(file_path)
			transcription_id = transcribe_podcast(content_url, **kwargs)
			podcast_metadata[download] = transcription_id

		all_transcription_metadata[podcast.name] = podcast_metadata.copy()

	return all_transcription_metadata

def upload_to_assembly_ai(file_path):
	headers = {'authorization': assembly_ai_key}
	endpoint = 'https://api.assemblyai.com/v2/upload'
	response = requests.post(endpoint, headers=headers, data=read_file(file_path))
	upload_url = response.json()['upload_url']
	return upload_url

def transcribe_podcast(url, **kwargs):
	headers = {
		"authorization": assembly_ai_key,
	    "content-type": "application/json",
	}
	
	json = {'audio_url': url}
	for key, value in kwargs.items():
		json[key] = value

	print(json)
	endpoint = 'https://api.assemblyai.com/v2/transcript'
	response = requests.post(endpoint, headers=headers, json=json)
	transcription_id = response.json()['id']
	return transcription_id

def read_file(filename, chunk_size=5242880):
    with open(filename, 'rb') as _file:
        while True:
            data = _file.read(chunk_size)
            if not data:
                break
            yield data

def save_transcription_metadata(metadata, file_path='./podcast_downloader/transcripts/metadata.json'):
	with open(file_path,'w') as f:
		json.dump(metadata, f)

def load_json(file_path):
	with open(file_path) as json_file:
		dictionary = json.load(json_file)
	return dictionary

def save_transcriptions_locally(podcast_list):
	# Load transcription metadata
	metadata = load_json('./podcast_downloader/transcripts/metadata.json')
	for podcast in podcast_list:
		podcast_transcriptions = metadata[podcast.name]
		for episode, transcription_id in podcast_transcriptions.items():
			episode_name = os.path.splitext(episode)[0]
			output_path = f'{podcast.transcription_directory}/{episode_name}.txt'
			print('Trying to save', output_path)
			transcription  = wait_and_get_assembly_ai_transcript(transcription_id)
			with open(output_path, 'w') as f:
				f.write(transcription['text'])

def get_assembly_ai_transcript(transcription_id):
	headers = {'authorization': assembly_ai_key}
	endpoint = f'https://api.assemblyai.com/v2/transcript/{transcription_id}'

	response = requests.get(endpoint, headers=headers)
	return response

def wait_and_get_assembly_ai_transcript(transcription_id):
	while True:
		transcription_result = get_assembly_ai_transcript(transcription_id).json()

		if transcription_result['status'] == 'completed':
			print("Got transcript")
			break

		elif transcription_result['status'] == 'error':
			raise RuntimeError(f"Transcription failed: {transcription_result['error']}")

		else:
			print('Transcript not available, trying again in 10 seconds...')
			time.sleep(10)

	return transcription_result


if __name__ == '__main__':
	print("\n--- Transcribing episode... ---\n")

	

	podcast_list = [Podcast(podcast_data['name'], podcast_data['rss_feed_url'])]

	metadata = create_transcripts(podcast_list, language_code=podcast_data['language'])
	print('Uploaded transcripts')
	save_transcription_metadata(metadata)
	save_transcriptions_locally(podcast_list)
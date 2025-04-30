# Take input from songs_clusters.py file (recommended_songs.csv).
# Run those songs on Spotify App using spotify API function calls.
# Store the action performed on songs (Liked, Percentaged listened) for each song.
# Output file will now have meta data + Liked + Percentaged listened which will be input for preprocessing_ppoInput final.

import pandas as pd
import time
import json
import spotipy
from spotipy.oauth2 import SpotifyOAuth

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id='enter_your_client_id',
    client_secret='enter_your_client_secret',
    redirect_uri='http://localhost:8888/callback',
    scope='user-modify-playback-state user-read-playback-state user-library-read'
))


df = pd.read_csv('recommended_songs.csv')
track_ids = df['song_id'].tolist()
track_uris = [f'spotify:track:{tid}' for tid in track_ids]


def add_to_queue(track_uri):
    try:
        sp.add_to_queue(track_uri)
        print(f"Songs Queued: {track_uri}")
        return True
    except Exception as e:
        print(f"Error in adding queue: {e}")
        return False

def is_track_liked(track_uri):
    try:
        track_id = track_uri.split(":")[-1]
        return sp.current_user_saved_tracks_contains([track_id])[0]
    except Exception as e:
        print(f"Error checking liked songs: {e}")
        return False

def get_latent_vector(track_id):
    row = df[df['song_id'] == track_id]
    if not row.empty:
        return row.filter(like='latent_').values[0].tolist()
    return []

def get_track_metadata(track_id):
    row = df[df['song_id'] == track_id]
    if not row.empty:
        latent_vector = row.filter(like='latent_').values[0].tolist()
        artist = row['artist'].values[0]
        cluster = int(row['cluster'].values[0]) if 'cluster' in row else -1
        return latent_vector, artist, cluster
    return [], "Unknown", -1

def get_duration_ms(track_uri):
    try:
        track_id = track_uri.split(":")[-1]
        return sp.track(track_id)['duration_ms']
    except Exception as e:
        print(f"Error in getting duration of {track_uri}: {e}")
        return 1

def play_recommendations():
    results = []
    sp.start_playback(uris=[track_uris[0]])
    start_time = time.time()
    last_played_uri = track_uris[0]
    total_tracks = len(track_uris)
    # print(total_tracks)
    played_count = 1
    for uri in track_uris[1:]:
        add_to_queue(uri)
        time.sleep(0.5)
    while played_count < total_tracks + 1:
        try:
            playback = sp.current_playback()
            if not playback or not playback['is_playing']:
                time.sleep(2)
                continue
            current_track_uri = playback['item']['uri']
            if current_track_uri != last_played_uri:
                time_spent = time.time() - start_time  # in seconds
                track_id = last_played_uri.split(":")[-1]
                duration_ms = get_duration_ms(last_played_uri)
                duration_sec = duration_ms / 1000
                percentage_listened = round((time_spent / duration_sec), 2)
                liked = 1 if is_track_liked(last_played_uri) else 0
                latent_vector, artist, cluster = get_track_metadata(track_id)
                results.append({
                    "track_id": track_id,
                    "artist": artist,
                    "cluster": cluster,
                    "latent_vector": latent_vector,
                    "percentage_listened": percentage_listened,
                    "liked": liked
                })
                print(f"Logged: {track_id} | {percentage_listened}% | liked: {liked}")
                # Prepare for next track
                last_played_uri = current_track_uri
                start_time = time.time()
                played_count += 1

        except Exception as e:
            print(f"Playback error: {e}")
            break
        time.sleep(2)

    with open("user_song_interactions.json", "w") as f:
            json.dump(results, f, indent=2)
    print("Results saved to user_song_interactions.json")

play_recommendations()

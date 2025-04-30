# import spotipy
# from spotipy.oauth2 import SpotifyOAuth
# import time
#
# # Your credentials
# client_id = 'bb5c9237a1ef4e8f8fd8aa6122d651d8'
# client_secret = 'a6c7be4ac4e14e568ef3e5c9fae2c5bf'
# redirect_uri = 'http://localhost:8888/callback'
# scope = 'user-read-playback-state user-modify-playback-state user-read-private user-read-currently-playing user-library-read'
#
# # Authenticate
# sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
#     client_id=client_id,
#     client_secret=client_secret,
#     redirect_uri=redirect_uri,
#     scope=scope
# ))
#
# # Get account info
# user = sp.current_user()
# account_type = user.get('product', 'unknown')
# print(f"Your account type: {account_type.upper()}")
#
# # Get devices
# devices = sp.devices()
# print("\nAvailable Devices:")
# for device in devices['devices']:
#     status = "ACTIVE" if device['is_active'] else "inactive"
#     print(f"- {device['name']} ({device['type']}) [{status}]")
#
# # Check for active devices
# active_devices = [d for d in devices['devices'] if d['is_active']]
# if not active_devices:
#     print("\nNo active devices found")
# else:
#     print(f"\nActive device: {active_devices[0]['name']}")
#
#     try:
#         # Get current playback
#         current = sp.current_playback()
#         #         # Attempt a simple playback operation to test premium status
#         #         sp.current_playback()
#         #         print("\nAttempting to stop playback...")
#         #
#         #         # # Try stopping playback
#         #         # # sp.pause_playback()
#         #         # print("✅ Playback stopped successfully")
#         #         # sp.next_track()
#         #         # time.sleep(10)
#         #         # sp.previous_track()
#         #         # sp.volume(100)
#         if current and current['is_playing']:
#             track = current['item']
#             print("\n🎵 Now Playing:")
#             print(f"Track: {track['name']}")
#             # print(f"Artist(s): {', '.join([a['name'] for a in track['artists'])}")
#             print(f"Album: {track['album']['name']}")
#             print(f"Duration: {round(track['duration_ms'] / 60000, 2)} minutes")
#             print(f"Popularity: {track['popularity']}/100")
#             print(f"Explicit: {'Yes' if track['explicit'] else 'No'}")
#             print(f"DEBUG - Track ID: {track['id']}")
#
#
#             # Get audio features
#             features = sp.audio_features(track['id'])
#             print(f"Audio Features Response: {features}")
#
#         else :
#             print("\nNo track currently playing")
#
#     except Exception as e:
#         print(f"\nError fetching playback info: {e}")
#
#
# import spotipy
# from spotipy.oauth2 import SpotifyOAuth
#
# # Authentication
# sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
#     client_id = 'bb5c9237a1ef4e8f8fd8aa6122d651d8',
#     client_secret = 'a6c7be4ac4e14e568ef3e5c9fae2c5bf',
#     redirect_uri='http://localhost:8888/callback',
#     scope='user-modify-playback-state'
# ))
#
#
# def play_track_by_id(track_id):
#     try:
#         # 1. Verify track exists
#         track = sp.track(track_id)
#         if not track:
#             print("❌ Track not found")
#             return False
#
#         # 2. Get active device
#         devices = sp.devices()
#         active_devices = [d for d in devices['devices'] if d['is_active']]
#
#         if not active_devices:
#             print("⚠️ No active devices. Open Spotify on any device first.")
#             return False
#
#         # 3. Play the track
#         sp.start_playback(
#             uris=[f"spotify:track:{track_id}"],  # Format as Spotify URI
#             device_id=active_devices[0]['id']
#         )
#         print(f"🎶 Now playing: {track['name']} by {', '.join(a['name'] for a in track['artists'])}")
#         return True
#
#     except Exception as e:
#         print(f"Error: {e}")
#         return False
#
#
# # Example usage
# play_track_by_id("4FA8GD0xIwIHoExnGOdRoP")  # Daft Punk - Get Lucky

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time

# Authentication
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id='bb5c9237a1ef4e8f8fd8aa6122d651d8',
    client_secret = 'a6c7be4ac4e14e568ef3e5c9fae2c5bf',
    redirect_uri = 'http://localhost:8888/callback',
    scope='user-read-playback-state user-modify-playback-state'
))

class TrackMonitor:
    def __init__(self):
        self.play_count = 0
        self.skip_count = 0
        self.last_play_time = None
        self.current_position = 0

    def play_track(self):
        """Play the specific track"""
        try:
            # Get active device
            devices = sp.devices()
            active_devices = [d for d in devices['devices'] if d['is_active']]

            if not active_devices:
                print("⚠️ Open Spotify on any device first")
                return False

            # Play the track
            sp.start_playback(
                uris=[f"spotify:track:{TARGET_TRACK_ID}"],
                device_id=active_devices[0]['id']
            )
            print("▶️ Now playing your track")
            return True

        except Exception as e:
            print(f"Playback error: {e}")
            return False

    def monitor_interactions(self, duration=300):
        """Monitor interactions with the track for 5 minutes"""
        print(f"🔍 Monitoring interactions with track {TARGET_TRACK_ID}...")
        start_time = time.time()

        while time.time() - start_time < duration:
            current = sp.current_playback()

            if current and current['item']:
                # Track interactions
                if current['item']['id'] == TARGET_TRACK_ID:
                    if self.last_play_time is None:  # First detection
                        self.play_count += 1
                        self.last_play_time = time.time()
                        print("🎵 Track started playing")

                    # Detect skip (played <30 seconds)
                    if current['progress_ms'] < 30000 and current['is_playing']:
                        if time.time() - self.last_play_time < 30:
                            self.skip_count += 1
                            print(f"⏭ Skip detected ({self.skip_count} total)")

                # Detect replay (track restarted)
                elif self.last_play_time and current['item']['id'] != TARGET_TRACK_ID:
                    if time.time() - self.last_play_time < 10:  # Changed within 10s
                        self.play_count += 1
                        print(f"🔄 Replay detected ({self.play_count} plays total)")

            time.sleep(5)  # Check every 5 seconds

        print("\n📊 Interaction Report:")
        print(f"• Total plays: {self.play_count}")
        print(f"• Skips: {self.skip_count}")


# Usage
monitor = TrackMonitor()
if monitor.play_track():  # First play
    monitor.monitor_interactions()  # Track for 5 minutes
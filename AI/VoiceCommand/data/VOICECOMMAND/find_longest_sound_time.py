import os
import torchaudio

def find_longest_audio_duration(folder_path):
    longest_duration = 0
    longest_file = None

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3"):
            file_path = os.path.join(folder_path, filename)
            try:
                # Load the audio file
                waveform, sample_rate = torchaudio.load(file_path)
                # Calculate the duration in seconds
                duration = waveform.shape[1] / sample_rate
                if duration > longest_duration:
                    longest_duration = duration
                    longest_file = filename
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    return longest_file, longest_duration

if __name__ == "__main__":
    folder_path = r"c:\Users\dangv\Desktop\Multidisciplinary-Project-HK242\AI\VoiceCommand\data\VOICECOMMAND"
    longest_file, longest_duration = find_longest_audio_duration(folder_path)
    if longest_file:
        print(f"The longest audio file is '{longest_file}' with a duration of {longest_duration:.2f} seconds.")
    else:
        print("No valid .mp3 files found in the folder.")
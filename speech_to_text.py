import whisper

from moviepy.editor import VideoFileClip

def extract_audio_from_mp4(mp4_path, mp3_path):
    # Load the video clip
    video = VideoFileClip(mp4_path)
    
    # Extract the audio as an AudioFileClip object
    audio = video.audio
    
    # Write the audio to an MP3 file
    audio.write_audiofile(mp3_path)

    # Close the video and audio instances
    video.close()
    audio.close()

# mp4_path = input("video path: ")  # specify your MP4 file path
# mp3_path = "output_audio.mp3"  # specify the output MP3 file path

# extract_audio_from_mp4(mp4_path, mp3_path)
# print(f"Audio extracted and saved to {mp3_path}")

model = whisper.load_model("base")

def extract_text(video_path):
    extract_audio_from_mp4(video_path, "tmp.mp3")
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio("tmp.mp3")
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(result.text)
    return result.text
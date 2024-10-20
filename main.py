import streamlit as st
import subprocess
import tempfile
import os
from moviepy.editor import VideoFileClip, AudioFileClip
from google.cloud import speech
import io
import assemblyai as aai
import cv2
import google.generativeai as genai
import requests
import json

# PlayHT API initialization
playht_user_id = os.environ["PLAYHT_USER_ID"]
playht_api_key = os.environ["PLAYHT_API_KEY"]
PLAYHT_URL = 'https://api.play.ht/api/v2/tts/stream'

genai.configure(api_key=os.environ["API_KEY"])

# Function to get video length
def get_video_length(video_path):
    # Ensure the path is valid
    if not os.path.exists(video_path):
        return "Error: Video file not found."

    # Open the video file
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        return "Error: Could not open video."

    # Get the total number of frames
    frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    # Get the frames per second (fps)
    fps = video.get(cv2.CAP_PROP_FPS)

    # Calculate the duration in seconds
    duration = frames / fps if fps > 0 else 0

    video.release()
    return duration

# Function to extract audio from video
def extract_audio(video_path):
    try:
        video_clip = VideoFileClip(video_path)

        audio_path = video_path.replace(".mp4", ".mp3")
        audio_path = audio_path.replace(".mov", ".mp3") 

        video_clip.audio.write_audiofile(audio_path)

        video_clip.close()

        return audio_path
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Function to transcribe audio using AssemblyAI (API)
def transcribe_audio(audio_path):
    try:
        aai.settings.api_key = os.environ["Audio_To_Text_Api"]

        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_path)

        if transcript.status == aai.TranscriptStatus.error:
            st.error(f"Transcription failed: {transcript.error}")
            return None
        else:
            return transcript.text

    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

def text_to_audio(text):
    headers = {
        'X-USER-ID': playht_user_id,
        'AUTHORIZATION': playht_api_key,
        'accept': 'audio/mpeg',
        'content-type': 'application/json'
    }

    data = {
        'text': text,
        'voice_engine': 'Play3.0',
        'voice': 's3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json',
        'output_format': 'mp3'
    }
    
    response = requests.post(PLAYHT_URL, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        audio_file_path = os.path.join(os.getcwd(), "result.mp3")
        with open(audio_file_path, "wb") as audio_file:
            audio_file.write(response.content)
        return audio_file_path 
    else:
        st.error(f"Error: {response.text}")
        return None


def correct_transcript_genai(transcript):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"Correct the following transcript by fixing grammatical errors and removing filler words like 'umm' or 'hmm', but do not summarize or remove any meaningful content:\n{transcript}"
        response = model.generate_content(prompt)
        
        if hasattr(response, 'error') and response.error:
            st.error(f"Transcription failed: {response.error}")
            return None
        else:
            return response.text

    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

def replace_audio_in_video(video_path, audio_path, output_path):
    video_clip = VideoFileClip(video_path)
    new_audio = AudioFileClip(audio_path)
    
    final_video = video_clip.set_audio(new_audio)

    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
    
    return output_path  


# Main function for the Streamlit app
def main():
    st.title("Video to Audio and Transcript Converter")

    # Upload video file
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov"])

    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            temp_video_file.write(uploaded_video.read())
            video_path = temp_video_file.name  # Path to the saved video

        # Get video length and ensure it's less than 2 minutes
        video_length = get_video_length(video_path)
        if isinstance(video_length, str):
            st.write(video_length)  # Error message
            return

        if video_length > 120:
            st.write("Please upload a video that is less than 2 minutes long.")
            return

        # Display the uploaded video in the Streamlit app
        st.video(uploaded_video)
        st.write(f"Uploaded video: {uploaded_video.name}")

        if st.button("Submit"):
            st.write("Processing...")

            # Extract audio from the video 🗣️
            audio_path = extract_audio(video_path)
            if audio_path:
                # Provide a download link for the extracted audio file
                with open(audio_path, "rb") as audio_file:
                    st.audio(audio_file.read(), format="audio/mp3")

                # Convert audio to transcript 👀
                st.write("Transcribing audio...")
                transcript = transcribe_audio(audio_path)
                if transcript:
                    st.write("Transcript:")
                    st.text_area("Transcript", transcript)

                    correct_transcript = correct_transcript_genai(transcript)

                    if correct_transcript:
                        st.write("Corrected Transcript:")
                        st.text_area("Corrected Transcript", correct_transcript)

                        # Generate audio from corrected transcript :)
                        audio_file_path = text_to_audio(transcript)

                        if audio_file_path:
                            with open(audio_file_path, 'rb') as audio_file:
                                st.audio(audio_file.read(), format="audio/mp3")

                            # Replace audio in the video with the generated audio
                            output_video_path = "output.mp4"
                            new_video_path = replace_audio_in_video(video_path, audio_file_path, output_video_path)

                            if new_video_path:
                                st.video(new_video_path) 
                            else:
                                st.write("Failed to replace audio in video.")
                        else:
                            st.write("Failed to generate audio from text.")
                    else:
                        st.write("Failed to correct the transcript.")
            else:
                st.write("Failed to extract audio.")

    else:
        st.write("Please upload a video file.")

if __name__ == "__main__":
    main()

import requests
import speech_recognition as sr
path = "C:\\Users\\PMLS\\Desktop\\ChatbotFYP\\flaskchatbot\\audio_uploaded.mp3"
r = sr.Recognizer()
with sr.AudioFile(path) as source:
    audio_text = r.record(source)

    text = r.recognize_google(audio_text)
    print('Converting audio transcripts into text ...')
    print(text)
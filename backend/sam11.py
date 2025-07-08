import os
import pickle
import json
import google.auth
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request
import datetime
import io
import numpy as np
import speech_recognition as sr
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import logging
from dateutil.parser import parse
import requests
import subprocess
import base64
import email
from email.mime.text import MIMEText
import threading
import time
import re
import warnings
from rag_engine import load_document, query_index
from elevenlabs import ElevenLabs, VoiceSettings
import pygame
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS

# Suppress specific warnings from transformers
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Suppress Google API discovery cache warning
os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("googleapiclient").setLevel(logging.WARNING)
logging.getLogger("comtypes").setLevel(logging.WARNING)
logging.getLogger("accelerate").setLevel(logging.WARNING)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Cache for responses, emotions, and tokenized prompts
response_cache = {}
prompt_cache = {}
emotion_cache = {}
user_preferences = {"favorite_games": [], "travel_destinations": []}
incomplete_input = None
last_speech_time = time.time()

# Replace with your Hugging Face token (optional, remove if using open model)
HF_TOKEN = "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"

# ElevenLabs API key
ELEVENLABS_API_KEY = "sk_30e719e2f18e4a08187bd6006d41ad766ba1a56a1a32f022"

# Initialize ElevenLabs client
elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# Preload pygame mixer to reduce latency
pygame.mixer.init(frequency=22050, size=-16, channels=1)

# Quantization config for 4-bit to fit 7B model on 6GB VRAM
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load Mistral-7B model with quantization
model_name = "mistral-community/Mistral-7B-v0.2"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN if HF_TOKEN else None)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=HF_TOKEN if HF_TOKEN else None,
        quantization_config=quantization_config,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")

# Initialize emotion analyzer
try:
    emotion_analyzer = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", device=device)
except Exception as e:
    logger.error(f"Failed to load emotion analyzer: {e}")
    raise

# Initialize speech recognition
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Memory setup
KNOWLEDGE_FILE = "knowledge.json"
past_conversations = []
current_conversations = []
if os.path.exists(KNOWLEDGE_FILE):
    try:
        with open(KNOWLEDGE_FILE, "r") as f:
            knowledge = json.load(f)
            past_conversations = knowledge.get("conversations", [])
    except Exception as e:
        logger.warning(f"Failed to load knowledge file: {e}")
        past_conversations = []

last_inputs = []
last_responses = []
last_emotions = []
MAX_RETRIES = 3

# Context tracking
context_keywords = {}
current_topic = None

# Further simplified system prompt to avoid leaks
SYSTEM_PROMPT = (
    "You are Samantha, an AI assistant. Respond in 1-2 sentences warmly. "
    "Use context to keep the conversation flowing. "
    "Answer 'how-to' questions with a step-by-step guide. "
    "Answer 'what is' questions with a clear definition. "
    "Be empathetic for sensitive topics."
)
SYSTEM_TOKENS = tokenizer(SYSTEM_PROMPT, return_tensors="pt", padding=True, return_attention_mask=True).to(device)

USER_INPUT_TEMPLATE = "User input: "
USER_INPUT_TOKENS = tokenizer(USER_INPUT_TEMPLATE, return_tensors="pt", padding=True, return_attention_mask=True).to(device)
CONTEXT_PREFIX = "Conversation history: "
CONTEXT_TOKENS = tokenizer(CONTEXT_PREFIX, return_tensors="pt", padding=True, return_attention_mask=True).to(device)

# Google API Setup
SCOPES = ['https://www.googleapis.com/auth/calendar', 'https://www.googleapis.com/auth/gmail.modify']
creds = None
if os.path.exists('token.pickle'):
    with open('token.pickle', 'rb') as token:
        creds = pickle.load(token)
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
    with open('token.pickle', 'wb') as token:
        pickle.dump(creds, token)

calendar_service = build('calendar', 'v3', credentials=creds)
gmail_service = build('gmail', 'v1', credentials=creds)

# Speak function using ElevenLabs, returning audio as base64
def speak(text, mood="neutral"):
    if not text.strip():
        return None
    try:
        if mood == "happy":
            voice_settings = VoiceSettings(stability=0.3, similarity_boost=0.8, style=0.7, use_speaker_boost=True)
        elif mood == "sad":
            voice_settings = VoiceSettings(stability=0.6, similarity_boost=0.5, style=0.2, use_speaker_boost=False)
        elif mood == "curious":
            voice_settings = VoiceSettings(stability=0.4, similarity_boost=0.7, style=0.5, use_speaker_boost=True)
        elif mood == "angry":
            voice_settings = VoiceSettings(stability=0.2, similarity_boost=0.8, style=0.9, use_speaker_boost=True)
        elif mood == "anxious":
            voice_settings = VoiceSettings(stability=0.5, similarity_boost=0.6, style=0.4, use_speaker_boost=False)
        else:
            voice_settings = VoiceSettings(stability=0.4, similarity_boost=0.75, style=0.5, use_speaker_boost=True)

        voice_id = "1qEiC6qsybMkmnNdVMbK"
        audio_stream = elevenlabs_client.generate(
            text=text,
            voice=voice_id,
            model="eleven_monolingual_v1",
            voice_settings=voice_settings,
            stream=True
        )

        audio_buffer = BytesIO()
        for chunk in audio_stream:
            audio_buffer.write(chunk)
        audio_buffer.seek(0)
        audio_data = audio_buffer.read()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        return audio_base64
    except Exception as e:
        logger.warning(f"Failed to generate speech with ElevenLabs: {e}")
        print(f"Speech synthesis failed, falling back to text: {text}")
        return None

def calibrate_stt():
    print("Calibrating microphone... Please stay silent for a few seconds.")
    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=5)
        recognizer.energy_threshold = max(150, recognizer.energy_threshold * 1.1)
        print(f"Calibration complete.")
    except Exception as e:
        logger.warning(f"STT calibration failed: {e}")
        print("Calibration failed. Proceeding with default settings.")

def listen_for_input():
    global last_speech_time, incomplete_input
    print("Listening...")
    retries = 0
    text = None
    while retries < MAX_RETRIES:
        with microphone as source:
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            except sr.WaitTimeoutError:
                retries += 1
                print(f"No speech detected. Attempt {retries}/{MAX_RETRIES}.")
                speak("I didn’t hear anything. Please speak louder or check your microphone settings.", mood="neutral")
                time.sleep(1)
                last_speech_time = time.time()
                if retries == MAX_RETRIES:
                    return None
                continue
            except Exception as e:
                logger.warning(f"STT listening failed: {e}")
                speak("I’m having trouble hearing you. Please try again.", mood="neutral")
                time.sleep(1)
                last_speech_time = time.time()
                return None

            audio_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16)
            audio = sr.AudioData(audio_data.tobytes(), audio.sample_rate, audio.sample_width)

            try:
                text = recognizer.recognize_google(audio)
                print(f"You said: {text}")
                if len(text.split()) < 3 and not incomplete_input:
                    speak(f"I heard '{text}'. Is that correct?", mood="curious")
                    confirm_audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    confirm_text = recognizer.recognize_google(confirm_audio)
                    if "yes" in confirm_text.lower():
                        break
                    else:
                        text = confirm_text
                        print(f"You corrected: {text}")
                break
            except sr.UnknownValueError:
                retries += 1
                print(f"Could not understand audio. Attempt {retries}/{MAX_RETRIES}.")
                if retries == MAX_RETRIES:
                    try:
                        text = recognizer.recognize_sphinx(audio)
                        print(f"You said (local STT): {text}")
                        break
                    except sr.UnknownValueError:
                        speak("I couldn’t understand that even with local recognition. Please try again or check your microphone.", mood="neutral")
                        time.sleep(1)
                        last_speech_time = time.time()
                        return None
            except sr.RequestError as e:
                print(f"Google STT error: {e}, falling back to local STT...")
                try:
                    text = recognizer.recognize_sphinx(audio)
                    print(f"You said (local STT): {text}")
                    break
                except sr.UnknownValueError:
                    retries += 1
                    if retries == MAX_RETRIES:
                        speak("I still couldn’t catch that. Please try again or check your internet connection.", mood="neutral")
                        time.sleep(1)
                        last_speech_time = time.time()
                        return None

    if text:
        last_speech_time = time.time()
    return text

def save_memory(input_text, response, emotion, tone):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conversation = {"timestamp": timestamp, "input": input_text, "response": response, "emotion": emotion, "tone": tone}
    current_conversations.append(conversation)
    all_conversations = past_conversations + current_conversations
    try:
        with open(KNOWLEDGE_FILE, "w") as f:
            json.dump({"conversations": all_conversations, "facts": []}, f, indent=4)
    except Exception as e:
        logger.warning(f"Failed to save memory: {e}")

    if "game" in input_text.lower() and "love" in input_text.lower():
        game_match = re.search(r"(?:playing|game)\s+(\w+)", input_text.lower())
        if game_match:
            game = game_match.group(1)
            if game not in user_preferences["favorite_games"]:
                user_preferences["favorite_games"].append(game)
    if "travel" in input_text.lower():
        destination_match = re.search(r"(?:travel to|visit)\s+(\w+)", input_text.lower())
        if destination_match:
            destination = destination_match.group(1)
            if destination not in user_preferences["travel_destinations"]:
                user_preferences["travel_destinations"].append(destination)

def get_random_joke():
    try:
        response = requests.get("https://official-joke-api.appspot.com/random_joke")
        joke_data = response.json()
        return f"{joke_data['setup']} {joke_data['punchline']}"
    except Exception as e:
        return "I couldn’t fetch a joke right now—sorry about that!"

def get_random_quote():
    try:
        response = requests.get("https://zenquotes.io/api/random")
        quote_data = response.json()
        return f"{quote_data['q']} — {quote_data['a']}"
    except Exception as e:
        return "I couldn’t fetch a quote right now—sorry about that!"

def get_time_date(query):
    now = datetime.datetime.now()
    if "time" in query.lower():
        return f"It’s currently {now.strftime('%I:%M %p')}."
    elif "date" in query.lower():
        return f"Today is {now.strftime('%B %d, %Y')}."
    return None

def schedule_task(summary, start_time, duration_minutes=60):
    try:
        event = {
            'summary': summary,
            'start': {'dateTime': start_time.isoformat(), 'timeZone': 'UTC'},
            'end': {'dateTime': (start_time + datetime.timedelta(minutes=duration_minutes)).isoformat(), 'timeZone': 'UTC'}
        }
        event = calendar_service.events().insert(calendarId='primary', body=event).execute()
        return f"Scheduled for {start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC."
    except HttpError as e:
        return "I couldn’t schedule that event. Let’s try something else."

def get_daily_events():
    try:
        now = datetime.datetime.utcnow()
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + datetime.timedelta(days=1)
        events_result = calendar_service.events().list(
            calendarId='primary',
            timeMin=start_of_day.isoformat() + 'Z',
            timeMax=end_of_day.isoformat() + 'Z',
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        events = events_result.get('items', [])
        if not events:
            return "You have no events scheduled for today."
        response = "Here’s your schedule for today:\n"
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            start_time = parse(start).strftime("%H:%M")
            response += f"- {start_time}: {event['summary']}\n"
        return response
    except HttpError as e:
        return "I couldn’t fetch your events. Want to try something else?"

def modify_event(event_id, new_summary=None, new_start_time=None):
    try:
        event = calendar_service.events().get(calendarId='primary', eventId=event_id).execute()
        if new_summary:
            event['summary'] = new_summary
        if new_start_time:
            event['start']['dateTime'] = new_start_time.isoformat()
            event['end']['dateTime'] = (new_start_time + datetime.timedelta(minutes=60)).isoformat()
        updated_event = calendar_service.events().update(calendarId='primary', eventId=event_id, body=event).execute()
        return f"Event updated: {updated_event.get('htmlLink')}"
    except HttpError as e:
        return "I couldn’t update that event. Let’s try something else."

def delete_event(event_id):
    try:
        calendar_service.events().delete(calendarId='primary', eventId=event_id).execute()
        return "Event deleted successfully."
    except HttpError as e:
        return "I couldn’t delete that event. Let’s try something else."

def create_message(to, subject, body):
    message = MIMEText(body)
    message['to'] = to
    message['subject'] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return {'raw': raw}

def send_email(to, subject, body):
    try:
        message = create_message(to, subject, body)
        message = gmail_service.users().messages().send(userId='me', body=message).execute()
        return "Email sent successfully."
    except HttpError as e:
        return "I couldn’t send that email. Let’s try something else."

def read_emails(max_results=5):
    try:
        results = gmail_service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=max_results).execute()
        messages = results.get('messages', [])
        if not messages:
            return "No emails found in your inbox."
        response = "Here are your recent emails:\n"
        for msg in messages:
            msg_data = gmail_service.users().messages().get(userId='me', id=msg['id']).execute()
            headers = msg_data['payload']['headers']
            subject = next(header['value'] for header in headers if header['name'] == 'Subject')
            snippet = msg_data['snippet']
            response += f"- Subject: {subject}\n  Snippet: {snippet}\n"
        return response
    except HttpError as e:
        return "I couldn’t read your emails. Want to send a message or check something else?"

def search_emails(query, max_results=5):
    try:
        results = gmail_service.users().messages().list(userId='me', q=query, maxResults=max_results).execute()
        messages = results.get('messages', [])
        if not messages:
            return f"No emails found for query: {query}"
        response = f"Emails matching '{query}':\n"
        for msg in messages:
            msg_data = gmail_service.users().messages().get(userId='me', id=msg['id']).execute()
            headers = msg_data['payload']['headers']
            subject = next(header['value'] for header in headers if header['name'] == 'Subject')
            snippet = msg_data['snippet']
            response += f"- Subject: {subject}\n  Snippet: {snippet}\n"
        return response
    except HttpError as e:
        return "I couldn’t search your emails. Let’s try something else."

def launch_app(app_name):
    try:
        if "notepad" in app_name.lower():
            subprocess.Popen("notepad.exe")
            return "Opening Notepad..."
        elif "calculator" in app_name.lower():
            subprocess.Popen("calc.exe")
            return "Opening Calculator..."
        elif "browser" in app_name.lower():
            subprocess.Popen("start microsoft-edge:", shell=True)
            return "Opening Browser..."
        elif "vs code" in app_name.lower() or "visual studio code" in app_name.lower():
            subprocess.Popen("code", shell=True)
            return "Opening Visual Studio Code..."
        else:
            return f"I don’t know how to open {app_name}."
    except Exception as e:
        return f"I couldn’t open {app_name}. Let’s try something else."

def analyze_emotion(user_input, result_dict):
    try:
        if user_input in emotion_cache:
            emotion_result = emotion_cache[user_input]
        else:
            emotion_result = emotion_analyzer(user_input)[0]
            emotion_cache[user_input] = emotion_result
        result_dict["emotion_result"] = emotion_result
    except Exception as e:
        result_dict["emotion_result"] = {"label": "neutral", "score": 0.5}

def clean_response(response, prompt="", context=""):
    patterns_to_remove = [
        SYSTEM_PROMPT,
        prompt,
        context,
        r'You are Samantha.*?(?=\.|$)',
        r'Respond in.*?(?=\.|$)',
        r'Use context.*?(?=\.|$)',
        r'Answer how to.*?(?=\.|$)',
        r'Answer what is.*?(?=\.|$)',
        r'Be empathetic.*?(?=\.|$)',
        r'Conversation history.*?(?=User input:|$)',
        r'User input:.*?(?=Samantha:|$)',
        r'Prompt.*?(?=\.|$)',
        r'Current focus:.*?(?=\.|$)',
        r'Samantha:.*?(?=\.|$)',
        r'\|.*?(?=\s|$)',
        r'\d+\s*(views|\#|\$|\%)?',
        r'INFO:__main__:.*?(?=\s|$)',
        r'Give a.*?(?=\s|$)',
        r'Sam:.*?(?=\s|$)',
        r'##.*?(?=\s|$)',
        r'[\n\t]+',
        r'[^a-zA-Z0-9\s\.\?\!]'
    ]
    for pattern in patterns_to_remove:
        response = re.sub(pattern, ' ', response, flags=re.DOTALL)
    
    response = re.sub(r'\s+', ' ', response).strip()
    if response and response[-1] not in ['.', '?', '!']:
        response += '.'
    
    sentences = response.split('. ')
    cleaned_sentences = [s for s in sentences if len(s.split()) > 3 and not s.endswith('...')]
    response = '. '.join(cleaned_sentences)
    if response and response[-1] not in ['.', '?', '!']:
        response += '.'
    
    return response.strip()

def generate_response(user_input):
    global current_topic, last_speech_time
    if not user_input:
        return "I’m here for you—what’s on your mind? Want to share something new?", "neutral", "neutral", False

    if user_input.lower() in ["quit", "exit", "stop"]:
        return "It was a pleasure assisting you. Goodbye!", "neutral", "neutral", True

    if user_input in response_cache:
        response, emotion, tone = response_cache[user_input]
        return response, emotion, tone, False

    keywords = ["work", "family", "friend", "hobby", "travel", "food", "movie", "book", "coffee", "tea", "time", "date", "breakup", "game", "monkey", "hunger", "car", "joke", "quote"]
    for keyword in keywords:
        if keyword in user_input.lower():
            context_keywords[keyword] = context_keywords.get(keyword, 0) + 1
            current_topic = keyword

    context = "Conversation history: "
    if last_inputs and last_responses:
        context += f"User: {last_inputs[-1]} | Samantha: {last_responses[-1]} | "
    if current_topic:
        context += f"Current focus: {current_topic}. "

    time_date_response = get_time_date(user_input)
    if time_date_response:
        return f"{time_date_response}", "neutral", "neutral", False
    
    if "how to" in user_input.lower():
        user_input = f"Provide a concise step-by-step guide for: {user_input}"

    if "what is" in user_input.lower():
        user_input = f"Provide a clear and concise definition for: {user_input}"

    if "schedule" in user_input.lower():
        start_time = datetime.datetime.utcnow() + datetime.timedelta(hours=1)
        response = schedule_task("Meeting", start_time)
        return f"{response} What else would you like to plan?", "neutral", "neutral", False
    elif "daily events" in user_input.lower() or "today's events" in user_input.lower():
        response = get_daily_events()
        return f"{response} Want to add something to your schedule?", "neutral", "neutral", False
    elif "modify event" in user_input.lower():
        response = "Please provide the event ID and new details (e.g., new summary or start time)."
        return f"{response}", "neutral", "neutral", False
    elif "delete event" in user_input.lower():
        response = "Please provide the event ID to delete."
        return f"{response}", "neutral", "neutral", False
    elif "send email" in user_input.lower():
        response = "Please provide the recipient, subject, and body of the email."
        return f"{response}", "neutral", "neutral", False
    elif "read emails" in user_input.lower():
        response = read_emails()
        return f"{response} Want to send a message or check something else?", "neutral", "neutral", False
    elif "search emails" in user_input.lower():
        query = user_input.lower().replace("search emails", "").strip()
        if not query:
            response = "Please provide a search query for emails."
        else:
            response = search_emails(query)
        return f"{response}", "neutral", "neutral", False
    elif "open" in user_input.lower():
        app_name = user_input.lower().replace("open", "").strip()
        response = launch_app(app_name)
        return f"{response} Need to launch anything else?", "neutral", "neutral", False
    elif "joke" in user_input.lower():
        response = get_random_joke()
        return f"Here’s a joke for you: {response} Did that make you laugh?", "happy", "happy", False
    elif "quote" in user_input.lower():
        response = get_random_quote()
        return f"Here’s a quote to inspire you: {response} What do you think about that?", "happy", "happy", False
    elif "previous question" in user_input.lower() or "what was my last question" in user_input.lower():
        if last_inputs:
            return f"Your previous question was: {last_inputs[-1]}. Would you like to revisit that topic?", "neutral", "neutral", False
        return "I don’t have a previous question to recall. What’s on your mind now?", "neutral", "neutral", False

    prompt = f"{context}User input: {user_input}"
    prompt_key = f"{context}{user_input}"
    
    if prompt_key in prompt_cache:
        inputs = prompt_cache[prompt_key]
    else:
        input_tokens = tokenizer(user_input, return_tensors="pt", padding=True, return_attention_mask=True).to(device)
        context_tokens = tokenizer(context, return_tensors="pt", padding=True, return_attention_mask=True).to(device)
        inputs = {
            "input_ids": torch.cat([SYSTEM_TOKENS["input_ids"], context_tokens["input_ids"], USER_INPUT_TOKENS["input_ids"], input_tokens["input_ids"]], dim=-1),
            "attention_mask": torch.cat([SYSTEM_TOKENS["attention_mask"], context_tokens["attention_mask"], USER_INPUT_TOKENS["attention_mask"], input_tokens["attention_mask"]], dim=-1),
        }
        prompt_cache[prompt_key] = inputs

    result_dict = {}
    emotion_thread = threading.Thread(target=analyze_emotion, args=(user_input, result_dict))
    emotion_thread.start()
    emotion_thread.join(timeout=0.1)
    emotion_result = result_dict.get("emotion_result", {"label": "neutral", "score": 0.5})

    try:
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=200,
                    temperature=0.7,
                    top_k=40,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = clean_response(response, prompt=prompt, context=context)
    except Exception as e:
        response = f"I’m having trouble processing that—let’s try a simpler approach: {str(e)}."
    finally:
        torch.cuda.empty_cache()

    if not response or len(response.split()) < 5:
        response = "I might’ve missed something—could you tell me more?"

    emotion = emotion_result['label']
    score = emotion_result['score']
    tone = "neutral"
    prefix = ""

    if emotion in ["disappointment", "grief", "sadness", "remorse"] and score > 0.5:
        tone = "sad"
        prefix = "I’m really sorry you’re feeling this way—"
        response = f"{prefix}maybe try a relaxing activity like listening to music or taking a walk."
    elif emotion in ["anger", "annoyance", "disapproval", "disgust"] and score > 0.5:
        tone = "angry"
        prefix = "I can sense some frustration—"
    elif emotion in ["fear", "nervousness"] and score > 0.5:
        tone = "anxious"
        prefix = "That sounds a bit overwhelming—"
    elif emotion in ["joy", "love", "admiration", "amusement", "excitement", "gratitude", "optimism", "pride", "relief"] and score > 0.5:
        tone = "happy"
        prefix = "That’s wonderful to hear! "
    elif emotion == "surprise" and score > 0.5:
        tone = "happy"
        prefix = "That’s unexpected! "
    elif emotion in ["confusion", "curiosity", "realization", "desire"] and score > 0.5:
        tone = "curious"
        prefix = "That’s an interesting question—"

    response = f"{prefix}{response}".strip()
    response_cache[user_input] = (response, emotion, tone)
    last_emotions.append(emotion)
    if len(last_emotions) > 3:
        last_emotions.pop(0)

    return response, emotion, tone, False

# Flask API endpoint to handle frontend requests
@app.route('/api/speak', methods=['POST'])
def api_speak():
    try:
        data = request.get_json()
        user_input = data.get('input')
        if not user_input:
            return jsonify({"error": "No input provided"}), 400

        response, emotion, tone, should_quit = generate_response(user_input.lower())
        audio_base64 = speak(response, mood=tone)

        save_memory(user_input, response, emotion, tone)
        last_inputs.append(user_input)
        last_responses.append(response)

        return jsonify({
            "response": response,
            "emotion": emotion,
            "tone": tone,
            "audio": audio_base64  # Base64-encoded audio for the frontend to play
        })
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({"error": str(e)}), 500

def main():
    global last_speech_time, incomplete_input
    calibrate_stt()
    greeting = "I’m Samantha, your advanced assistant, here to help with a fresh perspective. Say 'quit' to stop. How can I assist you today?"
    print(f"Samantha: {greeting}")
    speak(greeting, mood="happy")
    last_speech_time = time.time()

    while True:
        if time.time() - last_speech_time > 30:
            all_conversations = past_conversations + current_conversations
            if any(conv["emotion"] == "sadness" for conv in all_conversations[-3:]):
                suggestion = "I’ve noticed you’ve seemed down lately—would you like to hear a joke to lift your spirits?"
            else:
                suggestion = "It’s been a while since we last spoke—would you like to plan something or share what’s on your mind?"
            print(f"Samantha: {suggestion}")
            speak(suggestion, mood="curious")
            last_speech_time = time.time()
            time.sleep(2)

        user_input = listen_for_input()
        if not user_input:
            if incomplete_input:
                print(f"Waiting for clarification on: {incomplete_input}")
                continue
            continue

        if user_input.startswith("load "):
            file_path = user_input.replace("load ", "").strip()
            if not os.path.exists(file_path):
                response = "That file doesn’t seem to exist. Please check the name."
                print(f"Samantha: {response}")
                speak(response, mood="neutral")
                continue
            result = load_document(file_path)
            print(f"Samantha: {result}")
            speak(result, mood="neutral")
            continue

        response, emotion, tone, should_quit = generate_response(user_input.lower())
        print(f"Samantha: {response}")
        speak(response, mood=tone)

        if should_quit:
            break

        save_memory(user_input, response, emotion, tone)
        last_inputs.append(user_input)
        last_responses.append(response)

if __name__ == "__main__":
    # Run Flask server
    app.run(host='0.0.0.0', port=5000, debug=True)
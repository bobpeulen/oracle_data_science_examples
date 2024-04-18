
import os
import ocifs
import numpy as np
import torch
import pandas as pd
import whisper
import torchaudio
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import ffmpeg
import ocifs
from json import loads, dumps
from qdrant_client import models, QdrantClient
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#fetch environmment variables:
recording_name = os.environ.get("recording_name", "dammac recording")
print("Fetching environment variable called " + recording_name)


######## ----------------------------------------------------- #################
# Step 1
######## ----------------------------------------------------- #################

#create local folder (in job)
path_input_locally = "/home/datascience/input_recording/"
bucket = "oci://West_BP@frqap2zhtzbe/dammac/"

try:       
    if not os.path.exists(path_input_locally):         
        os.makedirs(path_input_locally)    

except OSError: 
    print ('Error: Creating directory of input recording')

#copy recording from bucket to local folder
fs = ocifs.OCIFileSystem()
print(fs.ls(bucket))
fs.get(bucket, path_input_locally, recursive=True, refresh=True)

######## ----------------------------------------------------- #################
# Step 2 - Load model and Detect languages
######## ----------------------------------------------------- #################

#load whisper model
model = whisper.load_model("base")

#for each recording in the folder detect langauge
for recording in os.listdir(path_input_locally):
        if (recording.endswith(".mp3")):
            
            audio_recording = os.path.join(path_input_locally, recording)
            
            # load audio and pad/trim it to fit 30 seconds
            audio = whisper.load_audio(audio_recording)
            audio = whisper.pad_or_trim(audio)
            

            # make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            # detect the spoken language
            _, probs = model.detect_language(mel)
            print(f"Detected language: {max(probs, key=probs.get)}")




######## ----------------------------------------------------- #################
# Step 3 Transcripte recordings
######## ----------------------------------------------------- #################

output = []

##ffmpeg
!wget -O - -q  https://github.com/yt-dlp/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz | xz -qdc| tar -x
    
for recording in os.listdir(path_input_locally):
        if (recording.endswith(".mp3")):
            
            audio_recording = os.path.join(path_input_locally, recording)
            
            #transcribe recording
            result = model.transcribe(audio_recording)
            
            #append result for each recording to list
            output.append(result['text'])
                       
            print(recording + " is transcribed")

#all in dataframe
df_transcriptions = pd.DataFrame(output, columns=['text'])

######## ----------------------------------------------------- #################
# Step 3 Create embeddings
######## ----------------------------------------------------- #################

encoder = SentenceTransformer('multi-qa-distilbert-cos-v1') # Model to create embeddings

## transform df_transcritiopons to correct format for Qrant
texts_damac = df_transcriptions.to_json(orient = 'records')
documents_damac = loads(texts_damac)

#establish connection to Qdrant vector database
qdrant = QdrantClient("138.3.241.32", port=6333) # Create in-memory Qdrant instance

# Create collection to store books
qdrant.recreate_collection(
    collection_name="damac_v1",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(), # Vector size is defined by used model
        distance=models.Distance.COSINE
    )
)



# Let's vectorize descriptions and upload to qdrant
qdrant.upload_records(
    collection_name="damac_v1",
    records=[
        models.Record(
            id=idx,
            vector=encoder.encode(doc["text"]).tolist(),
            payload=doc
        ) for idx, doc in enumerate(documents_damac)
    ]
)

print("-------------------------------------------------------")
print("-------------------------------------------------------")
print("Encoded Text / Embeddings are pushed to Qdrant")
print("-------------------------------------------------------")
print("-------------------------------------------------------")

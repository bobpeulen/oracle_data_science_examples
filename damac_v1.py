
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
from dammac.vector_engine.vector_engine.utils import vector_search, id2details                           ## this is different for Job vs notebook because of folder structure
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#fetch environmment variables:
recording_name = os.environ.get("recording_name", "dammac recording")
print("Fetching environment variable called " + recording_name)

#ffmpeg
os.system("cp ./dammac/ffmpeg ./usr/bin/ffmpeg")  #copy file to other direcotry
os.system("cp ./dammac/ffprobe ./usr/bin/ffprobe")

os.system("chmod +rwx ./usr/bin/ffprobe  #change permission")
os.system("chmod +rwx ./usr/bin/ffmpeg")


######## ----------------------------------------------------- #################
# Step 1
######## ----------------------------------------------------- #################

#create local folder (in job)
path_input_locally = "/home/datascience/input_recording/" 

try:       
    if not os.path.exists(path_input_locally):         
        os.makedirs(path_input_locally)    

except OSError: 
    print ('Error: Creating directory of input recording')

#copy recording from bucket to local folder
fs = ocifs.OCIFileSystem()
fs.invalidate_cache("oci://West_BP@frqap2zhtzbe/dammac/*.mp3")
fs.get("oci://West_BP@frqap2zhtzbe/dammac/*.mp3", path_input_locally , recursive=True, refresh=True)

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

#create an id column. 
df_transcriptions['id_index'] = df_transcriptions.index

# Instantiate the sentence-level DistilBERT
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# Check if CUDA is available ans switch to GPU
if torch.cuda.is_available():
    model = model.to(torch.device("cuda"))
print(model.device)

# Convert abstracts to vectors
embeddings = model.encode(df_transcriptions.text.to_list(), show_progress_bar=True)

######## ----------------------------------------------------- #################
# Step 4 Apply FAISS
######## ----------------------------------------------------- #################

# Step 1: Change data type
embeddings = np.array([embedding for embedding in embeddings]).astype("float32")

# Step 2: Instantiate the index
index = faiss.IndexFlatL2(embeddings.shape[1])

# Step 3: Pass the index to IndexIDMap
index = faiss.IndexIDMap(index)

# Step 4: Add vectors and their IDs
index.add_with_ids(embeddings, df_transcriptions.id_index.values)

print(f"Number of vectors in the Faiss index: {index.ntotal}")

print("-------------------------------------------------------")
print("-------------------------------------------------------")
print("Embeddings are created and FAISS is applied. Pickle files or embeddings can be pushed to Vector DB or anything else")
print("-------------------------------------------------------")
print("-------------------------------------------------------")

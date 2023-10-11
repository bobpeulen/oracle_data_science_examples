# **oracle_data_science_examples**
Example notebooks in using OCI Data Science, from deploying GANs to small examples in how to work with autonomous database.

## 1. Running and deploying DeepFace in Face Recognition and comparison (deepface_oda_v3.ipynb)
- Deploy 3 different Deep Face models in one model deployment. Invoke as REST API
- Create two images in one json payload

## 2. Speech to Text using Whisper, Generate Embeddings and apply indexing using FAISS (whisper_embeddings_faiss.ipynb)
- Loop through multiple recordings (mp3 files) and transcribe using Whisper
- Generate Embeddings using BERT on transcribed texts
- Apply FAISS for indexing
- Create one Job to automate newly incoming recordings

## 3. Speech to Text using Whisper, Generate and Store Embeddings using Qdrant Vector Database (speech_to_text_qdrant_vector_db.ipynb)
- Loop through multiple recordings (mp3 files) and transcribe using Whisper
- Generate and store embeddings in Qdrant Vector Database using Qdrant Python SDK
- Qdrant Vector Database running on seperate instance in Oracle Cloud

## 4. Different examples in OCI Data Science (Custom_Conda_Run_Jobs_Examples.ipynb)
- Create and publish a custom conda environment
- Authenticate using config file
- Create connection with an Autonomous Database
- Push PD dataframe to the Autonomous Database as a new table (and append)
- Create and run a Data Science Jobs, using a .py file stored in local (Data Science) directory, with environment variables referring to Object Storage buckets

## 5. Delete log groups (delete_logs.ipynb)
- Batch delete logs inside a Log Group

 ## 6. Batch delete all items in a Compartment (batch_delete_projects.ipynb)
- Batch delete all model deployments
- Batch delete all models in model catalog
- Batch delete all notebook sessions
- Batch delete all projects

## 7. CTGAN (deploying_gan.ipynb)
- Use CTGan as a deployed model on OCI to generate tabular synthetic data automatically on input .csv files, output is new rows of .csv files

## 8. CTGAN in notebook session (generate_synthetic_data.ipynb)
- Use CTGan to generate synthetic .csv file. From object storage to object storage
- Use CTGan to generate synthetic data from autonomous database table, to a new, synthetic database table

## 9. Video Analytics on Soccer Video (demo_spl.ipynb)
- Use open source to track scoccer players, the ball, and referee in a video
- Use a custom trained AI Vision model (recognizing logos near the soccer pitch) to extract the logos from the video. E.g., how often was "xx" logo in screen?

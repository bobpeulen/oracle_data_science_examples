# **oracle_data_science_examples**
Example notebook in using OCI Data Science, from deploying GANs to small examples in how to work with autonomous database.

## Running and deploying DeepFace in Face Recognition and comparison (deepface_oda_v3.ipynb)
- Deploy 3 different Deep Face models in one model deployment. Invoke as REST API
- Create two images in one json payload

## Different examples in OCI Data Science (Custom_Conda_Run_Jobs_Examples.ipynb)
- Create and publish a custom conda environment
- Authenticate using config file
- Create connection with an Autonomous Database
- Push PD dataframe to the Autonomous Database as a new table (and append)
- Create and run a Data Science Jobs, using a .py file stored in local (Data Science) directory, with environment variables referring to Object Storage buckets

## Delete log groups (delete_logs.ipynb)
- Batch delete logs inside a Log Group

## CTGAN (deploying_gan.ipynb)
- Use CTGan as a deployed model on OCI to generate tabular synthetic data automatically on input .csv files, output is new rows of .csv files

## CTGAN in notebook session (generate_synthetic_data.ipynb)
- Use CTGan to generate synthetic .csv file. From object storage to object storage
- Use CTGan to generate synthetic data from autonomous database table, to a new, synthetic database table

## Deploying GPT2 (deploying_gpt2.ipynb)
- Using OCI Data Science to locally load and test GPT2 and storing the model in the Model Catalog
- Deploying the GPT2 model and making a prediction using the deployed model

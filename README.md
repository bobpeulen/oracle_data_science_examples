# oracle_data_science_examples
Small example notebooks to use on Oracle Data Science

## Examples in Custom_Conda_Run_Jobs_Examples.ipynb
- Create and publish a custom conda environment
- Authenticate using config file
- Create connection with an Autonomous Database
- Push PD dataframe to the Autonomous Database as a new table (and append)
- Create and run a Data Science Jobs, using a .py file stored in local (Data Science) directory, with environment variables referring to Object Storage buckets

## Delete log groups in delete_logs.ipynb
- Batch delete logs inside a Log Group

## CTGAN in deploying_gan.ipynb
- Use CTGan as a deployed model on OCI to generate tabular synthetic data automatically on input .csv files, output is new rows of .csv files

## CTGAN in notebook session (generate_synthetic_data.ipynb)
- Use CTGan to generate synthetic .csv file. From object storage to object storage
- Use CTGan to generate synthetic data from autonomous database table, to a new, synthetic database table

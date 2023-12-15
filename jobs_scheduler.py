%%writefile ./scheduler_jobs.py

import schedule
import time
import ads
from ads.jobs import Job, DataScienceJobRun

#####################
#################### Define Jobs OCIDS
####################

#*********************************************************#
## set Job OCIDS
JOB_OCIDS = ["ocid1.datasciencejob.oc1.eu-frankfurt-1.amaaaaaangencdya2vnth3iaslq34nastb4ez6wlo6y2vnpo65iigefbcftq", "ocid1.datasciencejob.oc1.eu-frankfurt-1.amaaaaaangencdya2vnth3iaslq34nastb4ez6wlo6y2vnpo65iigefbcftq"]

## Set the frequency in minutes
frequency = 5
#*********************************************************#

##################
################## Function
##################

def job_api(JOB_OCIDS):
    
    for job_ocid in JOB_OCIDS:
        
        print("Job OCID = " +str(job_ocid))
        
        #load Job
        job = Job.from_datascience_job(job_ocid)
        
        #Run Job
        job.run()    

# run the function job() every XX minute
schedule.every(frequency).minutes.do(job_api, JOB_OCIDS)

while True:  
    schedule.run_pending()  

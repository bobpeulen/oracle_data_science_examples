import oci
import requests
from oci.signer import Signer
import io
import json
import logging
from fdk import response


def handler(ctx, data: io.BytesIO = None):
    logging.getLogger().info("Inside handler")
    try:
        signer = oci.auth.signers.get_resource_principals_signer()

        bucket_name = "LagMetricFiles"
        namespace = "frzpemb9ufe8"
        #sub_bucket = "LagMetrics/ocid1.serviceconnector.oc1.eu-frankfurt-1.amaaaaaapixtsjiarpunoxhi6tvcw3jdbgwyt6xzv4cl4zryhuigxesyyjmq"
    
        body = json.loads(data.getvalue())
        latest_file_name = body["data"]["resourceName"]

        #cal the HTTP endpoint
        calldata = {'file_name':latest_file_name, 'bucket_name':bucket_name, 'namespace':namespace}
        logging.getLogger().info(str(calldata))

        url = "https://modeldeployment.eu-frankfurt-1.oci.customer-oci.com/ocid1.datasciencemodeldeployment.oc1.eu-fr[…]wdvkiqwjwhjqslqtbaj3re5hvgunquc2u2a/predict"

        resp = requests.post(url, json=calldata, auth=signer)
        full_response = resp.json()
    except (Exception, ValueError) as ex:
        logging.getLogger().info('error: ' + str(ex))
    return response.Response(
        ctx,
        response_data={"result":"success","filename":latest_file_name,"response":full_response},
        headers={"Content-Type": "application/json"}
    )

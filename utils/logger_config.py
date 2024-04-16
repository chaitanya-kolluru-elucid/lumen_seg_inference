import logging
import os
import datetime

logger = logging

# Configure logging
filename = os.path.join('./logs', f'{datetime.datetime.now():%d-%b-%y-%H:%M:%S}-run_inference_pipeline.log')

logger.basicConfig(filename=filename, 
                    filemode='w', 
                    level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#%%
# import libraries
import os
import csv
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

#%%
# Path to load the file
path = '/home/esssfff/Documents'
file = 'telco-customer-churn.csv.gz'

class Split(beam.DoFn):
    """Converts line into dictionary"""
    def process(self, element):
        element = element.encode('utf-8')
    
        cols = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 
                'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 
                'StreamingMovies', 'Contract', 'PaperlessBilling', 
                'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']
    
        line = element.split(',')

        return [{k:v for k,v in zip(cols, line)}]

# Create the pipeline
with beam.Pipeline(options=PipelineOptions()) as p:
    
    # Read each line
    lines = (p | 'Read_File' >> 
        beam.io.ReadFromText(os.path.join(path, file),
                             skip_header_lines=1))

    binary = (lines
        | "Read_Lines" >>
            beam.ParDo(Split())
        | "Write_File" >> 
            beam.io.textio.WriteToText(os.path.join(path,'extracted'))    
        )    

    p.run()


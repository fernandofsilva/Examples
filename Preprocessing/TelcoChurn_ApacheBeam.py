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
    def process(self, element, cols):
        # Define the code
        element = element.encode('utf-8')

        #Split the lines in comma
        line = element.split(',')

        # return a dictionary with col name as key
        return [{k:v for k,v in zip(cols, line)}]

class Binary(beam.DoFn):
    """Converts """
    def process(self, element, cols):
        pass      

# Create the pipeline
with beam.Pipeline(options=PipelineOptions()) as p:
    
    cols = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
            'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
            'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 
            'TotalCharges', 'Churn']

    # Read each line and split by comma
    lines = (p 
            | 'Read_File' >> 
               beam.io.ReadFromText(os.path.join(path, file), 
                                    skip_header_lines=1)
            | 'Read_Lines' >>
               beam.ParDo(Split(), cols)
            )

    bin_cols = ['Partner', 'PaperlessBilling', 'PhoneService', 'Dependents',
                'Churn']
    
    # Transform Binary Columns
    binary = (lines
              | beam.ParDo(Binary(), bin_cols)
              )
    
    #     
    
    
    
    
    write = (lines
        | "Write_File" >> 
           beam.io.textio.WriteToText(os.path.join(path,'extracted'))    
        )    

    p.run()


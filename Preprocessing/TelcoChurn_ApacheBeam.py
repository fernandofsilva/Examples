#%%
# import libraries
import os
import csv
import apache_beam as beam

#%%
# Path to load the file
path = '/home/esssfff/Documents'
file = 'telco-customer-churn.csv.gz'

def bin(value):
    if value == 'Yes':
        return 1
    else:
        return 0


with beam.Pipeline('DirectRunner') as pipeline:
    
    binary = (pipeline
        | "Read_Csv_File" >> 
            beam.io.ReadFromText(os.path.join(path, file))
        | "Read_Lines" >>
            beam.Map(lambda line: next(csv.reader([line])))
        | "Extract_Fields" >> 
            beam.Map(lambda fields: (fields[0], (bin(fields[3]), 
                                                 bin(fields[4]))))
            )

    (binary 
        | "Map_Tupples" >> 
            beam.Map(lambda (binary, data): 
                '{},{}'.format(binary, ','.join(data))) 
        | "Write_File" >> 
            beam.io.textio.WriteToText(os.path.join(path,'extracted')))

    pipeline.run()


#%%
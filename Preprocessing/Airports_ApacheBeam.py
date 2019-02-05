#%%
# import libraries
import os
import csv
import apache_beam as beam

#%%
# Path to load the file
path = '/home/esssfff/Documents/Git/Examples/Datasets/'

with beam.Pipeline('DirectRunner') as pipeline:
    
    airports = (pipeline
        | beam.io.ReadFromText(path+'airports.csv.gz')
        | beam.Map(lambda line: next(csv.reader([line])))
        | beam.Map(lambda fields: (fields[0], (fields[21], fields[26])))
        )

    (airports 
        | beam.Map(lambda (airport, data): '{},{}'.format(airport, ','.join(data))) 
        | beam.io.textio.WriteToText(path+'extracted_airports'))

    pipeline.run()


#%%
lambda (airport, data): '{},{}'.format(airport, ','.join(data))
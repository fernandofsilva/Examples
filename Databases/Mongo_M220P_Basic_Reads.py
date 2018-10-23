#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: esssfff
"""

import pymongo

# defined the uri for the connection on the altas cluster
uri = "mongodb+srv://m220student:m220password@mflix-j3hvq.mongodb.net/test"

# connection with the uri defined
client = pymongo.MongoClient(uri)
#client = pymongo.MongoClient(uri, connectTimeoutMS=200, retryWriters=True)

# status of the connection
client.stats

# check the list of the databases
client.list_database_names()

# connect in the database
mflix = client['mflix']

# check the list of the collection inside in mflix database
mflix.list_collection_names()

# connec in the collection
movies = mflix.movies

# count the number of documents in a collection 
movies.count_documents({})

# Return the first document of any kind
movies.find_one()

# find the firt document there Salma Hayek in the cast 
movies.find_one( { "cast": "Salma Hayek" } )

# Return a cursor option that can be treat as any python iterable
movies.find( { "cast": "Salma Hayek" } )

# count the documents of the cursor
movies.find( { "cast": "Salma Hayek" } ).count()

# access the documents in the cursor
cursor = movies.find( { "cast": "Salma Hayek" } )
from bson.json_util import dumps
print(dumps(cursor, indent=2))

# access the document of the curos but show just the title
cursor = movies.find( { "cast": "Salma Hayek" }, { "title": 1 } )
print(dumps(cursor, indent=2))

# remove the id of the documents
cursor = movies.find( { "cast": "Salma Hayek" }, { "title": 1, "_id": 0 } )
print(dumps(cursor, indent=2))

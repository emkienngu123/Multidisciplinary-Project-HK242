from env import *
from Adafruit_IO import Client
import csv

USERNAME = AIO_USERNAME
KEY      = AIO_KEY
aio      = Client(USERNAME, KEY)

feeds =  AIO_FEED_ID

for feed in feeds:
    feed_key = feed
    all_entries = aio.data(feed_key, max_results=None)
    csv_filename = feed_key + '.csv'
    fieldnames   = ['id', 'created_at', 'value']

    with open(csv_filename, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Write each Adafruit IO data entry as a row
        for entry in all_entries:
            writer.writerow({
                'id':         entry.id,
                'created_at': entry.created_at,
                'value':      entry.value
            })






import json
import os
import pandas as pd
import h5py


def get_path(filename):
    filepath = os.path.join("data", filename)
    return filepath

if __name__ == "__main__": 
    '''data summary stats'''
    # Load the JSON file
    with open(get_path("fin_data_summary_stats.json"), 'r') as file:
        summary_status = json.load(file)

    # Explore its content
    # print(json.dumps(summary_status, indent=4))

    ''' metadata'''
    # Load the CSV file into a DataFrame
    metadata_df = pd.read_csv(get_path('fin_data_metadata.csv'))

    # Explore its content
    #print(metadata_df.head())

    '''h5 file'''
    path = get_path("final_dataset.h5")
    with h5py.File(path, 'r') as file:
        #data = file['name']
        
        # get the values ranging from index 0 to 15
        print(file)





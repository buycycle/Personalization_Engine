import pandas as pd
def read_json_logs(file_path):
    with open(file_path, 'r') as file:
        data = [eval(line.strip().replace('null', 'None')) for line in file.readlines()]
    df = pd.DataFrame(data)
    return df



def extract_bike_id(df, levelname):
    df = df[df['levelname'] == levelname]
    ids = df['bike_id'].unique()
    ids = pd.DataFrame(ids)

    return ids.astype('int')
    
def write_ids(log_path, out_path, levelname):
    df = read_json_logs(log_path)
    df = extract_bike_id(df, levelname)
    df.to_csv(out_path)


import requests
import os
import shutil
import pandas as pd


class IMDBDataProcessor:
    
    def __init__(self, base_dir):

        self.base_dir = base_dir
    
    def download_and_unpack_data(self, base_url):
        """
        Download the compressed data from the file server and unpack
        """
        
        self.filename = base_url.split("/")[-1]
        # download to local workspace
        cf = requests.get(base_url, allow_redirects=True)
        with open(os.path.join(self.base_dir, self.filename), 'wb') as f:
            f.write(cf.content)
    
        # unpack the compressed file
        shutil.unpack_archive("{}/{}".format(self.base_dir, self.filename), extract_dir="inputs")
    
    def create_dataframe_from_txt_files(self, data_path, sample):
        """
        create a pandas dataframe by parsing all reviews from train and test folders
        """

        # pos folder
        f_path = "{}/pos".format(data_path)
        files = os.listdir(f_path)
        f_data = [open(os.path.join(f_path,f),encoding="ISO-8859-1").read() for f in files] 
        pos_df = pd.DataFrame(data=list(zip(files, f_data)), columns=["filename", "body"])
        pos_df["label"] = 1

        # neg folder
        f_path = "{}/neg".format(data_path)
        files = os.listdir(f_path)
        f_data = [open(os.path.join(f_path,f),encoding="ISO-8859-1").read() for f in files] 
        neg_df = pd.DataFrame(data=list(zip(files, f_data)), columns=["filename", "body"])
        neg_df["label"] = 0

        return pd.concat([pos_df.head(sample), neg_df.head(sample)])
    
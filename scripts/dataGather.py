import pandas as pd
import subprocess
import os
from FredFetch import downloadPopData

state = "Michigan"
county = "Alcona"
save_to_parquet = True
projectDirectory = os.getcwd()

downloadPopData(state, county, projectDirectory)

subprocess.run(["Rscript", "scripts/Fredify.R", county, str(int(save_to_parquet))], check=True)

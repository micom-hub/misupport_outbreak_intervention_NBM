import pandas as pd
import subprocess
import os
from scripts.FredFetch import downloadPopData

state = "Michigan"
county = "Chippewa"
save_to_parquet = True
projectDirectory = os.getcwd()

downloadPopData(state, county, projectDirectory)

subprocess.run(["Rscript", "scripts/Fredify.R", county, str(int(save_to_parquet))], check=True)

import os
import pandas as pd
import json

from scripts.FredFetch import downloadPopData
from scripts.SynthDataProcessing import synthetic_data_process
from scripts.network_model import  ModelParameters, DefaultModelParams, NetworkModel  # noqa: F401

#suppressing plot outputs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.show = lambda *args, **kwargs: None #monkeypatch it

############################################
# Run Settings: Alter runParameters values #
############################################

runParameters: ModelParameters = {
#Epidemiological Parameters
    "base_transmission_prob": 0.8,
    "incubation_period": 10.5,
    "infectious_period": 5,
    "gamma_alpha": 20,
    "incubation_period_vax": 10.5,
    "infectious_period_vax": 5,
    "relative_infectiousness_vax": 0.05,
    "vax_efficacy": 0.997,
    "vax_uptake": 0.25,
    "susceptibility_multiplier_under_five": 2.0,


#Number of contacts assigned to each individual from each location
    "wp_contacts": 10,
    "sch_contacts": 10,
    "gq_contacts": 20,
    "cas_contacts": 10,

#Weighting of each contact type
    "hh_weight": 1,
    "wp_weight": .5,
    "sch_weight": .6,
    "gq_weight": .3,
    "cas_weight": .1,

#Simulation settings
    "n_runs": 5,
    "run_name": "driver_run",
    "overwrite_edge_list": True, #Must be true to render changes in contact structure
    "simulation_duration": 45,
    "dt": 1,
    "I0": [22],
    "seed": 2026,
    "county": "Alcona", 
    "state": "Michigan",
    "save_plots": True,
    "save_data_files": True,
    "make_movie": False,
    "display_plots": True
}

# ----- Process data for model run -----

#Check if /data contains a file named for params["county"]
county = runParameters["county"].lower().capitalize()
state = runParameters["state"].lower().capitalize()
cd = os.getcwd()

#Check if project directory has a data folder
data_dir = os.path.join(cd, "data")
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

#Check if the data folder contains the county's synthetic population data, if not download it
files = os.listdir(data_dir)
matching = [f for f in files if f.startswith(county)]

if not matching:
    downloadPopData(state = state, county = county)

    matching = [f for f in os.listdir(data_dir) if f.startswith(county)]

#Check if there is already a contacts.parquet file within matching
countyfoldername = matching[0]
countyfolder = os.path.join(data_dir, countyfoldername)

if countyfolder.endswith('.zip'):
    contacts_df = synthetic_data_process(county, save_files = runParameters["save_data_files"])
    if runParameters["save_data_files"]: #if saving data as a folder, delete the zip
        os.remove(countyfolder)
else:
    parquet_saved = False
    for filename in os.listdir(countyfolder):
        if filename == "contacts.parquet":
            parquet_saved = True
            parquet_path = os.path.join(countyfolder, filename)

    if parquet_saved:
        contacts_df = pd.read_parquet(parquet_path)
    else:
        contacts_df = synthetic_data_process(county, save_files = runParameters["save_data_files"])



# ---- Run the Model ------

print("Initializing model...")
model = NetworkModel(contacts_df = contacts_df, params = runParameters)

print("Running model...")
model.simulate()

print("Building plots...")
for run in range(model.n_runs):
    model.epi_curve(run_number= run, suffix = f"run_{run+1}")
    model.cumulative_incidence_plot(run_number= run, 
        suffix = f"run_{run+1}", strata = "age")
    model.cumulative_incidence_plot(run_number= run, 
        suffix = f"run_{run+1}", strata = "sex")
model.cumulative_incidence_spaghetti()




if runParameters["make_movie"]:
    model.make_movie()

#Save parameters to json file and save graphml file
if runParameters["save_data_files"]:
    model.make_graphml_file(t = model.simulation_end_day)
    with open(model.results_folder + "/run_parameters.json", "w") as f:
        json.dump(runParameters, f, indent = 4)



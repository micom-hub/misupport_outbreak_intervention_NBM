# Script to go and download the FRED synthetic data for a specified state and county
#NOTE uses chromedriver which won't be supported later 2026
#NOTE check occasionally to ensure that the synthpop download url hasn't changed

from selenium import webdriver 
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

import os
import re
import glob
import zipfile

#@profile
def downloadPopData(state, county, projectDirectory = os.getcwd()):
    """Function that downloads population data from the FRED Public Health Synthetic Population Website
    Wheaton, W.D., U.S. Synthetic Population 2010 Version 1.0 Quick Start Guide, RTI International, May 2014.

    *Note, will overwrite prior population files for the same county

    Args: --- ENSURE CORRECT SPELLINGS
        state: Name of a US State
        county: Name of a County in said state
        projectDirectory: Filepath to where the project is stored

    Output: A {county}.zip file in the data folder of the projectDirectory
    """
    cd = projectDirectory
    data_dir = cd + "/data"

    options = webdriver.ChromeOptions()
    prefs = {
        "download.default_directory": data_dir,
        "download.prompt_for_download": False,
        "directory_upgrade": True,
    }
    options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=options)

    driver.get("https://fred.publichealth.pitt.edu/syn_pops")

    # Normalize case
    state = state.lower().capitalize()
    county = county.lower().capitalize()

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "countries"))
    )
    country_dropdown = Select(driver.find_element(By.ID, "countries"))
    country_dropdown.select_by_visible_text("USA")

    WebDriverWait(driver, 15).until(
        lambda d: len(Select(d.find_element(By.ID, "states")).options) > 1
    )

    state_dropdown = Select(driver.find_element(By.ID, "states"))
    state_dropdown.select_by_visible_text(state)

    WebDriverWait(driver, 15).until(
        lambda d: len(Select(d.find_element(By.ID, "file")).options) > 1
    )
    county_dropdown = Select(driver.find_element(By.ID, "file"))
    county_dropdown.select_by_visible_text(county)

    WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "download")))
    download_button = driver.find_element(By.ID, "download")
    download_button.click()

    # Wait for download to finish (very basic!)
    import time

    time.sleep(1)  # Increase if file is large!

    driver.quit()

    # Rename the zip file
    zip_files = glob.glob(os.path.join(data_dir, "*.zip"))
    if not zip_files:
        raise FileNotFoundError("No Zip file was downloaded to {data_dir}")
    # find the most recent one
    latest_zip = max(zip_files, key=os.path.getmtime)

    # find the numeric prefix as it is important
    match = re.search(r"/(\d+)", latest_zip)
    if match:
        county_prefix = match.group(1)

    # Create new zip path name the folder
    new_zip_path = os.path.join(data_dir, f"{county}.zip")
    # shutil.move(latest_zip, new_zip_path)

    if county_prefix:  # if zip name contains FRED code rather than county name

        # For each file in the original zipfile (county_prefix.csv), change to (county.csv)
        with zipfile.ZipFile(latest_zip, "r") as zin, zipfile.ZipFile(
            new_zip_path, "w"
        ) as zout:
            for item in zin.infolist():
                filename = item.filename
                new_name = re.sub(re.escape(county_prefix), county, filename)
                with zin.open(item) as source:
                    zout.writestr(new_name, source.read())
        os.remove(latest_zip)
    print(f"Data for {county} County, {state} saved to {new_zip_path}")

    return(new_zip_path)


if __name__ == "__main__":
    state = "Michigan"
    county = "Chippewa"
    downloadPopData(state, county, os.getcwd())

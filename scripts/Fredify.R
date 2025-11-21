#Input: zip file downloaded from FRED synthetic population repo named {county}.zip
#Output: DFs containing necessary information for contact networks
#Takes approximately 3 seconds per 1,000,000 population for a given county from zip -> df
#Takes about 20 seconds per 1,000,000 population to save df -> .parquet file
if (!require(tidyverse)) {install.packages("tidyverse")}
if (!require(jsonlite)) {install.packages("jsonlite")}
if (!require(data.table)) {install.packages("data.table")}
if (!require(arrow)) {install.packages("arrow")}


args <- commandArgs(trailingOnly = TRUE)
#Take arguments when called from command line/script, must be in order of county name, and whether or not to
county <- str_to_title(str_to_lower(args[1]))
save_files <- as.logical(as.integer(args[2]))

# Check Files, Unzip FRED file, and set up workspace -------------------------------------------

data_dir <- file.path(getwd(), "data")
if (!dir.exists(data_dir)) {
  stop("Attempted to handle data before downloading FRED synthpop sets")
}

if(file.exists(file.path(data_dir, paste0(county, ".zip")))){
  zip <- file.path(data_dir, paste0(county, ".zip"))
  unzip(zip, exdir = data_dir)
  #file.remove(zip)#Add this back when done testing
  county_dat <- file.path(data_dir, county)
} else {stop(paste0("Data file for ", county, " County not found at ", data_dir))
}


for (filename in list.files(county_dat)) {
  if(!(filename %in% c("hospitals.txt", "METADATA.txt") | str_ends(filename, ".parquet") )){
    table_name <- str_remove(filename, ".txt")
    assign(table_name, read_tsv(file.path(county_dat, filename)))
    rm(table_name, filename)}
}


# Data Wrangling ----------------------------------------------------------

#Build df contacts containing all contacts of each individual
contacts <- people %>%
  rename(PID = sp_id, hh_id = sp_hh_id, sch_id = school_id, wp_id = work_id) %>%
  mutate(gq_id = NA, gq = 0)

contacts <- gq_people %>%
  rename(PID = sp_id, gq_id =  sp_gq_id) %>%
  mutate(gq = 1,  hh_id = NA, sch_id = NA, wp_id = NA, relate = NA, race = NA) %>%
  bind_rows(contacts)

contacts$sch_id <- na_if(contacts$sch_id, "X")
contacts$wp_id <- na_if(contacts$wp_id, "X")

contacts_dt <- as.data.table(contacts)

sch_members_dt <- contacts_dt[
  !is.na(sch_id),
  .(sch_members = list(PID)),
  by = sch_id
]

wp_members_dt <- contacts_dt[
  !is.na(wp_id),
  .(wp_members = list(PID)),
  by = wp_id
]

gq_members_dt <- contacts_dt[
  gq == 1,
  .(gq_members = list(PID)),
  by = gq_id
]

hh_members_dt <- contacts_dt[
  !is.na(hh_id),
  .(hh_members = list(PID)),
  by = hh_id
]

contacts_dt <- hh_members_dt[contacts_dt, on = "hh_id"]
contacts_dt <- gq_members_dt[contacts_dt, on = "gq_id"]
contacts_dt <- wp_members_dt[contacts_dt, on = "wp_id"]
contacts_dt <- sch_members_dt[contacts_dt, on = "sch_id"]



#Build a DF of locations
school_locations <- sch_members_dt[, .(
  id = as.character(sch_id),
  members = sch_members,
  type = "School"
)]
work_locations <- wp_members_dt[, .(
  id = as.character(wp_id),
  members = wp_members,
  type = "Workplace"
)]
gq_locations <- gq_members_dt[, .(
  id = as.character(gq_id),
  members = gq_members,
  type = "Group Quarters"
)]
hh_locations <- hh_members_dt[, .(
  id = as.character(hh_id),
  members = hh_members,
  type = "Household"
)]
locations_dt <- rbindlist(
  list(school_locations, work_locations, gq_locations, hh_locations),
  use.names = TRUE, fill = TRUE
)

locations_dt[, size := lengths(members)]


print("Data Wrangled!")
if (save_files) {
print("Saving to parquet")
write_parquet(contacts_dt, file.path(county_dat, "contacts.parquet"))
write_parquet(locations_dt, file.path(county_dat, "locations.parquet"))
print(paste0("Contact and Location Files Saved to ", county_dat))
}

#locations contains a list of each location, its members' PIDs, the locationt type, and size
#contacts contains each individual (PID), their corresponding gq, hh, wp, and/or sch
#age/sex/race of individual, and the PIDs of each of their contacts at each corresponding location 
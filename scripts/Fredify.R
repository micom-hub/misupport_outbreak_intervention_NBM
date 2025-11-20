#Input: zip file downloaded from FRED synthetic population repo named {county}.zip
#Output: DFs containing necessary information for contact networks
#Takes approximately 1 minute per 1,000,000 population for a given county

if(!require(tidyverse)){install.packages("tidyverse")}
if(!require(jsonlite)){install.packages("jsonlite")}


county <- "Washtenaw"
saveJSON <- TRUE

# Check Files, Unzip FRED file, and set up workspace -------------------------------------------


data_dir <- file.path(getwd(), "data")
if(!dir.exists(data_dir)){
  stop("Attempted to handle data before downloading FRED synthpop sets")}

if(file.exists(file.path(data_dir, paste0(county,".zip")))){
  zip <- file.path(data_dir, paste0(county,".zip"))
  unzip(zip, exdir = data_dir)
  file.remove(zip)#Add this back when done testing
  county_dat <- file.path(data_dir, county)
} else {stop(paste0("Data file for ", county, " County not found at ", data_dir))}


for(filename in list.files(county_dat)){
  if(!(filename %in% c("hospitals.txt", "METADATA.txt"))){
    table_name <- str_remove(filename, ".txt")
    assign(table_name, read_tsv(file.path(county_dat,filename)))
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

school_list <- unique(contacts$sch_id)

contacts$sch_id <- na_if(contacts$sch_id, "X")

sch_members <- contacts %>%
  filter(!is.na(sch_id)) %>%
  group_by(sch_id) %>%
  nest() %>%
  mutate(sch_members = map(.x = data, .f = ~.x %>% pull(PID))) %>%
  select(sch_id, sch_members)

contacts$wp_id <- na_if(contacts$wp_id, "X")
wp_members <- contacts %>%
  filter(!is.na(wp_id)) %>%
  group_by(wp_id) %>%
  nest() %>%
  mutate(wp_members = map(.x = data, .f = ~.x %>% pull(PID))) %>%
  select(wp_id, wp_members)

gq_members <- contacts %>%
  filter(gq == 1) %>%
  group_by(gq_id) %>%
  nest() %>%
  mutate(gq_members = map(.x = data, .f = ~.x %>% pull(PID))) %>%
  select(gq_id, gq_members)

hh_members <- contacts %>%
  group_by(hh_id) %>%
  filter(!is.na(hh_id)) %>%
  nest() %>%
  mutate(hh_members = map(.x = data, .f = ~.x %>% pull(PID))) %>%
  select(hh_id, hh_members)


contacts <- contacts %>%
  full_join(gq_members, by = join_by("gq_id" == "gq_id"), relationship = "many-to-one") %>%
  full_join(wp_members, by = join_by("wp_id" == "wp_id"), relationship = "many-to-one") %>%
  full_join(sch_members, by = join_by("sch_id" == "sch_id"), relationship = "many-to-one") %>%
  full_join(hh_members, by = join_by("hh_id" == "hh_id"), relationship = "many-to-one") 
  
contacts <- contacts %>%
  mutate(across(PID:race, as.character),
         age = as.numeric(age))


#Build a DF of locations
locations <- sch_members %>%
  mutate(type = "School") %>%
  rename(members = sch_members, id = sch_id)
  
locations <- wp_members %>%
  rename(members = wp_members, id = wp_id) %>%
  mutate(type = "Workplace") %>%
  bind_rows(locations)

locations <- gq_members %>%
  rename(members = gq_members, id = gq_id) %>%
  mutate(type = "Group Quarters", id = as.character(id)) %>%
  bind_rows(locations)

locations <- hh_members %>%
  rename(members = hh_members, id = hh_id) %>%
  mutate(type = "Household", id = as.character(id)) %>%
  bind_rows(locations)

locations["size"] <- lengths(locations$members)

locations  %>%
  group_by(type) %>%
  summarize(avg = mean(size), min = min(size), max = max(size))

if(saveJSON){
write_json(toJSON(contacts), file.path(county_dat, "contacts.json"))
write_json(toJSON(locations), file.path(county_dat, "locations.json"))
}



#locations contains a list of each location, its members' PIDs, the locationt type, and size
#contacts contains each individual (PID), their corresponding gq, hh, wp, and/or sch, 
#age/sex/race of individual, and the PIDs of each of their contacts at each corresponding location 




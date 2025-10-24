# Measles Modeling for Resource Allocation 

Author: Cameron Hempton

Created: 24/10/2025


### About
In an outbreak response, local health departments apply their resources to interrupt disease transmission chains. The size of these contact chains scales exponentially, making it intractable to reach out to all contacts in the case of a highly-contagious disease like measles.

Status-quo is to prioritize reaching out to exposured individuals chronologically as information comes in. However, this does not account for the notion that *"not all contacts are created equally"*, with some contacts being more likely to transmit infection than others. 

**This modeling project aims to:**
* Produce a measles outbreak simulation with various "exposure circumstances" and LHDs contacting exposed individuals to prompt behaviors that (hopefully) reduce transmission.
* Simulate a variety of candidate LHD calling protocols to assess how prioritization of cases handled  influences average/maximum outbreak size.

These objectives will contribute to the development of a risk stratification system for LHD call-lists based on known parameters surrounding the exposure so that cointact-tracing resources can be more efficiently allocated, more quickly reaching individuals that are more likely to be infected and therefore infectious.
# Network Outbreak Modeling for Local Health Department Resource Allocation Optimization

Author: Cameron Hempton

Created: 24/10/2025


### About
In an outbreak response, local health departments (LHDs) apply their resources to interrupt disease transmission chains. The size of these contact chains scales exponentially, making it intractable to reach out to all contacts in the case of a highly-contagious disease like measles.

Status-quo is to prioritize reaching out to exposured individuals chronologically as information comes in. However, this does not account for the notion that *"not all contacts are created equally"*, with some contacts being more likely to transmit infection than others. 

**This modeling project aims to:**
* Produce an outbreak simulation with various exposure circumstances and LHDs contacting exposed individuals to prompt behaviors that (hopefully) reduce transmission.
* Simulate a variety of candidate LHD calling protocols to assess how prioritization of cases handled influences average/maximum outbreak size.

Findings from these simulations will be used to design an optimized LHD calling algorithm to efficiently allocate limited employee resources towards maximally reducing outbreak size, outbreak rapidity, or both.

**Model Assumptions**
* Reinfection is not possible
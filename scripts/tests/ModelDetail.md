---
title: LHDsim Model Details
author: Cameron Hempton
date: June 12, 2026
---

# LHDsim: A Modular Outbreak Simulation Framework

This document provides a detailed overview of the LHDsim (Local Health Department Simulator) model, explaining the interactions between its core components: the outbreak simulation model and the Local Health Department (LHD) intervention system. It also describes the various algorithms and policies that guide the LHD's decision-making.

## 1. Core Components Overview

The LHDsim framework consists of two primary interacting systems:

*   **Outbreak Model (`scripts/simulation/outbreak_model.py`)**: This component simulates the spread of an infectious disease on a contact network, tracking individuals through different epidemiological states (Susceptible, Exposed, Infectious, Recovered).
*   **Local Health Department (LHD) System (`scripts/lhd/`)**: This component represents the LHD, which observes the simulated outbreak (through surveillance), builds a partial understanding of the contact network, and implements various interventions (like isolation, testing, and contact tracing) to mitigate the spread.

These two systems interact daily, with the outbreak model advancing the disease state and the LHD reacting to reported cases and applying interventions that influence the disease dynamics in subsequent steps.

## 2. The Outbreak Model (`scripts/simulation/outbreak_model.py`)

The `NetworkModel` class is the heart of the outbreak simulation. It models disease progression using a Susceptible-Exposed-Infectious-Recovered (SEIR) framework on a contact network.

### Key Aspects:

*   **SEIR States**: Individuals in the simulation can be in one of four states:
    *   **S (Susceptible)**: Can become infected.
    *   **E (Exposed/Pre-infectious)**: Infected but not yet infectious.
    *   **I (Infectious)**: Can transmit the disease to others.
    *   **R (Recovered)**: Immune and no longer infectious (may wane over time).
*   **Disease Progression**:
    *   **S -> E**: Occurs through transmission events from infectious individuals. The probability of transmission depends on `base_transmission_prob`, contact weights, and individual factors like vaccination status and age.
    *   **E -> I**: After an `incubation_period`, exposed individuals become infectious.
    *   **I -> R**: After an `infectious_period`, infectious individuals recover.
    *   **R -> S (Waning Immunity)**: If `conferred_immunity_duration` is set, recovered individuals can lose immunity and become susceptible again after a certain period.
*   **Contact Network**: The model operates on a network of individuals (`N` nodes) with defined contact types (e.g., household, workplace, school). Transmission occurs along these edges.
*   **Multipliers**: The LHD can influence transmission by modifying `in_multiplier` and `out_multiplier` arrays for individuals and contact types. These multipliers reduce the effective contact weight, thereby lowering transmission probabilities.
*   **Randomness**: The model is stochastic, meaning outcomes can vary even with identical initial conditions due to random processes (e.g., transmission events, incubation/infectious periods). Seeds are used to ensure reproducibility of specific runs.
*   **`step()` Method**: This method advances the simulation by one day, performing all SEIR transitions and determining new transmission events. It also generates an `epi_state` dictionary containing information about newly exposed and infectious individuals, which is passed to the LHD.
*   **`simulate()` Method**: Orchestrates the entire simulation for a specified number of replicates and duration, calling `step()` daily and interacting with the LHD.

## 3. The Local Health Department (LHD) System (`scripts/lhd/`)

The LHD system is responsible for observing the outbreak, building a knowledge base, and executing interventions. It's structured into several sub-components:

### 3.1. `lhd.py` (LocalHealthDepartment Class)

The `LocalHealthDepartment` class is the central orchestrator of LHD activities.

*   **Initialization (`__init__`)**: Sets up the LHD with a reference to the `NetworkModel`, defines daily capacity, and initializes `SurveillanceModel`, `LHDState`, policy algorithms, and an `Executor`.
*   **`step(t, epi_state)` Method**: This is the LHD's daily cycle:
    1.  **Process Expirations**: Reverts any interventions (e.g., isolation) that have reached their duration limit.
    2.  **Observe**: Uses the `SurveillanceModel` to process the true epidemiological state (`epi_state`) and any scheduled actions (like tests or traces), generating a `batch` of new observations (e.g., reported cases, traced contacts).
    3.  **Integrate Findings**: Updates its internal `LHDState` (knowledge base) with the new `batch` of observations.
    4.  **Propose Actions**: Policy algorithms (e.g., `IsolateNewCases`) generate `ActionProposal` objects based on the current `LHDState`.
    5.  **Plan Actions**: The `GreedyPlanner` selects a subset of these proposals to execute within the allocated `lhd_daily_capacity`.
    6.  **Execute Actions**: The `Executor` applies the selected actions to the `NetworkModel` (e.g., modifying multipliers for isolation, scheduling tests/traces in surveillance).
    7.  **Log**: Records daily activities and outcomes for analysis.
*   **Intervention Methods (`_apply_isolation`, `_order_trace`, `_order_test`)**: These methods are called by the `Executor` to directly implement interventions:
    *   `_apply_isolation`: Reduces `in_multiplier` and `out_multiplier` for specified nodes and contact types for a given duration.
    *   `_order_trace`: Schedules contact tracing for specified cases in the `SurveillanceModel`.
    *   `_order_test`: Schedules testing for specified nodes in the `SurveillanceModel`.

### 3.2. `surveillance.py` (SurveillanceModel Class)

The `SurveillanceModel` simulates how the LHD gains information about the outbreak, which is always imperfect and delayed.

*   **Baseline Detection**: When individuals transition from Exposed to Infectious, there's a `p_detect_inf` probability of them being detected and reported, subject to a `report_delay_days`.
*   **Case Status**: Tracks whether an individual has been `never detected`, `queued to report`, or `reported`.
*   **`order_trace(cases, params)`**: Adds cases to a queue for contact tracing. When processed, it identifies contacts from the true `neighbor_map` based on `recall_prob` and `max_per_case` limits.
*   **`order_test(nodes, params)`**: Adds nodes to a queue for testing. When processed, it simulates a test result based on the true epidemiological state (`epi_state`) and test parameters (`sens_pre`, `sens_inf`, `spec`). Positive tests are then queued as case reports.
*   **`step(t, epi_state)`**: Processes baseline detection, executes due test orders (`_process_due_tests`), executes due trace orders (`_process_due_traces`), and delivers any case reports that are due (`_deliver_due`). It returns a `batch` dictionary containing all new observations for the LHDState.

### 3.3. `state.py` (LHDState Class)

The `LHDState` acts as the LHD's internal "brain" or knowledge base, storing all information gathered through surveillance. This information is always partial and potentially outdated compared to the true `NetworkModel` state.

*   **Known Cases**: Tracks which individuals are `known_case`s, their `case_report_time`, `case_stage` (e.g., infectious), and known attributes like age and vaccination status. `new_cases_today` tracks recently reported cases.
*   **Known Contact Structure**: Stores `known_edges` (edges discovered through tracing) and `known_adj` (an adjacency list representing the LHD's current understanding of the network). `new_edges_today` tracks recently discovered edges.
*   **Intervention Tracking**: Keeps track of `isolated_until` and `quarantined_until` for each node, and `pending_tests` or `pending_traces` that have been ordered but not yet processed by surveillance.
*   **`process_batch(batch)`**: Integrates the daily `batch` of surveillance observations into the LHD's knowledge base, updating known cases and edges.

### 3.4. `algorithms_state.py` (Policy Algorithms)

Policy algorithms are classes that implement the `propose()` method. They analyze the current `LHDState` and generate a list of `ActionProposal` objects, suggesting specific interventions for specific individuals. Each proposal has an `action` type, a `target`, a `priority`, and a `cost_units`.

*   **`IsolateNewCases`**: Proposes to `isolate` any individuals who are `new_cases_today`.
*   **`TraceNewCases`**: Proposes to `trace` any individuals who are `new_cases_today`.
*   **`TestContactsOfKnownCases`**: Proposes to `test` individuals who are known contacts of `known_case_list` members, provided they are not already known cases or pending tests.
*   **`TraceEdgeEndpoints`**: Proposes to `trace` the endpoints (nodes) of any `new_edges_today` that were discovered through previous tracing efforts. This helps "crawl" the network.
*   **`IsolateNeighborsOfHighDegreeCases`**: Identifies `new_cases_today`, ranks them by their known degree (number of contacts), and proposes to `isolate` all known contacts of these high-degree cases.

### 3.5. `planner.py` (GreedyPlanner Class)

The `GreedyPlanner` is responsible for selecting which `ActionProposal`s to execute, given the LHD's daily resource `capacity`.

*   **`select(proposals, capacity)`**:
    1.  Deduplicates proposals (e.g., if multiple algorithms propose the same action for the same person, it keeps the one with the highest priority or lowest cost).
    2.  Sorts the unique proposals by a scoring function: `priority / cost_units` (higher score first).
    3.  Iterates through the sorted proposals, adding them to the `selected` list until the `capacity` is reached.
    4.  Returns an `ActionPlan` containing the `selected` proposals and resource usage.

### 3.6. `executor.py` (Executor Class)

The `Executor` takes the `ActionPlan` from the `Planner` and translates it into concrete actions on the `NetworkModel` and `SurveillanceModel`.

*   **`execute(lhd, t, plan)`**:
    1.  Groups selected `ActionProposal`s by action type and parameters.
    2.  For each group, it calls the appropriate LHD method:
        *   `isolate` actions call `lhd._apply_isolation()`.
        *   `trace` actions call `lhd._order_trace()`.
        *   `test` actions call `lhd._order_test()`.
    3.  Records how many actions were attempted and applied in an `ExecutionSummary`.

### 3.7. `tokens.py` (MultiplierToken Class)

`MultiplierToken` objects are used to track reversible interventions, specifically those that modify the `in_multiplier` or `out_multiplier` arrays in the `NetworkModel`.

*   **Purpose**: When an action like isolation is applied, a `MultiplierToken` is created, storing the nodes, contact types, and the factor by which multipliers were changed, along with an `expires_at` timestamp.
*   **`revert(model)`**: When the token expires, this method is called to undo the change, restoring the original multiplier values for the affected nodes and contact types.

### 3.8. `response_types.py`

This file defines data classes used for communication within the LHD system:

*   **`ActionProposal`**: Represents a suggestion for an intervention, including the action type, target, priority, cost, and parameters.
*   **`ActionPlan`**: Contains the list of `ActionProposal`s selected by the `Planner` for execution, along with capacity information.
*   **`ExecutionSummary`**: Summarizes the outcomes of the `Executor`'s actions, including counts of attempted, applied, and information-gathering orders.

### 3.9. `policy_catalog.py` (Policies)

The `policy_catalog.py` defines various LHD policies, each combining a set of algorithms and a `GreedyPlanner`. A policy dictates *what* the LHD prioritizes and *how* it allocates its resources.

*   **`observe_only`**:
    *   **Algorithms**: None.
    *   **Description**: The LHD simply observes the outbreak without taking any active interventions.
*   **`isolate_only`**:
    *   **Algorithms**: `IsolateNewCases`.
    *   **Description**: The LHD only isolates individuals who are newly reported as cases.
*   **`trace_only`**:
    *   **Algorithms**: `TraceNewCases`.
    *   **Description**: The LHD only performs contact tracing on individuals who are newly reported as cases.
*   **`trace_then_isolate`**:
    *   **Algorithms**: `TraceNewCases`, `IsolateNewCases`.
    *   **Description**: The LHD traces newly reported cases and isolates newly reported cases.
*   **`trace_and_test`**:
    *   **Algorithms**: `IsolateNewCases` (highest priority), `TraceNewCases`, `TestContactsOfKnownCases`.
    *   **Description**: The LHD prioritizes isolating new cases, then tracing new cases, and finally testing contacts of known cases.
*   **`network_crawl`**:
    *   **Algorithms**: `TraceNewCases`, `TraceEdgeEndpoints`.
    *   **Description**: The LHD traces newly reported cases and also traces the endpoints of any new contact edges discovered through previous tracing efforts, effectively "crawling" the network to expand its knowledge.
*   **`network_crawl_isolate`**:
    *   **Algorithms**: `TraceEdgeEndpoints`, `IsolateNewCases` (highest priority), `IsolateNeighborsOfHighDegreeCases`.
    *   **Description**: The LHD prioritizes isolating new cases, then isolating neighbors of high-degree new cases, and also traces newly discovered edge endpoints to expand its knowledge.

## 4. Interaction Flow: A Day in the Simulation

Here's a summary of how the `NetworkModel` and `LocalHealthDepartment` interact on a given day `t`:

1.  **NetworkModel.simulate()**: Calls `NetworkModel.step()`.
2.  **NetworkModel.step()**:
    *   Advances SEIR states (S->E, E->I, I->R, R->S).
    *   Determines new transmission events, creating `newly_exposed` individuals.
    *   Compiles an `epi_state` dictionary (containing `new_inf_ids`, `new_pre_ids`, `inf_ids`, `pre_ids`).
    *   Calls `self.lhd.step(t=t, epi_state=epi_state)`.
3.  **LHD.step(t, epi_state)**:
    *   **Processes Expiring Interventions**: `lhd.process_expirations(t)` reverts `MultiplierToken`s.
    *   **Observes**: `lhd.surveillance.step(t, epi_state)` processes the true `epi_state` to generate a `batch` of observations (e.g., new reported cases, results from tests/traces ordered on previous days).
    *   **Updates Knowledge**: `lhd.state.process_batch(batch)` updates the LHD's internal `LHDState` with the new observations.
    *   **Proposes Actions**: Each algorithm in the LHD's policy calls its `propose(lhd.state)` method, generating `ActionProposal`s.
    *   **Plans Actions**: `lhd.planner.select(proposals, lhd.daily_capacity)` chooses which proposals to execute.
    *   **Executes Actions**: `lhd.executor.execute(lhd, t, plan)` calls the appropriate `lhd._apply_isolation()`, `lhd._order_trace()`, or `lhd._order_test()` methods.
        *   `_apply_isolation()` directly modifies `model.in_multiplier` and `model.out_multiplier` and schedules a `MultiplierToken` for future reversion.
        *   `_order_trace()` and `_order_test()` add orders to the `lhd.surveillance` queues to be processed on future days.
    *   **Logs**: `lhd._log_day()` records daily activities.
4.  **NetworkModel.step() (continues)**: Returns control to the `NetworkModel`, which then continues its simulation loop. The changes made by the LHD (e.g., reduced multipliers) will affect transmission probabilities in the *next* `NetworkModel.step()`.

This cycle repeats for the entire simulation duration, allowing for dynamic interaction between disease spread and public health interventions.

## 5. Configuration (`scripts/config.py`)

The `ModelConfig` class in `scripts/config.py` centralizes all parameters for the simulation. It's structured into nested dataclasses:

*   **`EpiParams`**: Epidemiological parameters (e.g., `base_transmission_prob`, `incubation_period`, `vax_uptake`).
*   **`PopulationParams`**: Parameters related to the contact network structure (e.g., `wp_contacts`, `hh_weight`).
*   **`LHDParams`**: LHD-specific parameters (e.g., `lhd_daily_capacity`, `p_detect_inf`, `policy_name`, `trace_recall_prob`).
*   **`SimulationParams`**: General simulation settings (e.g., `n_replicates`, `simulation_duration`, `seed`, `county`).

This configuration system allows for easy modification and management of simulation parameters, including the ability to `copy_with` overrides for running parameter sweeps and variant analyses.

---

This detailed explanation should provide a comprehensive understanding of how the LHDsim model operates and how its various components interact to simulate disease outbreaks and public health interventions.

```
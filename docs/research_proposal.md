# Research Proposal — Occupant-Centric HVAC Optimization Using Preference Learning in Smart Buildings

## 1. Introduction

HVAC systems represent one of the largest energy loads in buildings. Traditional controllers rely on static setpoints and generalized comfort models (e.g., PMV), which fail to capture individual, time-varying thermal preferences shaped by occupancy behavior, activity level, clothing, physiology, and personal taste. This leads to energy waste, discomfort, and poor demand-side flexibility.

This project proposes an occupant-centric HVAC optimization framework combining **preference learning**, **adaptive thermal comfort modeling**, **reinforcement learning**, and **multi-objective optimization** to balance comfort, energy efficiency, cost, and sustainability.

## 2. Motivation

- **Static comfort assumptions** (PMV/PPD, fixed setpoints) ignore individual and temporal variation.
- **Energy inefficiency** from conservative comfort ranges and unnecessary cycling.
- **Underuse of smart building data** (IoT sensors, wearables, smart thermostats) by conventional controllers.

## 3. Problem Statement

Existing HVAC systems cannot adapt to dynamic occupant preferences, uncertain occupancy, and varying conditions without predefined thermal models. This research develops a preference-learning-based framework to infer comfort automatically, learn personalized preferences, and optimize control in real time.

## 4. Research Objectives

1. Develop a dynamic occupant thermal preference model from behavioral/environmental data.
2. Design a preference learning mechanism that infers comfort without explicit manual feedback.
3. Build a reinforcement-learning-based HVAC controller for adaptive thermal management.
4. Optimize HVAC operation across energy consumption, comfort, electricity pricing, and carbon emissions.
5. Benchmark against conventional, rule-based, and PMV-based control.

## 5. Research Questions

- How can thermal preferences be learned automatically from behavioral data?
- Can preference learning improve energy efficiency without sacrificing comfort?
- How does RL adapt HVAC control under uncertain occupancy?
- What is the optimal comfort/energy/cost trade-off?
- How does the proposed framework compare to traditional optimization methods?

## 6. Related Work (Summary)

- **HVAC optimization**: rule-based control, MPC, fuzzy control, evolutionary methods, RL. MPC is strong but needs accurate models and heavy computation.
- **Thermal comfort modeling**: PMV/PPD and adaptive comfort models assume an "average" occupant and struggle under dynamic conditions.
- **Preference learning**: infers preferences from observed behavior (thermostat adjustments, window use, fan use, occupancy, wearables) rather than explicit labels.
- **RL for HVAC**: DQN, PPO, SAC, DDPG offer model-free, adaptive, real-time control, with challenges around exploration safety, convergence, and sparse rewards.

## 7. Proposed Methodology

### 7.1 System Architecture (4 layers)

| Layer | Function |
|---|---|
| Data Acquisition | Collect sensor and occupant data |
| Preference Learning | Infer occupant comfort preferences |
| Optimization | Perform HVAC optimization |
| Control | Execute HVAC control actions |

### 7.2 Data

- **Environmental**: indoor/outdoor temperature, humidity, CO₂, air velocity, weather.
- **Occupant**: occupancy detection, thermostat interactions, window behavior, wearables, activity level.

### 7.3 Preference Learning Model

Candidates: Bayesian preference learning, inverse reinforcement learning, neural preference learning, contextual bandits. The implemented approach (see `docs/pipeline.html`) uses **Bayesian inference** to estimate a posterior over individual preferred temperature/humidity (T*, RH*) from behavioral likelihoods and a population-level prior.

### 7.4 RL HVAC Controller

Formulated as an MDP:

- **State**: indoor temperature/humidity, occupancy, electricity price, HVAC status, learned preference.
- **Action**: setpoint adjustment, airflow control, fan speed, compressor scheduling.
- **Reward**: weighted combination of personalized comfort score and energy penalty, e.g. `R = α·Comfort − β·Energy` (weights tunable; extended with action-smoothness and safety terms in the implemented pipeline).

### 7.5–7.7 Discomfort Modeling, Thermal Dynamics, Optimization Objective

Formulated as a multi-objective optimization problem subject to comfort and HVAC operational constraints (see implementation in `src/reward/` and `src/env_model/`).

## 8. Candidate Algorithms

| Function | Candidate Algorithms |
|---|---|
| Preference Learning | Bayesian Learning, IRL, Neural Networks |
| HVAC Optimization | PPO, SAC, DDPG, DQN |
| Prediction | LSTM, Transformer |
| Occupancy Detection | CNN, Random Forest |

## 9. Simulation & Experimental Setup

- **Tools**: Python, EnergyPlus (or data-driven surrogate), OpenAI Gym/Gymnasium, TensorFlow/PyTorch.
- **Datasets**: ASHRAE Building Dataset, Building Data Genome Project, Pecan Street Dataset, EnergyPlus simulated data, CAMaRSEC/AP1_BR, Chinese Thermal Comfort Dataset.
- **Scenarios**: varying occupancy schedules, dynamic pricing, extreme weather, multiple preference profiles.

## 10. Evaluation Metrics

| Metric | Description |
|---|---|
| Energy Consumption | HVAC electricity usage |
| Comfort Satisfaction | Occupant comfort level |
| Peak Demand Reduction | Demand-side flexibility |
| Cost Savings | Electricity bill reduction |
| Carbon Emission Reduction | Sustainability impact |
| Learning Convergence | RL training stability |

## 11. Expected Contributions

- A novel occupant-centric HVAC optimization framework using preference learning.
- A dynamic, personalized thermal comfort model.
- Integration of preference learning with reinforcement learning for HVAC control.
- Improved comfort/energy/cost balance.
- A scalable control architecture for smart buildings.

## 12. Expected Outcomes

Reduced HVAC energy consumption, improved comfort satisfaction, adaptation to dynamic user behavior, outperformance of PMV-based controllers, and enhanced building sustainability.

## 13. Significance

Contributes to smart building intelligence, sustainable energy management, human-centric automation, and AI-driven building control — aligned with smart city and net-zero building initiatives.

## 14. Future Extensions

Federated preference learning, multi-agent HVAC coordination, digital twin integration, safe RL, emotion-aware HVAC control, integration with renewable energy systems.

## 15. Conclusion

This research proposes an occupant-centric HVAC framework combining preference learning and reinforcement learning to dynamically learn preferences and adapt control in real time, aiming for improved comfort, reduced energy consumption, lower cost, and enhanced sustainability.

# Neural Network Autopilot

## Overview
This project involves implementing a neural network trained to act as an **autopilot** for a game in which a plane navigates through a cave.  

The goal is to:
- Extract optimised features recorded from the game.
- Save these features to a CSV file.
- Train a model to play the game autonomously.  

**Target outcomes:**
- Sub 1-minute training time.
- Survival of at least 30 seconds.

Key considerations:
- How to extract features from the game.
- What features are best to extract for training a strong model.
- Neural network design (neurons, hidden layers, regression vs. classification, etc.).
- How to train the model with the collected data.

---

## Feature Engineering
For this project, **27 features** were identified for each timestep that accurately describe the game:

- **Top and bottom edge × 5 columns**  
  Extracts the top and bottom of the cave for the 5 columns in front of the player.  

- **Cave centre × 5 columns**  
  Extracts the cave’s centre by dividing the area between the top and bottom by 2.  

- **Player’s current row (Y position, `playerRow`)**  
  Shows where the player is relative to the cave features.  

- **Last move made (`lastMove`)**  
  Represents the action taken in response to the cave state.  
  This is used as the label for the AI.  

This structure provides both **spatial context** (what is ahead of the player) and **action context** (how the player responded), while keeping feature count minimal for faster learning.

---

## Data Collection
- Data collection is toggled with the **D key**.  
- Each feature listed above is recorded at each game timestep.  
- Data is extracted from the game’s internal **1s and 0s model** (raw state).  
- Optimised features are derived from this model:
  - **Top/bottom of cave:** found by locating first and last `0` values.  
  - **Player column:** extracted directly during gameplay.  
  - **Last move:** updated each step.  
  - **Cave centre:** computed by `(top + bottom) / 2`.  

---

## Neural Network Design

### Type of Task
- **Classification problem** → 3 possible output classes.

### Design Reasoning
- **27 input neurons** → corresponds to extracted features.  
- **Hidden layers** → allow the network to capture patterns.  
- **Layer structure**:  
  - First layer: larger for more learning capacity.  
  - Second & third layers: progressively smaller for efficiency.  
- **Activation functions**:  
  - `TANH` for hidden layers (handles positive & negative values, good for movement).  
  - `SoftMax` for output layer (converts outputs into probabilities for the 3 moves).

### Training Setup
- **Algorithm:** ResilientPropagation  
  - Fast and reliable.  
  - No manual tuning required.  
  - Adjusts automatically based on error gradient → faster convergence.  
- **Data balancing:**  
  - The raw dataset was **70% “stay” moves**, creating a bias.  
  - To fix: counts of `up`, `stay`, and `down` moves were balanced by downsizing the larger sets to the smallest set size.  
  - Result: reduced total data, but improved fairness in training.

---

## Extras
Collecting quality data manually was tedious and noisy.  
To solve this:
- A **deterministic rules-based bot** was created to play the game.  
- This bot only moves when strictly necessary.  
- Result: cleaner, more optimal training data.

---

## Conclusion
The model proved to be highly effective:  

- Out of **20 runs**, it survived **30+ seconds in 15 runs**.  
- Best case: **500 seconds survival** (manually stopped).  

**Weakness:**  
- The model struggles when the cave entrance is at extreme positions (very high/low).  
- It often reacts too late to reach the entrance.  
- While programmatic fixes exist, they were considered **outside the AI’s learning scope**.  

Overall, the model demonstrates **strong autonomous performance** and successfully achieves the project’s goals.  

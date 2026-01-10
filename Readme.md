## STEP-BY-STEP: What to Do Now

### Step 1: Install Required Packages (2 minutes)

Open your terminal and run:

```bash
pip install gymnasium torch numpy matplotlib
```

**If you get errors**, try:
```bash
pip install --upgrade pip
pip install gymnasium torch numpy matplotlib
```

---

### Step 2: Test Your Setup (Optional - 1 minute)

Run the DQN demo to see if everything works:

```bash
python cartpole_dqn_fixed.py
```

**What you'll see:** A pre-trained DQN agent playing CartPole (should get ~500 score)

**If it fails:** You might need TensorFlow:
```bash
pip install tensorflow
```

**Note:** This is just a demo. Your project needs A2C, not DQN!

---

### Step 3: Run Agent 0 - YOUR ACTUAL PROJECT STARTS HERE (5-10 minutes)

This is the COMPLETE implementation of Agent 0 that you can run right now:

```bash
python agent0_complete.py
```

**What this does:**
- Trains an A2C agent (K=1, n=1) for 100,000 steps
- Evaluates every 20,000 steps
- Shows training progress in terminal
- Creates plots showing:
  - Episode returns during training
  - Evaluation returns over time
  - Actor and critic losses
  - Value function trajectory
- Saves a plot: `agent0_results.png`

**Expected output:**
```
Training Agent 0 (seed=42)
Max steps: 100000, Eval every: 20000 steps
------------------------------------------------------------
Step   1000 | Episode Return: 23 | Avg(10): 25.40 | ...
Step   2000 | Episode Return: 45 | Avg(10): 38.20 | ...
...
============================================================
EVALUATION at step 20000
Mean Return: 125.50 ± 45.23
Mean Value: 120.34
============================================================
...
FINAL EVALUATION
Mean Return: 450.23 ± 75.12
Mean Value Function: 485.67
Expected value with correct bootstrapping: ~500
```

**How long will it take?**
- 100,000 steps: ~5-10 minutes (for testing)
- 500,000 steps: ~25-40 minutes (for full training)

---

### Step 4: Understand What You Just Ran (10-15 minutes)

1. **Open agent0_complete.py** in a code editor
2. **Read through the code** and the comments
3. **Look at the plot** (agent0_results.png)
4. **Compare with A2C_PROJECT_GUIDE.md**

**Key things to understand:**
- How the Actor-Critic network is structured
- How bootstrapping works (see `compute_return` function)
- How the training loop works
- What the losses mean

---

### Step 5: Experiment with Correct vs Incorrect Bootstrapping (15 minutes)

The project asks you to compare correct vs incorrect bootstrapping!

**Experiment 1: Correct Bootstrapping (default)**
```bash
python agent0_complete.py
```
Note the final value function: Should be ~500

**Experiment 2: Incorrect Bootstrapping**

Edit `agent0_complete.py`, find the `compute_return` function (around line 125):

```python
# CHANGE THIS:
def compute_return(self, reward, next_state, terminated, truncated):
    if terminated and not truncated:
        return reward
    else:
        # Bootstrap
        ...

# TO THIS (WRONG - treats truncation as termination):
def compute_return(self, reward, next_state, terminated, truncated):
    if terminated or truncated:  # <-- WRONG!
        return reward  # No bootstrap
    else:
        # Bootstrap
        ...
```

Run again and compare:
```bash
python agent0_complete.py
```
Note the final value function: Will be much lower!

**Question to answer in your report:**
"Why is the value function different? Explain theoretically."

---

### Step 6: Now Build the Other Agents (This is your main work)

Now that you understand Agent 0, you need to implement:

- **Agent 1**: Add stochastic rewards (mask 90% of rewards)
- **Agent 2**: Add K=6 parallel workers
- **Agent 3**: Add n=6 step returns
- **Agent 4**: Combine K=6 and n=6

**How to approach this:**

1. **Copy agent0_complete.py** to agent1_stochastic.py
2. **Modify** the relevant parts (see guide)
3. **Run and compare** results
4. **Repeat** for agents 2, 3, 4

---

## File Purpose Summary

| File | Purpose | When to Use |
|------|---------|-------------|
| A2C_PROJECT_GUIDE.md | Theory & explanations | Read FIRST to understand concepts |
| agent0_complete.py | Working Agent 0 | Run SECOND to see A2C in action |
| a2c_starter_template.py | Empty template | Use if you want to code from scratch |
| cartpole_dqn_fixed.py | DQN demo | Optional - just to visualize CartPole |

---

## What to Submit (from project specs)

When you're done with ALL agents (0, 1, 2, 3, 4):

1. **Notebook (.ipynb)** with:
   - Code for all 5 agents
   - Plots comparing all agents (with error bars from 3 seeds!)
   - Analysis answering the questions in the PDF
   
2. **Python files** (.py):
   - Your agent implementations
   - Plotting/utility code
   
3. **Video (5 min max)**:
   - Walk through your code
   - Explain results
   - Discuss one interesting finding

4. **Live Q&A (8 min)**:
   - Be ready to explain ANY line of code
   - Answer theoretical questions

---

## Timeline Suggestion

You have until **Tuesday Jan 13, 9:00 AM**

- **Day 1 (Now)**: Install, run agent0_complete.py, understand the code
- **Day 2**: Implement Agent 1 (stochastic rewards)
- **Day 3**: Implement Agent 2 (K workers)
- **Day 4**: Implement Agent 3 (n-step returns)
- **Day 5**: Implement Agent 4 (combined), run all with 3 seeds
- **Day 6**: Create notebook, make plots, analyze results
- **Day 7**: Make video, final review, SUBMIT

---

## Debugging Tips

**If agent0_complete.py doesn't train well:**
- Check that returns are increasing (should reach ~500)
- Print value functions - should converge to ~500
- Check losses - should decrease then stabilize
- Try different seeds

**If you get errors:**
- "No module named 'gymnasium'" → `pip install gymnasium`
- "No module named 'torch'" → `pip install torch`
- Gym vs Gymnasium issue → Make sure you use `gymnasium` (not `gym`)

**If training is too slow:**
- Reduce max_steps to 100000 for testing
- Use 500000 for final runs with 3 seeds

---

## Questions? Next Steps?

1. Run `agent0_complete.py` first
2. Look at the output and plots
3. Read the code to understand how it works
4. Start modifying for Agent 1

Good luck! 🚀

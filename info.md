- **For structured projects → Use Python scripts.**
- **For exploration and EDA → Use Notebook.**


different parameters ?
 Temperature
 Frequency




different input methods ?
   Magnitude/phase angle of the real and imaginary resistance
   Array of arrays ( in re 1 re 2 re 60
Zim(ω1), Zim(ω2), ... Zim(ω60)]T are the real (Zre) and imaginary (Zim) parts of impedance spectra collected at 60 different frequencies (ωn, n = 1, 2, ..., 60))


 State vectors: We frame the problem as a regression task, and train a probabilistic machine learning model to learn the mapping Qn = f(sn, an), with uncertainty estimates, where sn is the battery state at the start of the nth cycle, an is the future action (the nth cycle charge/ discharge protocol), and Qn is the discharge capacity measured at the end of the cycle. The battery state vector sn is formed from the con- catenation of the real (Z re ) and imaginary (Z im ) components of the impedance measured at 57 frequencies, ω1, . . . , ω57, in the range 0.02Hz-20kHz; sn =1⁄2Zreðω1Þ,Zimðω1Þ,:::,Zreðω57Þ,Zimðω57Þ.




different algorithms:
 GPR
   (different)
 Gradient boosted decision trees (XGBoost)







Below is a **high-level explanation** of the information in your ASCII file (the “Technique”/“Sequence” table) and how it ties back to your DataFrame columns. I’ll then give **practical tips** for how to extract the charge/discharge currents needed for an “action vector,” especially given that you’re focusing on the EIS steps at `Ns = 1` or `Ns = 6`.

---

## 1. Understanding the ASCII “Technique”/“Sequence” Table

This portion of the file describes the **sequence of test steps** your instrument (BCS-815) performed. Each row labeled “Ns = 0, 1, 2, …” is a **step number**, and each column provides settings or conditions for that step. Some relevant columns:

- **Ns**  
  The step index. (For instance, `Ns = 1` might be “PEIS,” `Ns = 3` might be “CC” for constant-current, etc.)

- **ctrl_type**  
  The control type applied in that step, e.g. “PEIS,” “Rest,” “CC,” “CV,” “Loop,” etc.  
  - “PEIS” = Potentiostatic Electrochemical Impedance Spectroscopy  
  - “CC” = Constant Current  
  - “CV” = Constant Voltage  
  - “Rest” = zero current (open-circuit rest)  
  - “Loop” = a loop or repeated sequence.

- **charge/discharge**  
  Indicates whether that step is “Charge” or “Discharge” (or “Rest/EIS”).

- **ctrl1_val / ctrl2_val / ctrl3_val** (and the corresponding “unit” columns)  
  Numeric setpoints for frequency range (for EIS), current setpoints, voltage setpoints, etc.  
  - For a PEIS step, `ctrl2_val` might be the upper frequency (in kHz), and `ctrl3_val` might be the lower frequency (in mHz).  
  - For a CC step, `ctrl1_val` might be the current (in mA or A).  
  - For a CV step, `ctrl1_val` might be the voltage setpoint.

- **N** (the “C-rate multiplier” when “Apply I/C = C x N”)  
  If the battery is 4.0 Ah nominal, then setting `N = 2` could imply a **2C** rate (i.e., ~8 A). However, the file also shows “ctrl1_val = 100.0 mA” in some places, so the actual current might be constrained or set differently. Sometimes “N” is more of a multiplier for the test script logic, but the exact numeric current can come from `ctrl1_val`.

- **lim1_type**, **lim1_value**, etc.  
  These define stopping criteria or transitions (e.g., “stop if time > 10 s,” or “stop if voltage reaches 4.2 V”).

Put simply, **each row (Ns)** is a test step. “PEIS” steps correspond to EIS scans; “CC” might be a constant-current charge or discharge.

---

### Why We Care About `Ns = 1` or `Ns = 6`
You mentioned:

> “We only focus on when Ns State is 1 or 6 because that is the only point at which the team is conducting EIS testing.”

Indeed, from the table:

- **Ns=1** → `ctrl_type = PEIS` (EIS at the beginning?),  
- **Ns=6** → `ctrl_type = PEIS` again (EIS at another point in the sequence).

This aligns with your data: when your DataFrame has `Ns` = 1 or 6, you see columns like `freq/Hz`, `Re(Z)/Ohm`, `Im(Z)/Ohm`, etc. That is the EIS measurement.

But for building an **action vector** (the charge/discharge profile), you usually need to look at **the steps where the battery is actually charged or discharged**, e.g. `Ns=3` (CC, charge), `Ns=4` (CV, still charging), `Ns=8` (CC, discharge), or something along those lines.  

---

## 2. Mapping the ASCII Table to Your DataFrame

In your final Pandas DataFrame:

```
df.columns.values
array(['mode', 'ox/red', 'error', 'control changes', 'Ns changes',
       'counter inc.', 'Ns', 'I Range', 'time/s', 'control/V/mA',
       'Ecell/V', 'I/mA', 'dq/mA.h', '(Q-Qo)/mA.h', '|Energy|/W.h',
       'freq/Hz', '|Z|/Ohm', 'Phase(Z)/deg', 'Q charge/discharge/mA.h',
       'half cycle', 'Energy charge/W.h', 'Energy discharge/W.h',
       'Capacitance charge/µF', 'Capacitance discharge/µF', 'step time/s',
       'z cycle', 'Re(Z)/Ohm', 'Im(Z)/Ohm', 'Re(Y)/Ohm-1', 'Im(Y)/Ohm-1',
       '|Y|/Ohm-1', 'Phase(Y)/deg', 'x', 'Q discharge/mA.h',
       'Q charge/mA.h', 'Capacity/mA.h', 'Efficiency/%', 'control/V',
       'control/mA', 'cycle number', 'P/W', 'R/Ohm'],
      dtype=object)
```

- **`Ns`**: The step number from the ASCII file.  
- **`I/mA`**: The measured current (positive or negative).  
- **`Ecell/V`**: The measured voltage of the cell.  
- **`freq/Hz`, `Re(Z)/Ohm`, `Im(Z)/Ohm`**: The EIS frequency sweep data (only present in rows where `Ns` is for an EIS step).  
- **`cycle number`**: The cycle index that the software uses (it might increment each charge–discharge cycle).  

If you only keep rows where `Ns` = 1 or 6, you end up with the EIS data. That’s good for building your **state vector** (impedance). But for the “action vector,” which is “the concatenation of the nth cycle charge and discharge currents,” you typically need to **look at other `Ns` steps** (the actual charge or discharge steps) that occur *between* those EIS measurements.

---

## 3. Building the Action Vector from Your Data

When the paper says:

> “The action vector is formed from the concatenation of the nth cycle charge and discharge currents,”

it **usually** means something like:

\[
\mathbf{a}_n = \bigl[ I_\text{charge}(t_1), I_\text{charge}(t_2), \ldots , I_\text{discharge}(t_1), I_\text{discharge}(t_2), \ldots \bigr]
\]

- For a **simple** test (single constant-current charge, single constant-current discharge), that might reduce to just two scalars: `[I_charge, I_discharge]`.  
- For a **complex** test where current changes with time or you have multiple steps (CC + CV), you either store the entire time-profile or use summary stats (e.g., average charge current, average discharge current, total coulombs, etc.).  

### A) If Your Test Script is “CC Charge + CC Discharge”
1. Identify rows where “charge” happens. That might be `Ns=3` and `Ns=4` if it’s CC then CV (still net charging).  
2. Identify rows where “discharge” happens, maybe `Ns=8`.  
3. From each step, look at the measured column `I/mA`.  
   - Typically, if charging is *actually* +100 mA or +200 mA, you’ll see that in the data.  
   - For a single-step test, you could just do `mean_current_charge = df_charge[“I/mA”].mean()`; similarly for discharge.

### B) If Your Script Has a More Dynamic Profile
- Then you might store the entire current vs. time trace for the nth cycle (flattened).  
- Or store summary features (average charge current, total charge time, average discharge current, etc.).

---

### How to Do It in Practice

**Step 1**: Don’t filter out `Ns` except for the EIS portion if you want the charge/discharge data. Instead, keep all rows (or at least the rows for the relevant cycles) so you can see the current during CC or CV steps.  

**Step 2**: Group by `cycle number`, then within that group, split by “charge” vs. “discharge” or by `Ns` values that correspond to actual charging or discharging.  

**Step 3**: Compute how you want to represent the action. For example, a simple approach:

```python
action_vectors_by_cycle = {}

for cycle in df['cycle number'].unique():
    df_cycle = df[df['cycle number'] == cycle]

    # Identify charge rows (I>0 or your 'ox/red' column says "Charge")
    df_charge = df_cycle[(df_cycle['I/mA'] > 0)]
    # Identify discharge rows
    df_discharge = df_cycle[(df_cycle['I/mA'] < 0)]
    
    # Example: get average charge current and average discharge current
    I_charge_avg = df_charge['I/mA'].mean() if not df_charge.empty else 0
    I_discharge_avg = df_discharge['I/mA'].mean() if not df_discharge.empty else 0
    
    # Build the action vector
    action_vectors_by_cycle[cycle] = np.array([I_charge_avg, I_discharge_avg])
```

**Step 4**: Pair each `action_vectors_by_cycle[cycle]` with the EIS-based `state_vector` measured *before* or *after* that cycle. This depends on your labeling scheme—some researchers treat “cycle n” as the usage that happens **after** measuring EIS.  

**Important**: Since you said “We only focus on `Ns=1` or `Ns=6` for EIS,” that’s fine for the state vector, but the charge/discharge steps might be `Ns=3`, `Ns=4`, `Ns=8`, etc. So gather data from those steps to form the action vector.

---

## 4. What Do “N=1.00, 2.00, 3.75” or “C x N” Mean?

- In the ASCII header, you see lines like:
  ```
  Apply I/C = C x N
  N = 1.00, 2.00, ...
  ctrl1_val = 100.000 (mA)
  charge/discharge = "Charge" or "Discharge"
  ```
- “C x N” means the test script is intended to run at a multiple of the nominal capacity “C”. If your battery is 4.0 Ah, then “1C” = 4 A, “2C” = 8 A, etc.  
- However, the actual numeric current you see in `ctrl1_val` might be 100 mA or 4.2 V, etc. So there can be a mismatch if the script was custom or if it’s a partial setting.  
- Often, the actual test current you see in `I/mA` is the **true** measurement. The “N=2.00” in the script might be something that was leftover or not used exactly as we might interpret it theoretically.

---

## 5. Summarizing the Key Points

1. **ASCII Table**: Lists the entire test procedure step by step.  
2. **`Ns`**: The step index—`Ns=1` or `6` means EIS steps (PEIS). `Ns=3` or `8` might be CC steps, etc.  
3. **Focus on EIS**: You only keep rows where `Ns=1` or `6` to build your **state vector** from `Re(Z), Im(Z)` at different frequencies.  
4. **But for Action**: You typically want the **charging/discharging data** that happens in the same cycle. In your DataFrame, that might appear at `Ns=3,4,8` or any step labeled as charge or discharge. You can:
   - Group by “`cycle number`”  
   - Within each cycle, find the relevant “charge” vs. “discharge” data in `I/mA`.  
   - Summarize or store the full current profile.  
   - Concatenate into an action vector.

5. **Data Preprocessing**: If you only kept `Ns=1` or `6`, you won’t have the actual charge/discharge currents in your DataFrame. You need to keep (at least temporarily) the other `Ns` that correspond to the usage steps so you can build the action vector.

6. **Matching EIS to Action**: Typically, “EIS measurement” is done right before the next cycle’s usage. So you might say:  
   - The state vector for cycle \(n\) is the EIS data at `Ns=1` (or `6`) that occurs at the start of that cycle.  
   - The action vector for cycle \(n\) is the charge/discharge current that happens right after that EIS, in the same “cycle number.”  
   - The target (capacity) might be measured at the end of that cycle.  

---

### Final Thoughts

- **Interpretation** of “N,” “C x N,” and “ctrl1_val” can be confusing if the test script has extra settings.  
- The **most reliable** approach is usually to look at your logged column `I/mA` in the DataFrame. That tells you the actual current applied at each time (positive = charge, negative = discharge, or vice versa depending on your instrumentation).  
- To create the action vector, **concatenate** those charge/discharge currents (in some consistent manner) and store them alongside your EIS-based state vector.  

Once you do that, you’ll have:

\[
\text{Data for cycle }n: 
  \bigl(\underbrace{\text{EIS state vector}}_{s_n}, 
   \underbrace{\text{charge/discharge current profile}}_{a_n}, 
   \underbrace{\text{capacity at end}}_{Q_n}\bigr).
\]

That’s the typical “state, action, outcome” format you need for your modeling.

Dataset of the conference paper
"Deep Learning-Based State of Health and Remaining Useful Life Prediction for
 Lithium-Ion Batteries Using a Hybrid CNN-LSTM Architecture"

Source
  B. Saha and K. Goebel, "Battery data set," NASA Prognostics Data Repository,
  NASA Ames Research Center, Moffett Field, CA, 2007.
  The NASA PCoE battery aging data are U.S. Government work in the public domain.
  Full repository: NASA Prognostics Center of Excellence (PCoE) Data Set Repository.

Cells included (the four used in the paper)
  B0005.mat   training cell
  B0006.mat   training cell
  B0007.mat   training cell
  B0018.mat   unseen test cell
  These are the original NASA .mat files from the "BatteryAgingARC-FY08Q4" set.
  Lithium-ion 18650 cells cycled at room temperature (charge/discharge/impedance).

How they are used
  One row per discharge cycle, with eight inputs (Cycle, Voltage mean/min/max,
  Current mean, Temperature mean/max, Discharge time) and the targets SOH and RUL.
  SOH = capacity / initial capacity.
  RUL = cycles to the 70% SOH end-of-life point; cells that stay above 70%
  (B0007, B0018) use their last recorded cycle as the end-of-life reference.
  Train on B0005, B0006, B0007; test on the unseen cell B0018.

Note
  Only the four cells used in the paper are redistributed here. The other
  aging-ARC sets are available from the NASA PCoE repository above.

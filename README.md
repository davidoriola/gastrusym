Each experimental dataset includes the Condition, initial GFP fraction ($\phi$), GFP intensity $I(\phi)$, SD GFP intensity $\delta I(\phi)$ and Num samples. 

The dataset includes the following data:
- Experiment 211130
- Experiment 220118
- Experiment 220125
- Experiment 220503
- Experiment 220517 (not used)
- Experiment 220705
- Experiment 220712
- Experiment 220809

To normalize the GFP signal, the 24h data (excluding the $\phi=1$ datapoint) is fitted to a linear function and the GFP intensity for 100% T ($\phi=1$) is extrapolated ($I_1$). In this way, we obtain the average GFP intensity the aggregate
would have if all cells were GFP+. Then the fraction of GFP+ cells in the aggregate (state B cells) is estimated as $\phi_B=I(\phi)/I_1$. Similarly, we have $\delta \phi_B = \delta I(\phi)/I_1$. 

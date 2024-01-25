## Datasets

Each experimental dataset includes the `Condition`, initial `GFP fraction` ($\phi$), `GFP fluo` $I(\phi)$, `GFP fluo SD` $\delta I(\phi)$ and `Num samples` $N$, which corresponds to the number of gastruloids analyzed in the dataset for the certain condition. 

The dataset includes the following data:
- Dataset 1: Experiment 211130. `Condition`: Control, PDO3.
- Dataset 2: Experiment 220118. `Condition`: Control, PDO3.
- Dataset 3: Experiment 220125. `Condition`: Control, PDO3.
- Dataset 4: Experiment 220503. `Condition`: Control, PDO3.
- Dataset 5: Experiment 220705. `Condition`: Control, XAV.
- Dataset 6: Experiment 220712. `Condition`: Control, XAV.
- Dataset 7: Experiment 220517. `Condition`: Control, XAV.

The data used in Fig. 3b control corresponds to Datasets 1,2,3,4,5,6. `Condition`: Control, $(n=6)$. The data used in Fig. 3b PDO3 corresponds to Datasets 1,2,3. `Condition`: Control, $(n=3)$.

## Normalization of the GFP signal

To normalize the GFP signal, the 24h data (excluding the $\phi=1$ datapoint) is fitted to a linear function and the GFP intensity for 100% T ($\phi=1$) is extrapolated ($I_1$). In this way, we obtain the average GFP intensity the aggregate
would have if all cells were GFP+. Then the fraction of GFP+ cells in the aggregate (state B cells) is estimated as $\phi_B^{(i)}=I^{(i)}(\phi)/I_1^{(i)}$, where $i$ corresponds to experiment $i$. Similarly, we have $\delta \phi_B^{(i)} = \delta I(\phi)^{(i)}/I_1^{(i)}$. Next, we average over replicates to obtain $\phi_B =\langle \phi_B^{(i)} \rangle$ to perform the fits.  

## Fits to the control case


The fits were done using the following model:

This case reads:
```math
\begin{eqnarray}
\dot{\phi}_A &= & -  \frac{\phi_A }{1+\phi_A/K} \\
\dot{\phi}_B &= & \frac{\phi_A-\alpha \phi_B}{1+\phi_A/K} 
\end{eqnarray}
```
and can be solved analytically:
```math
\begin{eqnarray}
\phi_A(t) &=& K W \left(\frac{\phi_0}{K} e^{-t+\phi_0/K} \right) \\
\phi_B(t) &=& \frac{1}{\alpha-1} \left[\phi_A(t)+ (-1+\alpha-\alpha \phi_0)  \left(\frac{\phi_A(t)}{\phi_0} \right) ^{\alpha} \right],\quad  \alpha \neq 1 \nonumber \\
\phi_B(t) &=& \frac{\phi_A(t) }{\phi_0} \left[1-\phi_0 + \phi_0 \log \left( \frac{\phi_0}{K} \right) - \phi_0 \log \left(\frac{\phi_A(t)}{K} \right) \right],\quad  \alpha =1 \nonumber 
\end{eqnarray}
```



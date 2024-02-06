## Datasets cell proportion dynamics

Each experimental dataset includes the `Condition`, initial `GFP fraction` ($\phi$), `GFP fluo` $I(\phi)$, `GFP fluo SD` $\delta I(\phi)$ and `Num samples` $N$, which corresponds to the number of gastruloids analyzed in the dataset for the certain condition. 

The dataset includes the following data:
- Dataset 1: Experiment 211130. `Condition`: Control, PDO3.
- Dataset 2: Experiment 220118. `Condition`: Control, PDO3.
- Dataset 3: Experiment 220125. `Condition`: Control, PDO3.
- Dataset 4: Experiment 220503. `Condition`: Control
- Dataset 5: Experiment 220705. `Condition`: Control, SB43, XAV.
- Dataset 6: Experiment 220712. `Condition`: Control, SB43, XAV.

The data used in Fig. 3b control corresponds to Datasets 1,2,3,4,5,6. `Condition`: Control, $(n=6)$. The data used in Fig. 3b PDO3 corresponds to Datasets 1,2,3. `Condition`: Control, $(n=3)$. Data in Fig. S6 corresponds to Datasets 5 and 6 $(n=2)$.

## Normalization of the GFP signal

To normalize the GFP signal, the 24h data (excluding the $\phi=1$ datapoint) is fitted to a linear function and the GFP intensity for 100% T ($\phi=1$) is extrapolated ($I_1$). In this way, we obtain the average GFP intensity the aggregate
would have if all cells were GFP+. Then the fraction of GFP+ cells in the aggregate (state B cells) is estimated as $\phi_B^{(i)}=I^{(i)}(\phi)/I_1^{(i)}$, where $i$ corresponds to experiment $i$. Similarly, we have $\delta \phi_B^{(i)} = \delta I(\phi)^{(i)}/I_1^{(i)}$. Next, we average over replicates to obtain $\phi_B =\langle \phi_B^{(i)} \rangle$ to perform the fits.  

## Fits to the control case

The fits were performed using `curve_fit` from `scipy.optimize`. The fits in Fig 3b control were done using the following dimensionless model:
```math
\begin{eqnarray}
\dot{\phi}_A &= & -  \frac{\phi_A }{1+\phi_A/K} \\
\dot{\phi}_B &= & \frac{\phi_A-\alpha \phi_B}{1+\phi_A/K} 
\end{eqnarray}
```
with $\phi_A+\phi_B+\phi_C=1$, $\alpha=q/p$ and $K$ is the strength of the feedback. $p$ corresponds to the transition rate $A \rightarrow B$ and $q$ corresponds to the transition rate $B \rightarrow C$. Time has been rescaled by the rate $p$. The previous ODE system can be solved analytically and the solution reads:
```math
\begin{eqnarray}
\phi_A(t) &=& K W \left(\frac{\phi_0}{K} e^{-t+\phi_0/K} \right) \\
\phi_B(t) &=& \frac{1}{\alpha-1} \left[\phi_A(t)+ (-1+\alpha-\alpha \phi_0)  \left(\frac{\phi_A(t)}{\phi_0} \right) ^{\alpha} \right],\quad  \alpha \neq 1 \nonumber \\
\phi_B(t) &=& \frac{\phi_A(t) }{\phi_0} \left[1-\phi_0 + \phi_0 \log \left( \frac{\phi_0}{K} \right) - \phi_0 \log \left(\frac{\phi_A(t)}{K} \right) \right],\quad  \alpha =1 \nonumber 
\end{eqnarray}
```
where $\phi_0 \equiv \phi_A(0)=1-\phi_B(0)$ and $W$ is the Lambert W function. We simultaneously fit the averaged 24h, 48h and 72h datasets for the control case with 4 free parameters which are $\alpha, K, T_{24}, T_{48}$, where $T_{24}, T_{48}$ corresponds to the dimensionless 24h and 48h timepoints, respectively. Hence for the 24h fit we use $(\alpha, K, T_{24})$, for the 48h fit we use $(\alpha, K, T_{48})$ and for the 72h fit we use $(\alpha, K, 2T_{48}-T_{24})$. The timescale of the system is obtained as $p^{-1}=24 h/(T_{48}-T_{24})$. Once the parameters were obtained, the time evolution for states A, B and C was plotted using the model in Fig. 3C (left) choosing some initial conditions. 

## Fits to the PDO3 case
The fits were performed using `curve_fit` from `scipy.optimize`. The fits in Fig 3b PDO3 were done using following a first-order linear kinetics dimensionless model:
```math
\begin{eqnarray}
\dot{\phi}_A &= & - \phi_A\\
\dot{\phi}_B &= & \phi_A-\alpha \phi_B
\end{eqnarray}
```
The analytical solution to the previous ODE system reads:
```math
\begin{eqnarray}
\phi_A(t) &=& \phi_0 e^{-t} \\
\phi_B(t) &=& (1-\phi_0) e^{-\alpha t}+ \frac{\phi_0}{\alpha-1}(e^{-t}-e^{-\alpha t})
\end{eqnarray}
```
where $\phi_0 \equiv \phi_A(0) =1-\phi_B(0)$. Again, we simultaneously fit the averaged 24h, 48h and 72h datasets for the control case with now 3 free parameters which are $\alpha, T_{24}, T_{48}$. Once the parameters were obtained, the time evolution for states A, B and C was plotted using the model in Fig. 3C (right) choosing some initial conditions. 


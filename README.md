# 1. Cell proportion dynamics

This section corresponds to the data analysis done in the cell proportion experiments. 

<details>
  
<summary> Datasets cell proportion dynamics </summary>

## 1. Datasets cell proportion dynamics
  
The datasets are found in the folder `Analysis_fits`. Each experimental dataset includes the `Condition`, initial `GFP fraction` ($\phi$), `GFP fluo` $I(\phi)$, `GFP fluo SD` $\delta I(\phi)$ and `Num samples` $N$, which corresponds to the number of gastruloids analyzed in the dataset for the certain initial `GFP fraction`. 

The dataset includes the following data:
- Dataset 1: Experiment 211130. `Condition`: Control, PDO3.
- Dataset 2: Experiment 220118. `Condition`: Control, PDO3.
- Dataset 3: Experiment 220125. `Condition`: Control, PDO3.
- Dataset 4: Experiment 220503. `Condition`: Control
- Dataset 5: Experiment 220705. `Condition`: Control, SB43, XAV.
- Dataset 6: Experiment 220712. `Condition`: Control, SB43, XAV.

The data used in Fig. 5B control corresponds to Datasets 1,2,3,4,5,6. `Condition`: Control, $(n=6)$. The data used in Fig. 5B Nodal/Tgf- $\beta$ inhibition corresponds to Datasets 5,6. `Condition`: SB43, $(n=2)$
The data used in Fig. S13 corresponds to Datasets 1,2,3, `Condition`: PDO3, $(n=3)$ and Datasets 5,6, `Condition`: XAV, $(n=2)$ . Data in Fig. S12 corresponds to Datasets 1,2,3, 5 and 6.

</details>

<details>
  
<summary> Normalization of the GFP signal </summary>

## Normalization of the GFP signal

To normalize the GFP signal, the 24h data (excluding the $\phi=1$ datapoint) is fitted to a linear function and the GFP intensity for 100% T ($\phi=1$) is extrapolated ($I_1$). In this way, we obtain the average GFP intensity the aggregate
would have if all cells were GFP+. Then the fraction of GFP+ cells in the aggregate (state B cells) is estimated as $\phi_{B,i} = I_i(\phi)/I_{1,i}$, where $i$ is a label for the experiment. Similarly, we have $\delta \phi_{B,i} = \delta I_i(\phi)/I_{1,i}$. Finally, we average over replicates to obtain $\phi_B =\langle \phi_B^{(i)} \rangle$ to perform the fits.  

</details>

<details>
  
<summary> 3-state model and fits to the experimental data  </summary>

## 3-state model and fits to the experimental data 

The fits were performed using `curve_fit` from `scipy.optimize`. The fits in Fig 5 and Fig S13 were done using the following dimensionless model:

$$\dot{\phi}_A =  -  \frac{\alpha \phi_A }{1+\phi_A/K} $$ 

$$\dot{\phi}_B =  \frac{\alpha \phi_A- \phi_B}{1+\phi_A/K} $$

with $\phi_A+\phi_B+\phi_C=1$, $\alpha=p/q$ and $K$ is the strength of the feedback. $p$ corresponds to the transition rate $A \rightarrow B$ and $q$ corresponds to the transition rate $B \rightarrow C$. Time has been rescaled by the rate $p$. The previous ODE system can be solved analytically and the solution reads:

$$\phi_A(t) = K W \left(\frac{\phi_0}{K} e^{-\alpha t+\phi_0/K} \right) $$

$$\phi_B(t) = \frac{1}{1-\alpha} \left[\alpha \phi_A(t)+ (1-\alpha-\phi_0)  \left(\frac{\phi_A(t)}{\phi_0} \right) ^{1/\alpha} \right], \quad  \alpha \neq 1, \alpha \neq 0$$

$$\phi_B(t) = \frac{\phi_A(t) }{\phi_0} \left[1-\phi_0 + \phi_0 \log \left( \frac{\phi_0}{K} \right) - \phi_0 \log \left(\frac{\phi_A(t)}{K} \right) \right], \quad  \alpha =1 $$

$$\phi_B(t) = (1-\phi_0) \, \exp \left(-\frac{t}{1+\phi_0/K} \right) \quad \alpha=0 $$

where $\phi_0 \equiv \phi_A(0)=1-\phi_B(0)$ and $W$ is the Lambert W function. We simultaneously fit the averaged 24h, 48h and 72h datasets for the control case with 4 free parameters which are $\alpha, K, T_{24}, T_{48}$, where $T_{24}, T_{48}$ corresponds to the dimensionless 24h and 48h timepoints, respectively. Hence for the 24h fit we use $(\alpha, K, T_{24})$, for the 48h fit we use $(\alpha, K, T_{48})$ and for the 72h fit we use $(\alpha, K, 2T_{48}-T_{24})$. The timescale of the system is obtained as $p^{-1}=24 h/(T_{48}-T_{24})$. Once the parameters were obtained, the time evolution for states A, B and C was plotted using the model in Fig. 5C choosing some initial conditions. 

</details>

# Radial analysis

This section contains all the information regarding the radial analysis in the cell proportion experiments. 

<details>
  
<summary> Datasets radial analysis </summary>

## Datasets radial analysis
  
The datasets can be found in the repository `XXX`. They include the `.tif` files of the images corresponding to the brightfield (BF), GFP and SiR-DNA channels. The datasets also include the masks obtained after segmentation with [MOrgAna](https://github.com/LabTrivedi/MOrgAna). The different experimental replicates are:

- Replicate 1: Experiment 210610. (Fig. S7)
- Replicate 2: Experiment 211116. (Fig. 3C,D and Fig. S7)
- Replicate 3: Experiment 220510. (Fig. S7)

</details>

<details>
  
<summary> Radial analysis procedure </summary>

## Radial analysis procedure

We first pre-processed the SiR-DNA signal $I_{\rm sirDNA}$ for each aggregate by removing high intensity peaks corresponding to 0.3 \% of the total signal. Next, the relative intensity measure $\delta I = I_{\rm sirDNA}/\langle I_{\rm sirDNA} \rangle - I_{\rm GFP}/\langle I_{\rm GFP} \rangle$ was computed for each aggregate image, where the spatial average $\langle \ldots \rangle$ was performed in the mask region. The relative intensity radial profiles for each aggregate were obtained by counting the values $\delta I$ in a set of pixels at a distance $r$ from the center of mass of the aggregate and normalizing by this value. Finally, in order to average the profiles over different aggregates, we first normalised the radial axis by interpolating the profiles.

</details>

# Aspect ratio and polarization analysis

This section corresponds to the morphological and polarization analysis done in the cell proportion experiments. 

<details>
  
<summary> Aspect ratio and polarization analysis </summary>

To obtain the long ($L$) and short ($d$) axis of gastruloids, the brightfield images were straightened using [MOrgAna](https://github.com/LabTrivedi/MOrgAna). The averaged anteroposterior intensity profile $I(x)$ was obtained by analysing several gastruloids using  [MOrgAna](https://github.com/LabTrivedi/MOrgAna), where $x$ was the normalised anteroposterior axis such that $x \in [0,1]$. The polarization parameter $P$ was defined as the first dipolar moment $P(x) = \int_0^1 I(x)(x-1/2)dx / \int_0^1 I(x)$. 

The datasets are found in the folder `Morphometric_fluorescence_analysis`. Each experimental dataset includes the `Condition`, initial `GFP fraction` ($\phi$), average `Aspect ratio` $L/d$, average `Polarization` $P$ and `Num samples` $N$, which corresponds to the number of gastruloids analyzed in the dataset for the certain initial `GFP fraction`.

- Replicate 1: Experiment 220118. 
- Replicate 2: Experiment 220125. 
- Replicate 3: Experiment 220503.
- Replicate 4: Experiment 220705.
- Replicate 5: Experiment 220712.

</details>

# Fusion experiments and nanoindentation analysis

This section corresponds to the analysis of fusion events to characterise the mechanical properties of gastruloids and T+,T- aggregates. 

<details>
  
<summary> Fusion experiments and nanoindentation analysis </summary>

To obtain the long ($L$) axis of a fused assembly over time, segmentation and morphometric analysis was done using [MOrgAna](https://github.com/LabTrivedi/MOrgAna). The data was then fit to a model considering the cellular aggregates behave as Kelvin-Voigt viscoelastic materials [(see Oriola et al. Soft Matter, 18, 3771-3780 2022)](https://pubs.rsc.org/en/content/articlehtml/2022/sm/d2sm00063f). The nanoindentation methods are described in the Methods section of the main article. Different datasets and the corresponding scripts are included:

- Fusion of 24h,48h gastruloids and sorted T+ and T- aggregates: The data and the analysis can be found in the folder `Analysis_Fusion`. This folder contains the scripts to analyse the homotypic and heterotypic fusions using the morphometric data (`.json` or `.csv` output files) generated by [MOrgAna](https://github.com/LabTrivedi/MOrgAna). Additionally, the script `Analysis_rheological_parameters.py` combines the nanoindentation data with the fusion data to extract the surface tension $\gamma$ and shear viscosity $\eta$ of the aggregates. 
- Nanoindentation of gastruloids and sorted T+ and T- aggregates: The 'Chiaro' data and analysis can be found in the folder `Analysis_nanoindentation`.

</details>


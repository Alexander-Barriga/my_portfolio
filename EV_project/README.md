# EV Energy Cost by Performance Tier Analysis 

### Cluster EVs by performance metrics, create an energy cost model, and compare monthly energy costs between brands.

![Image](./plots/Cost_Model/Monthly%20Energy%20Cost%20Distribution%20by%20EV%20Tier.png)

#### TL;DR
- Understand the distributions of EV performance metrics and driver behavior at EV charging stations
- Run statistical hypothesis testing on EV driver charging behavior
- Create correlation model between EV performance metrics and price
- Cluster EVs based on performance metrics and create two tiers
- Create EV energy price model and compare results between EV tiers and brands

## Analysis 
This analysis is broken up into two notebooks: 

### Exploratory Data Analysis (EDA)
  
[EV_EDA.ipynb](./notebooks/EV_EDA.ipynb)
This notebook focuses on exploring both datasets that were used in this analysis by plotting and discussing various distributions and testing hypotheses. Also building a correlation model between EV performance metrics and the price.

### EV Tiers 
[EV_Cost_Model.ipynb](./notebooks/EV_Cost_Model.ipynb))
This notebook focuses on creating EV performance tiers, creating an energy cost model, and comparing discussing the results. The Image for this README file was created in this notebook. 
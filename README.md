# Transfer Learning Predicted the Self-Diffusion Coefficients of Light-Gas in Metal/Covalent Organic Frameworks

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

## Paper Information

- **Title**: Transfer Learning Predicted the Self-Diffusion Coefficients of Light Gases in Metal/Covalent Organic Frameworks
- **Authors**: Tianzi Peng<sup>a,†</sup>, Jiake Shen<sup>a,†</sup>, Shuya Guo<sup>c</sup>, Xiaoxiao Xia<sup>d,*</sup>, Wei Li<sup>a,b,*</sup>
- **Journal**: Acta Chimica Sinica (化学学报)
- **Volume & Issue**: 2026, 84(3): 305-315
- **DOI**: [10.6023/A25110385](https://doi.org/10.6023/A25110385)
- **Journal Link**: [https://sioc-journal.cn/Jwk_hxxb/EN/10.6023/A25110385](https://sioc-journal.cn/Jwk_hxxb/EN/10.6023/A25110385)

## Citation
If you find this code helpful, please cite the original paper:
```bibtex
@article{A25110385,
  title   = {Transfer Learning Predicted the Self-Diffusion Coefficients of Light Gases in Metal/Covalent Organic Frameworks},
  author  = {Peng, Tianzi and Shen, Jiake and Guo, Shuya and Xia, Xiaoxiao and Li, Wei},
  journal = {Acta Chimica Sinica},
  volume  = {84},
  number  = {3},
  pages   = {305--315},
  year    = {2026},
  doi     = {10.6023/A25110385}
}
```
## Abstract
The self-diffusion coefficient of gas molecules within metal/covalent organic frameworks (MOFs/COFs) is a critical physicochemical property that profoundly impacts their performance in gas storage, separation, chemical catalysis, and so on. Molecular dynamics (MD) simulation is a primary approach to assessing the self-diffusion of light-gas in nanoporous materials. With the explosive number of nanoporous materials, machine learning-assisted computational screening to accelerate the investigation of self-diffusion and explore their structure-property relationship has attracted much attention. However, the asymmetric development of the database between MOFs and other nanoporous materials (such as COFs) led to a data imbalance that challenged the development of machine learning for other porous materials, especially for computation-ready experimental (CoRE) databases. Meanwhile, transfer learning (TL) can mitigate such a challenge to enhance generalization by importing similar information extracted from a well-established database (such as CoRE MOFs). This study employs molecular dynamics simulations to predict the self-diffusion coefficients of eight light gases (H<sub>2</sub>, CH<sub>4</sub>, H<sub>2</sub>S, CO<sub>2</sub>, N<sub>2</sub>, C<sub>2</sub>H<sub>6</sub>, C<sub>3</sub>H<sub>8</sub>, C<sub>4</sub>H<sub>10</sub>) in the CoRE MOF database and five light gases (H<sub>2</sub>, CH<sub>4</sub>, H<sub>2</sub>S, CO<sub>2</sub>, N<sub>2</sub>) in the CoRE COF database. By utilizing the descriptor obtained from the nanoporous structure and the gas molecule, three ensemble-based and network-based transfer learning algorithms were trained. In detail, there are seven geometric descriptors obtained from the structure, including the largest cavity diameter (LCD), pore limiting diameter (PLD), largest free path diameter (LFPD), density (ρ), unit cell, void fraction (VF) and pore volume (PV), and four chemical descriptors obtained from light gas, including kinetic dynamic (Dia), quadrupole moment (Qua), polarizability (Pol) and dipole moment (Dip). The Two-Stage TrAdaBoost.R2 algorithm is adopted to adjust the parameter for the ensemble model for transfer learning, whereas the fine-tuning strategy is performed for the neural network for TL. Among them, the light gradient boosting machine (LGBM) was identified as a promising transfer learning model for high-accuracy (R<sub>2</sub>＝0.802) prediction of the self-diffusion. The kinetic diameter, polarizability of gas molecule, and pore limiting diameter of nanoporous structure are emerging as dominant descriptors with relative importance is 14%, 14%, and 12%, in which small Dia, Pol, and large PLD benefit the diffusion. Furthermore, the transfer learning LGBM model can predict the self-diffusion of three types of gas (C<sub>2</sub>H<sub>6</sub>, C<sub>3</sub>H<sub>8</sub>, C<sub>4</sub>H<sub>10</sub>) with a Spearman's correlation coefficient (SRCC) equal to 0.821. This work validates the feasibility of transfer learning-assisted high-throughput screening, offering a feasible approach for deep learning and cross-material studies of nanoporous materials under data scarcity constraints. 
**Keywords**: covalent-organic frameworks; metal-organic frameworks; self-diffusion coefficient; transfer learning; molecular dynamics simulation

## License
This project is licensed under the MIT License (LICENSE).  
Please refer to the original paper and data sources for any restrictions on datasets or model weights.   

## Contact
Feel free to open an Issue for any reproduction problems or suggestions.  
Corresponding authors (according to the paper): Xiaoxiao Xia and Wei Li. 


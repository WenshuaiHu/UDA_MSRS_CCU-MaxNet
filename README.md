# Unsupervised Domain Adaptation with Extended Multimodal LSTM for End-to-end Classification of Multisource Remote Sensing Data

Wen-Shuai Hu, Wei Li, Heng-Chao Li, Xudong Zhao, Mengmeng Zhang, and Ran Tao

Paper web page: [Unsupervised Domain Adaptation with Extended Multimodal LSTM for End-to-end Classification of Multisource Remote Sensing Data](https://xplorestaging.ieee.org/document/11480897).
___________

# Abstract:

Deep learning has gained success in unsupervised domain adaptation (UDA) for remote sensing (RS) data classification. However, existing methods lack specialized deep learning structure components for multisource RS (MSRS) data, restricting their performance. As such, a \underline{C}ross-modal \underline{C}ompensation-consistent \underline{U}-shape \underline{Ma}sked e\underline{x}tended multimodal long short-term memory (xMLSTM) neural \underline{Net}work (CCU-MaxNet) framework is proposed for UDA in end-to-end classification of MSRS data. Firstly, the xMLSTM cell is designed as a new module, with which its down-block and up-block are built to explore complementarity of MSRS data and long-term dependencies between their hierarchical features, serving as memory state. Then, a cross-modal differential compensation (CMDC) block is devised to enhance the complementary modality-specific information. Based on these blocks, a U-MaxNet is built to extract the representational features under the guide of memory state. A multimodal compensation consistent domain adaptation network (MCC-DANet) is further designed for domain alignment at both pixel and semantic levels. Finally, by integrating U-MaxNet and MCC-DANet, CCU-MaxNet extracts more discriminative domain-invariant features for cross-domain classification. Particularly, the proposed framework has excellent flexibility, providing a high-precision mode and a high-efficiency mode based on whether to use fractional Fourier transform to meet the requirements of different applications. Extensive experiments on four cross-domain MSRS datasets verify its superiority, compared to state-of-the-art methods. Our source code will be available at https://github.com/WenshuaiHu/UDA_MSRS_CCU-MaxNet.

Citation
---------------------
**Please kindly cite the papers if this code is useful and helpful for your research.**

Wen-Shuai Hu, Wei Li, Heng-Chao Li, Xudong Zhao, Mengmeng Zhang, and Ran Tao, "Unsupervised Domain Adaptation with Extended Multimodal LSTM for End-to-end Classification of Multisource Remote Sensing Data," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2026.3683514.  <br>

@ARTICLE{11480897, <br>
  author={Hu, Wen-Shuai and Li, Wei and Li, Heng-Chao and Zhao, Xudong and Zhang, Mengmeng and Tao, Ran}, <br>
  journal={IEEE Transactions on Geoscience and Remote Sensing},  <br>
  title={Unsupervised Domain Adaptation with Extended Multimodal LSTM for End-to-end Classification of Multisource Remote Sensing Data},  <br>
  year={2026}, <br>
  volume={}, <br>
  number={}, <br>
  pages={1-1}, <br>
  keywords={Feeds;Antennas;Apertures;Filtering;Field programmable gate arrays;Filters;Pixel;LoRa;Radio access networks;Regional area networks;Unsupervised domain adaptation;multisource remote sensing data;extended multimodal long short-term memory (xMLSTM);cross-modal differential compensation;memory state;multimodal data generation;end-to-end classification}, <br>
  doi={10.1109/TGRS.2026.3683514}}


# Requirements

CUDA Version: 11.8 <br>
Pytorch Version: 2.0 <br>
Python Version: 3.10 <br>

# Note
System-specific notes
---------------------
Please refer to the file `requirements.txt` for the running environment of this code.

Contact Information:
--------------------

Wen-Shuai Hu: wshuswjtu@163.com<br>
He is currently an Assistant Professor with the School of Information Science and Technology, Southwest Jiaotong University, Chengdu, China. 

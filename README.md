# ECNN
Ensemble of CNN for multi-focus image fusion code 

Date of update: 29 April 2020

proposed in:
M. Amin-Naji, A. Aghagolzadeh, and M. Ezoji, “Ensemble of CNN for Multi-Focus Image Fusion”, Information Fusion, vol. 51, pp. 21–214, 2019.  DOI: https://doi.org/10.1016/j.inffus.2019.02.003

(C) Mostafa Amin-Naji, Babol Noshirvani University of Technology, Mostafa.Amin.Naji@Gmail.com, PLEASE CITE THE ABOVE PAPER IF YOU USE THIS CODE My Official Website: www.Amin-Naji.com

Hint: These codes are written based on PyTorch (PyTorch is an open source machine learning library based on the Torch library)

Please do not hesitate to contact me if there is any bug in codes or question from the paper.

Abstract: The Convolutional Neural Networks (CNNs) based multi-focus image fusion methods have recently attracted enormous attention. They greatly enhanced the constructed decision map compared with the previous state of the art methods that have been done in the spatial and transform domains. Nevertheless, these methods have not reached to the satisfactory initial decision map, and they need to undergo vast post-processing algorithms to achieve a satisfactory decision map. In this paper, a novel CNNs based method with the help of the ensemble learning is proposed. It is very reasonable to use various models and datasets rather than just one. The ensemble learning based methods intend to pursue increasing diversity among the models and datasets in order to decrease the problem of the overfitting on the training dataset. It is obvious that the results of an ensemble of CNNs are better than just one single CNNs. Also, the proposed method introduces a new simple type of multi-focus images dataset. It simply changes the arranging of the patches of the multi-focus datasets, which is very useful for obtaining the better accuracy. With this new type arrangement of datasets, the three different datasets including the original and the Gradient in directions of vertical and horizontal patches are generated from the COCO dataset. Therefore, the proposed method introduces a new network that three CNNs models which have been trained on three different created datasets to construct the initial segmented decision map. These ideas greatly improve the initial segmented decision map of the proposed method which is similar, or even better than, the other final decision map of CNNs based methods obtained after applying many post-processing algorithms. Many real multi-focus test images are used in our experiments, and the results are compared with quantitative and qualitative criteria. The obtained experimental results indicate that the proposed CNNs based network is more accurate and have the better decision map without post-processing algorithms than the other existing state of the art multi-focus fusion methods which used many post-processing algorithms.

Keywords: Multi-focus image fusion, Deep learning, Convolutional neural network, Ensemble learning, Decision map




![ECNN patch feeding](https://github.com/mostafaaminnaji/ECNN/blob/master/Data/The%20schematic%20diagram%20of%20generating%20three%20datasets%20according%20to%20the%20proposed%20patch%20feeding.PNG)

![ECNN Network](https://github.com/mostafaaminnaji/ECNN/blob/master/Data/ECNN%20Network.PNG)

![ECNN flowchat of fusion](https://github.com/mostafaaminnaji/ECNN/blob/master/Data/ECNN%20flowhart%20of%20fusion%20of%20two%20images.PNG)



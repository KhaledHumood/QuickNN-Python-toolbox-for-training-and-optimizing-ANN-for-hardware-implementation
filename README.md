# QuickNN-Python-toolbox-for-training-and-optimizing-ANN-for-hardware-implementation
abstract: Realizing deep neural networks in hardware is becoming increasingly challenging as application require ever more layers and weights, incurring computational and storage costs. Current approaches lack the ability to customize Neural network based on hardware design decisions like the number and resolution of neural network inputs, weight quantization, layer quantization, network structure -including number of layers- and activation functions. In this work, we present QuickNN, a Python open-source toolbox for training and optimizing multi-layer perception neural networks designed to aid ANN hardware designers. The toolbox allows the customization of most hardware-related design decisions and can provide rapid results to assess the impact of these decisions on performance. In this paper we demonstrate the toolbox on the MNIST dataset, evaluating accuracy for a range of design choices. By way of example, a novel hybrid weight ternary quantization model is implemented in QuickNN that shows improvement in classification performance when compared to state-of-art.

This toolbox is evaluated using  Spyder interface. Its highly reccomened to download Spyder to avoid any errors/bugs. Spyder by anaconda is used to run the Python codes to import the data and train the neural networks where the following libraries are used: **numpy, matplotlib.pyplot, pickle, copy, scipy.io, skimage.measure and skimage.transform**.

The top file is: **NN_Main.py**, you can characetrize and optimize the NN structures and parameters in **NN_Main.py** and the function automiatically call other functions. 

The activation function can be modified from **NN_activation_function.py**

Citation: 

**Text**: K. Humood, A. Serb, S. Wang and T. Prodromakis, "QuickNN: Python Toolbox for Training and Optimizing ANN for Hardware Implementation," 2023 IEEE 66th International Midwest Symposium on Circuits and Systems (MWSCAS), Tempe, AZ, USA, 2023, pp. 531-535, doi: 10.1109/MWSCAS57524.2023.10405963.

**BibTex:** @INPROCEEDINGS{Humood2023,
  author={Humood, Khaled and Serb, Alex and Wang, Shiwei and Prodromakis, Themis},
  booktitle={2023 IEEE 66th International Midwest Symposium on Circuits and Systems (MWSCAS)}, 
  title={QuickNN: Python Toolbox for Training and Optimizing ANN for Hardware Implementation}, 
  year={2023},
  volume={},
  number={},
  pages={531-535},
  keywords={Training;Quantization (signal);Costs;Circuits and systems;Artificial neural networks;Hardware;Integrated circuit modeling;Python;Neural;Network;Training;MNIST;Ternary;Quantization;Toolbox;Multi-layer;Perceptions},
  doi={10.1109/MWSCAS57524.2023.10405963}}

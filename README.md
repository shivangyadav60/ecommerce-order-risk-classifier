# Amazon Risk Classifier

A deep learning pipeline built with Keras and TensorFlow to identify account risk profiles, such as fraudulent accounts, high-risk transactions, or vendor return compliance issues, using synthetic e-commerce behavioral metrics.

## Business Impact and Problem Statement
Detecting bad actors early is essential for protecting retail ecosystems, avoiding revenue leakage, and maintaining customer trust. This project builds a Deep Neural Network (DNN) capable of processing user behavioral metrics, transaction distributions, and metadata to automatically flag risk dimensions before they disrupt operations.

The classifier maps targets into three distinct risk categories:
* 0: Normal (Standard safe transaction profile)
* 1: Fraud Risk (High-probability anomalous actor)
* 2: Return Risk (Exploitative return behavior patterns)

## Architecture and Preprocessing Pipeline
The network is designed using the Keras Functional API to establish structural multi-input feature routing. 

### Data Engineering Practices
* Dimensionality Control: High-cardinality identity parameters (order_id, order_date) are dropped during feature space generation. This prevents categorical column explosions during one-hot encoding, saving system memory and mitigating overfitting noise.
* Vector Realignment: Categorical markers are transformed via pandas get_dummies, aligned into continuous matrices, and mapped to float32 and int32 NumPy tensors ready for backend ingestion.
* Network Topology: The architecture anchors an explicit symbolic input layer based on the number of columns, routes dimensions into a core Dense layer with 8 units and ReLU activation, and resolves configurations using a final Dense layer with 3 units and Softmax classification.
* Optimization Strategy: Compiled using the Adam optimizer, resolving errors via sparse_categorical_crossentropy to calculate structural integer maps natively without manual one-hot matrix alignment overhead.

## Tech Stack
* Deep Learning Framework: Keras 3.x / TensorFlow Backend
* Data Processing: NumPy, Pandas, Scikit-Learn
* Visualization: Matplotlib

## Performance and Optimization History
The network tracks convergence parameters dynamically across training steps. Both optimization trajectories (Loss minimization and Accuracy scaling) are saved directly out to the environment.

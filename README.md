# OHeapKNN: Oblivious Heap-Enhanced Privacy-Preserving K-Nearest Neighbors Classification Over Outsourced Data

> 🚧 **Note**  
> The code is still under active development and may be further refined in future updates.

---

## 📖 Introduction

This repository provides the implementation of **OHeapKNN**, a privacy-preserving **K-Nearest Neighbors (KNN)** classification framework designed for outsourced data scenarios.

---

## 🔑 Key Contributions

The main contributions of this work are summarized as follows:

### 1️⃣ Efficient Nearest Neighbors Selection Protocol
We propose a general protocol that satisfies fundamental security requirements by adopting a **parallel-path update strategy** inspired by heap-based designs.  

- Reduces communication rounds from **O(n log k)** to **O(n + log k)**  

---

### 2️⃣ Oblivious Heap Design
We design a novel **oblivious heap** to hide access patterns.  

- Enables **OH-enhanced OHeapKNN**  
- Reduces communication rounds from **O(nk)** to **O(n + log k)**  

---

### 3️⃣ Formal Security Guarantees
All protocols are formally proven secure under:

- Real/ideal world paradigm  
- Universal composability framework  

---

### 4️⃣ Extensive Experimental Evaluation
Experiments on diverse real-world datasets demonstrate that **OHeapKNN and OHeapKNN_b significantly outperform state-of-the-art schemes** in:

- 🚀 Runtime performance  
- 📡 Communication efficiency
- 🔧 Extensibility to other KNN-related schemes for further optimization  
- 🔄 Compatibility with diverse distance functions  

---

## 🧩 Versions

This repository includes three versions of the protocol:

- **Heap-based**  
  → Sequential heap update (baseline implementation)

- **OHeapKNN_b**  
  → Parallel-path heap update (efficiency-optimized version)

- **OHeapKNN**  
  → Oblivious heap-based implementation with access pattern hiding


---

## 📚 References

> [1] G. Lin, R. Zhou, S. Chen, W. Han, J. Tan, W. Fang, L. Wang, and T. Wei, *Kona: An Efficient Privacy-Preservation Framework for KNN Classification by Communication Optimization,* _ICML_ 2025.

> [2] Z. Li, H. Wang, S. Zhang, W. Zhang, and R. Lu, *SecKNN: FSS-Based Secure Multi-Party KNN Classification Under General Distance Functions,* _IEEE TIFS_, 2024.

> [3] L. Liu et al., *Toward Highly Secure yet Efficient KNN Classification Scheme on Outsourced Cloud Data,* _IEEE IoT Journal_, 2019.

> [4] H. Morita et al., *Constant-Round Client-Aided Secure Comparison Protocol,* _ESORICS_ 2018.

---

## 📁 Code Structure
- `Heap-based.cpp`  
  → Baseline implementation with sequential heap update.
- `OHeapKNN_b.cpp`  
  → Implementation of OHeapKNN with **parallel-path heap update**, improving efficiency via level-wise batched operations.

- `OHeapKNN.cpp`  
  → Full **oblivious heap-based implementation** of OHeapKNN, providing access pattern hiding and stronger security guarantees.

---

## 🔗 Dependencies

This project is developed based on the **Garnet framework**:

👉 https://github.com/FudanMPL/Garnet

Please refer to the original Garnet repository for:
- environment setup
- compilation instructions
- runtime configuration

---
## ▶️ How to Run

This implementation is built upon the Garnet framework and follows its standard KNN execution pipeline.

For environment setup, data preparation, and execution instructions, please refer to:

👉 https://github.com/FudanMPL/Garnet/blob/main/docs/knn.md

In particular, the execution workflow includes:

- Setting up the Garnet framework  
- Preparing the input datasets  
- Configuring the network environment  
- Generating the required certificates  
- Compiling and running the protocol

This project follows the Garnet execution pipeline while replacing the KNN protocol with our OHeapKNN variants.

Detailed scripts and example configurations will be provided in future updates.

---

## 📜 License & Acknowledgment

This project is built upon the open-source [Garnet framework](https://github.com/FudanMPL/Garnet).

Garnet is released under the BSD 3-Clause License. This project is distributed in compliance with the licensing requirements of Garnet.

The Garnet framework incorporates several third-party components, including:
- MP-SPDZ
- SPDZ-2
- SPDZ-BMR-ORAM
- SCALE-MAMBA
- NFGen

These components are distributed under their respective open-source licenses.  
For detailed licensing information, please refer to the original Garnet repository.

---

## ⚠️ Disclaimer

This code is provided **"as is"**, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability and fitness for a particular purpose.

In no event shall the authors or contributors be liable for any damages arising from the use of this software.

---



## 📌 Notes

- This code is intended for **research and experimental purposes**.
- If you use this work in your research, please cite the corresponding paper (to be updated).

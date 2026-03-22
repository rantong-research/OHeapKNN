# OHeapKNN: Oblivious Heap-Enhanced Privacy-Preserving K-Nearest Neighbors Classification Over Outsourced Data

> 🚧 **Note**  
> The code is currently being organized and will be released soon.

---

## 📖 Introduction

This repository provides the implementation of **OHeapKNN**, a privacy-preserving **K-Nearest Neighbors (KNN)** classification framework designed for outsourced data scenarios.

---

## 🔑 Key Contributions

The main contributions of this work are summarized as follows:

### 1️⃣ Efficient Nearest Neighbors Selection Protocol
We propose a general protocol that satisfies fundamental security requirements by adopting a **parallel-path update strategy** inspired by heap-based designs.  

- Reduces communication rounds from **O(n log k)** → **O(n + log k)**  

---

### 2️⃣ Oblivious Heap Design
We design a novel **oblivious heap** to hide access patterns.  

- Enables **OH-enhanced OHeapKNN**  
- Reduces communication rounds from **O(nk)** → **O(n + log k)**  

---

### 3️⃣ Formal Security Guarantees
All protocols are formally proven secure under:

- Real/ideal world paradigm  
- Universal composability framework  

---

### 4️⃣ Extensive Experimental Evaluation
Experiments on diverse real-world datasets demonstrate that **OHeapKNN significantly outperforms state-of-the-art schemes** in:

- 🚀 Runtime performance  
- 📡 Communication efficiency
- 🔧 Extensibility to other KNN-related schemes for further optimization  
- 🔄 Compatibility with diverse distance functions  

---

## 🧩 Versions

This repository includes two versions of the protocol:

### ⚡ Basic Version
- Optimized for **efficiency**  
- Reduced communication overhead  
- ❗ Does **not** fully hide access patterns  

---
### 🔐 Secure Version
- Based on an **Oblivious Heap**  
- Hides access patterns  
- Provides **stronger security guarantees**


---

## 📚 References

> [1] G. Lin, R. Zhou, S. Chen, W. Han, J. Tan, W. Fang, L. Wang, and T. Wei, *Kona: An Efficient Privacy-Preservation Framework for KNN Classification by Communication Optimization,* _ICML_ 2025.

> [2] Z. Li, H. Wang, S. Zhang, W. Zhang, and R. Lu, *SecKNN: FSS-Based Secure Multi-Party KNN Classification Under General Distance Functions,* _IEEE TIFS_, 2024.

> [3] L. Liu et al., *Toward Highly Secure yet Efficient KNN Classification Scheme on Outsourced Cloud Data,* _IEEE IoT Journal_, 2019.

> [4] H. Morita et al., *Constant-Round Client-Aided Secure Comparison Protocol,* _ESORICS_ 2018.

---

---

## 📁 Code Structure

- `OHeapKNN_b.cpp`  
  → **Basic version** of OHeapKNN.

- `OHeapKNN.cpp`  
  → **Secure version** of OHeapKNN.

---

## 🔗 Dependencies

This project is developed based on the **Garnet framework**:

👉 https://github.com/FudanMPL/Garnet

Please follow the original Garnet repository for:
- environment setup
- compilation instructions
- runtime configuration

---
## ▶️ How to Run

> 🚧 This section will be updated soon.

The implementation is built upon the Garnet framework.  
Detailed instructions for environment setup, compilation, and execution will be provided after the code organization is finalized.

In general, the workflow follows:
1. Set up the Garnet environment  
2. Compile the project  
3. Run the protocol with two-party configuration  

More details (including example commands and datasets) will be added in the next update.

---

## 📜 License & Acknowledgment

This project is built upon the open-source [Garnet framework](https://github.com/FudanMPL/Garnet).

Garnet is released under the BSD 3-Clause License. This project follows the same licensing requirements.

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

This software is provided **"as is"**, without warranty of any kind, express or implied, including but not limited to:

- merchantability  
- fitness for a particular purpose  

In no event shall the authors or contributors be liable for any damages arising from the use of this software.

---



## 📌 Notes

- This codebase is intended for **research and experimental purposes**.
- If you use this work in your research, please cite the corresponding paper (to be updated).

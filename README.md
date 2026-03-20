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

> [1] G. Lin, R. Zhou, S. Chen, W. Han, J. Tan, W. Fang, L. Wang, and T. Wei, *Kona: An Efficient Privacy-Preservation Framework for KNN Classification by Communication Optimization,* ICML 2025.

> [2] Z. Li, H. Wang, S. Zhang, W. Zhang, and R. Lu, *SecKNN: FSS-Based Secure Multi-Party KNN Classification Under General Distance Functions,* IEEE TIFS, 2024.

> [3] L. Liu et al., *Toward Highly Secure yet Efficient KNN Classification Scheme on Outsourced Cloud Data,* IEEE IoT Journal, 2019.

> [4] H. Morita et al., *Constant-Round Client-Aided Secure Comparison Protocol,* ESORICS 2018.

---

## 📌 Notes

- This project is a research prototype.  
- For academic use, please cite the corresponding paper (to be added).  

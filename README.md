OHeapKNN: Oblivious Heap-Enhanced Privacy-Preserving K-Nearest Neighbors Classification Over Outsourced Data
📖 Introduction

🚧 Note: The code is currently being organized and will be uploaded soon.

This repository provides the implementation of OHeapKNN, a privacy-preserving K-Nearest Neighbors (KNN) classification framework designed for outsourced data scenarios.

🔑 Key Contributions

The main contributions of this work are summarized as follows:

Efficient Neighbor Selection Protocol
We propose a general protocol that satisfies fundamental security requirements by adopting a parallel-path update strategy inspired by heap-based designs.
This reduces the communication rounds for selecting the $k$ nearest neighbors from O(n log k) to O(n + log k).

Oblivious Heap Design
We design a novel oblivious heap to hide access patterns.
Based on this structure, we introduce OH-enhanced OHeapKNN, reducing the communication complexity from O(nk) to O(n + log k).

Formal Security Guarantees
All protocols are formally proven secure under the real/ideal world paradigm the Universal Composability (UC) framework

Extensive Experimental Evaluation
Experiments on diverse real-world datasets show that OHeapKNN significantly outperforms state-of-the-art methods in:

runtime performance

communication efficiency

🧩 Versions

This repository includes two versions of the protocol:

🔐 Secure Version
Based on an Oblivious Heap, which hides access patterns and provides stronger security guarantees.

⚡ Basic Version
Focuses on efficiency with reduced communication overhead, but does not fully hide access patterns.

📚 References

[1] G. Lin, R. Zhou, S. Chen, W. Han, J. Tan, W. Fang, L. Wang, and T. Wei, ``Kona: An Efficient Privacy-Preservation Framework for KNN Classification by Communication Optimization,'' in Proceedings of the 42nd International Conference on Machine Learning (ICML 2025), Vancouver, Canada, Jul. 2025. doi: 1VqxIgyQlp

[2] Z. Li, H. Wang, S. Zhang, W. Zhang, and R. Lu, ``SecKNN: FSS-Based Secure Multi-Party KNN Classification Under General Distance Functions,'' IEEE Transactions on Information Forensics and Security, vol. 19, pp. 1326–1341, Jan. 2024. doi: 10.1109/TIFS.2023.3337940

[3] L. Liu et al., ``Toward highly secure yet efficient KNN classification scheme on outsourced cloud data,'' \textit{IEEE Internet Things J.}, vol. 6, no. 6, pp. 9841--9852, Dec. 2019.

[4] H. Morita, N. Attrapadung, T. Teruya, S. Ohata, K. Nuida, and G. Hanaoka, ``Constant-Round Client-Aided Secure Comparison Protocol,'' in Computer Security (ESORICS 2018). Cham: Springer International Publishing, Aug. 2018, pp. 395–415. doi: 10.1007/978-3-319-98989-1_20

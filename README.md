Parallel Heap Optimized Privacy-Preserving KNN
📖 Introduction

This project implements a parallel heap optimized privacy-preserving k-Nearest Neighbors (KNN) algorithm.
The main contributions are:

Parallel Heap Optimization: Leveraging parallelized heap operations to reduce communication rounds and computational overhead.

Two Versions:

Secure Version: Based on an Oblivious Heap, which hides access patterns for stronger security guarantees.

Basic Version: Focused on efficiency with reduced communication, but without full access-pattern hiding.

Secure Comparison Protocol: A simplified implementation of a constant-round secure comparison protocol based on the work of Morita et al. [1].

⚙️ Requirements

Operating System: Linux (Ubuntu 20.04/22.04 recommended)

Compiler: g++ >= 9.4 with C++11/14 support


📚 References

[1] H. Morita, N. Attrapadung, T. Teruya, S. Ohata, K. Nuida, and G. Hanaoka,
“Constant-Round Client-Aided Secure Comparison Protocol,”
in Computer Security (ESORICS 2018). Cham: Springer International Publishing, Aug. 2018, pp. 395–415.
doi: 10.1007/978-3-319-98989-1_20

[2] Z. Li, H. Wang, S. Zhang, W. Zhang, and R. Lu,
“SecKNN: FSS-Based Secure Multi-Party KNN Classification Under General Distance Functions,”
IEEE Transactions on Information Forensics and Security, vol. 19, pp. 1326–1341, Jan. 2024.
doi: 10.1109/TIFS.2023.3337940

[3] G. Lin, R. Zhou, S. Chen, W. Han, J. Tan, W. Fang, L. Wang, and T. Wei,
“Kona: An Efficient Privacy-Preservation Framework for KNN Classification by Communication Optimization,”
in Proceedings of the 42nd International Conference on Machine Learning (ICML 2025), Vancouver, Canada, Jul. 2025.
doi: 1VqxIgyQlp

# From_Zero_To_Sgemm
optimize a sgemm function fast as cublas gradually. 
m = n = k = 1024
0. cublas 202125 cycle
1. naive 
2. memory coalesce
3. tiling
4. threadtiling: 262495 cycle (77%)
5. vectorized memory: 249158 cycle (81.12%)
6. bank conflict: 241714 cycle(83%)
7. double buffer: 255620 cylcle (79.07%)

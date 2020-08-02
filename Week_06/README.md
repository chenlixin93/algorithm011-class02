### 动态规划 Dynamic Programming

1. Wiki 定义： https://en.wikipedia.org/wiki/Dynamic_programming

2. “Simplifying a complicated problem by breaking it down into simpler sub-problems”  (in a recursive manner) 

3. Divide & Conquer + Optimal substructure 

分治 + 最优子结构

### 关键点

动态规划 和 递归或者分治 没有根本上的区别（关键看有无最优的子结构） 

共性：找到重复子问题

差异性：最优子结构、中途可以淘汰次优解

### 例题

1. 最优子结构  opt[n] = best_of(opt[n-1], opt[n-2], ...) 

2. 储存中间状态：opt[i] 

3. 递推公式（美其名曰：状态转移方程或者 DP 方程）  

Fib: opt[i] = opt[n-1] + opt[n-2] 

二维路径：opt[i,j] = opt[i+1][j] + opt[i][j+1] (且判断a[i,j]是否空地）


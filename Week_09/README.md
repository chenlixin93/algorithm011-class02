### 高级动态规划

**动态规划 Dynamic Programming**

1. Wiki 定义： https://en.wikipedia.org/wiki/Dynamic_programming

2. “Simplifying a complicated problem by breaking it down into simpler sub-problems”  (in a recursive manner) 

3. Divide & Conquer + Optimal substructure 

分治 + 最优子结构

**DP顺推模板**

```python

function DP():  

    dp = [][] # 二维情况  

    for i = 0 .. M {     
        for j = 0 .. N {        
            dp[i][j] = _Function(dp[i’][j’]...)     
        }   
    }

    return dp[M][N]; 

```

**关键点**

动态规划 和 递归或者分治 没有根本上的区别（关键看有无最优的子结构） 

共性：找到重复子问题

差异性：最优子结构、中途可以淘汰次优解

### 字符串算法

**Atoi 代码示例**

```java
// Java
public int myAtoi(String str) {
    int index = 0, sign = 1, total = 0;
    //1. Empty string
    if(str.length() == 0) return 0;
    //2. Remove Spaces
    while(str.charAt(index) == ' ' && index < str.length())
        index ++;
    //3. Handle signs
    if(str.charAt(index) == '+' || str.charAt(index) == '-'){
        sign = str.charAt(index) == '+' ? 1 : -1;
        index ++;
    }
    
    //4. Convert number and avoid overflow
    while(index < str.length()){
        int digit = str.charAt(index) - '0';
        if(digit < 0 || digit > 9) break;
        //check if total will be overflow after 10 times and add digit
        if(Integer.MAX_VALUE/10 < total ||            
        	Integer.MAX_VALUE/10 == total && Integer.MAX_VALUE %10 < digit)
            return sign == 1 ? Integer.MAX_VALUE : Integer.MIN_VALUE;
        total = 10 * total + digit;
        index ++;
    }
    return total * sign;
}
```
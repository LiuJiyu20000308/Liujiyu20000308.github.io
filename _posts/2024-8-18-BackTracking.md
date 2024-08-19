---
layout: post
title: Leetcode记录：回溯
date: 2024-8-18 14:00 +0800
tags: [数据结构与算法]
toc: true
---

## 回溯框架

回溯法，一般可以解决如下几种问题：
1. 组合问题：N个数里面按一定规则找出k个数的集合
2. 切割问题：一个字符串按一定规则有几种切割方式
3. 子集问题：一个N个数的集合里有多少符合条件的子集
4. 排列问题：N个数按一定规则全排列，有几种排列方式
5. 棋盘问题：N皇后，解数独等等

回溯法解决的问题都可以抽象为树形结构，因为回溯法解决的都是在集合中递归查找子集，集合的大小就构成了树的宽度，递归的深度就构成了树的深度。递归就要有终止条件，所以必然是一棵高度有限的树（N叉树）。

回溯的三个重要部分：
1. 回溯函数模板返回值以及参数：回溯算法需要的参数可不像二叉树递归的时候那么容易一次性确定下来，所以一般是先写逻辑，然后需要什么参数，就填什么参数。
2. 回溯函数终止条件：什么时候达到了终止条件，树中就可以看出，一般来说搜到叶子节点了，也就找到了满足条件的一条答案，把这个答案存放起来，并结束本层递归。
3. 回溯搜索的遍历过程

```cpp
void backtracking(参数) {
    if (终止条件) {
        存放结果;
        return;
    }

    for (选择：本层集合中元素（树中节点孩子的数量就是集合的大小）) {
        处理节点;
        backtracking(路径，选择列表); // 递归
        回溯，撤销处理结果
    }
}
```

## 组合问题

### 组合（最基本的回溯）
Leetcode 77. 给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。

```cpp
vector<vector<int>> result;
vector<int> path;
void backtracking(int n, int k, int startIndex) {
    if (path.size() == k) {
        result.push_back(path);
        return;
    }
    for (int i = startIndex; i <= n - (k - path.size()) + 1; i++) { // 优化的地方
        path.push_back(i); // 处理节点
        backtracking(n, k, i + 1);
        path.pop_back(); // 回溯，撤销处理的节点
    }
}
vector<vector<int>> combine(int n, int k) {
    backtracking(n, k, 1);
    return result;
}
```

### 组合求和I
Leetcode 39. 给定一个无重复元素的数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。candidates 中的数字可以**无限制重复被选取。**
**只需注意遍历的时候startIdx还可以从自身开始即可。**
```cpp
vector<vector<int>> result;
vector<int> path;
void backtracking(vector<int>& candidates, int target, int sum, int startIndex) {
    if (sum == target) {
        result.push_back(path);
        return;
    }

    // 如果 sum + candidates[i] > target 就终止遍历
    for (int i = startIndex; i < candidates.size() && sum + candidates[i] <= target; i++) {
        sum += candidates[i];
        path.push_back(candidates[i]);
        backtracking(candidates, target, sum, i); //唯一注意的地方，不是i+1
        sum -= candidates[i];
        path.pop_back();

    }
}
vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
    result.clear();
    path.clear();
    sort(candidates.begin(), candidates.end()); // 需要排序
    backtracking(candidates, target, 0, 0);
    return result;
}
```

### 组合求和II
Leetcode 216. 找出所有相加之和为 n 的 k 个数的组合。组合中只允许含有 1 - 9 的正整数，并且每种组合中不存在重复的数字。

**只需改变一下终止条件即可。**
```cpp
vector<vector<int>> result; // 存放结果集
vector<int> path; // 符合条件的结果
// targetSum：目标和，也就是题目中的n。
// k：题目中要求k个数的集合。
// sum：已经收集的元素的总和，也就是path里元素的总和。
// startIndex：下一层for循环搜索的起始位置。
void backtracking(int targetSum, int k, int sum, int startIndex) {
    if (path.size() == k) {
        if (sum == targetSum) result.push_back(path);
        return; // 如果path.size() == k 但sum != targetSum 直接返回
    }
    for (int i = startIndex; i <= 9; i++) {
        sum += i; // 处理
        path.push_back(i); // 处理
        backtracking(targetSum, k, sum, i + 1); // 注意i+1调整startIndex
        sum -= i; // 回溯
        path.pop_back(); // 回溯
    }
}
vector<vector<int>> combinationSum3(int k, int n) {
    result.clear(); // 可以不加
    path.clear();   // 可以不加
    backtracking(n, k, 0, 1);
    return result;
    }
```

### 组合求和III（有重复元素）
给定一个数组 candidates 和一个目标数 target ，找出 candidates 中所有可以使数字和为 target 的组合。candidates 中的每个数字在每个组合中只能使用一次。
说明： 所有数字（包括目标数）都是正整数。**解集不能包含重复的组合。**

本题的难点在于：**集合（数组candidates）有重复元素，但还不能有重复的组合。**组合问题可以抽象为树形结构，那么“使用过”在这个树形结构上是有两个维度的，一个维度是同一树枝上使用过，一个维度是同一树层上使用过。所以我们要去重的是同一树层上的“使用过”，同一树枝上的都是一个组合里的元素，不用去重。

如果candidates[i] == candidates[i - 1] 并且 used[i - 1] == false，就说明：前一个树枝，使用了candidates[i - 1]，也就是说同一树层使用过candidates[i - 1]，此时for循环里就应该做continue的操作。
**注意，树层去重的话，需要对数组排序！**
```cpp
vector<vector<int>> result;
vector<int> path;
void backtracking(vector<int>& candidates, int target, int sum, int startIndex, vector<bool>& used) {
    if (sum == target) {
        result.push_back(path);
        return;
    }
    for (int i = startIndex; i < candidates.size() && sum + candidates[i] <= target; i++) {
        // used[i - 1] == true，说明同一树枝candidates[i - 1]使用过
        // used[i - 1] == false，说明同一树层candidates[i - 1]使用过
        // 要对同一树层使用过的元素进行跳过
        if (i > 0 && candidates[i] == candidates[i - 1] && used[i - 1] == false) {
            continue;
        }
        sum += candidates[i];
        path.push_back(candidates[i]);
        used[i] = true;
        backtracking(candidates, target, sum, i + 1, used); // 和39.组合总和的区别1，这里是i+1，每个数字在每个组合中只能使用一次
        used[i] = false;
        sum -= candidates[i];
        path.pop_back();
    }
}
vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
    vector<bool> used(candidates.size(), false);
    path.clear();
    result.clear();
    // 首先把给candidates排序，让其相同的元素都挨在一起。
    sort(candidates.begin(), candidates.end());
    backtracking(candidates, target, 0, 0, used);
    return result;
}
```
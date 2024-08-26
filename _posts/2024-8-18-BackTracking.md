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

本题的难点在于：**集合（数组candidates）有重复元素，但还不能有重复的组合。** 组合问题可以抽象为树形结构，那么“使用过”在这个树形结构上是有两个维度的，一个维度是同一树枝上使用过，一个维度是同一树层上使用过。所以我们要去重的是同一树层上的“使用过”，同一树枝上的都是一个组合里的元素，不用去重。

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

## 分割问题

### 分割回文串
Leetcode 131. 给定一个字符串 s，将 s 分割成一些子串，使每个子串都是回文串。返回 s 所有可能的分割方案。

其实切割问题类似组合问题:
```cpp
vector<vector<string>> result;
vector<string> path; // 放已经回文的子串
void backtracking (const string& s, int startIndex) {
    // 如果起始位置已经大于s的大小，说明已经找到了一组分割方案了
    if (startIndex >= s.size()) {
        result.push_back(path);
        return;
    }
    for (int i = startIndex; i < s.size(); i++) {
        if (isPalindrome(s, startIndex, i)) {   // 是回文子串
            // 获取[startIndex,i]在s中的子串
            string str = s.substr(startIndex, i - startIndex + 1);
            path.push_back(str);
        } else {                                // 不是回文，跳过
            continue;
        }
        backtracking(s, i + 1); // 寻找i+1为起始位置的子串
        path.pop_back(); // 回溯过程，弹出本次已经添加的子串
    }
}
bool isPalindrome(const string& s, int start, int end) {
    for (int i = start, j = end; i < j; i++, j--) {
        if (s[i] != s[j]) {
            return false;
        }
    }
    return true;
}
vector<vector<string>> partition(string s) {
    result.clear();
    path.clear();
    backtracking(s, 0);
    return result;
}
```
也可以提前利用DP把回文串计算好：
```cpp
 vector<vector<string>> result;
vector<string> path; // 放已经回文的子串
vector<vector<bool>> isPalindrome; // 放事先计算好的是否回文子串的结果
void backtracking (const string& s, int startIndex) {
    // 如果起始位置已经大于s的大小，说明已经找到了一组分割方案了
    if (startIndex >= s.size()) {
        result.push_back(path);
        return;
    }
    for (int i = startIndex; i < s.size(); i++) {
        if (isPalindrome[startIndex][i]) {   // 是回文子串
            // 获取[startIndex,i]在s中的子串
            string str = s.substr(startIndex, i - startIndex + 1);
            path.push_back(str);
        } else {                                // 不是回文，跳过
            continue;
        }
        backtracking(s, i + 1); // 寻找i+1为起始位置的子串
        path.pop_back(); // 回溯过程，弹出本次已经添加的子串
    }
}
void computePalindrome(const string& s) {
    // isPalindrome[i][j] 代表 s[i:j](双边包括)是否是回文字串 
    isPalindrome.resize(s.size(), vector<bool>(s.size(), false)); // 根据字符串s, 刷新布尔矩阵的大小
    for (int i = s.size() - 1; i >= 0; i--) { 
        // 需要倒序计算, 保证在i行时, i+1行已经计算好了
        for (int j = i; j < s.size(); j++) {
            if (j == i) {isPalindrome[i][j] = true;}
            else if (j - i == 1) {isPalindrome[i][j] = (s[i] == s[j]);}
            else {isPalindrome[i][j] = (s[i] == s[j] && isPalindrome[i+1][j-1]);}
        }
    }
}
vector<vector<string>> partition(string s) {
    result.clear();
    path.clear();
    computePalindrome(s);
    backtracking(s, 0);
    return result;
}
```
### 复原IP地址
Leetcode 93. 给定一个只包含数字的字符串，复原它并返回所有可能的 IP 地址格式。有效的 IP 地址 正好由四个整数（每个整数位于 0 到 255 之间组成，且不能含有前导 0），整数之间用 '.' 分隔。

仍是要检查每个字符串是否满足性质，满足则继续划分，**注意insert函数的作用效果以及插入逗点后backTracking的startIdx**：
```cpp
vector<string> result;// 记录结果
// startIndex: 搜索的起始位置，pointNum:添加逗点的数量
void backtracking(string& s, int startIndex, int pointNum) {
    if (pointNum == 3) { // 逗点数量为3时，分隔结束
        // 判断第四段子字符串是否合法，如果合法就放进result中
        if (isValid(s, startIndex, s.size() - 1)) {
            result.push_back(s);
        }
        return;
    }
    for (int i = startIndex; i < s.size(); i++) {
        if (isValid(s, startIndex, i)) { // 判断 [startIndex,i] 这个区间的子串是否合法
            s.insert(s.begin() + i + 1 , '.');  // 在i的后面插入一个逗点
            pointNum++;
            backtracking(s, i + 2, pointNum);   // 插入逗点之后下一个子串的起始位置为i+2
            pointNum--;                         // 回溯
            s.erase(s.begin() + i + 1);         // 回溯删掉逗点
        } else break; // 不合法，直接结束本层循环
    }
}
// 判断字符串s在左闭又闭区间[start, end]所组成的数字是否合法
bool isValid(const string& s, int start, int end) {
    if (start > end) {
        return false;
    }
    if (s[start] == '0' && start != end) { // 0开头的数字不合法
            return false;
    }
    int num = 0;
    for (int i = start; i <= end; i++) {
        if (s[i] > '9' || s[i] < '0') { // 遇到非数字字符不合法
            return false;
        }
        num = num * 10 + (s[i] - '0');
        if (num > 255) { // 如果大于255了不合法
            return false;
        }
    }
    return true;
}
vector<string> restoreIpAddresses(string s) {
    result.clear();
    if (s.size() < 4 || s.size() > 12) return result; // 算是剪枝了
    backtracking(s, 0, 0);
    return result;
}
```

## 子集问题

### 子集I
Leetcode 78. 给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。

如果把 子集问题、组合问题、分割问题都抽象为一棵树的话，那么**组合问题和分割问题都是收集树的叶子节点，而子集问题是找树的所有节点！** 其实子集也是一种组合问题，因为它的集合是无序的，子集{1,2} 和 子集{2,1}是一样的。那么既然是无序，取过的元素不会重复取，**写回溯算法的时候，for就要从startIndex开始，而不是从0开始！**

```cpp
vector<vector<int>> result;
vector<int> path;
void backtracking(vector<int>& nums, int startIndex) {
    result.push_back(path); // 收集子集，要放在终止添加的上面，否则会漏掉自己
    if (startIndex >= nums.size()) { // 终止条件可以不加
        return;
    }
    for (int i = startIndex; i < nums.size(); i++) {
        path.push_back(nums[i]);
        backtracking(nums, i + 1);
        path.pop_back();
    }
}
vector<vector<int>> subsets(vector<int>& nums) {
    result.clear();
    path.clear();
    backtracking(nums, 0);
    return result;
}
```

### 子集II
Leetcode 90. 给定一个可能包含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。

**先排序再used判断可以将{2，3}，{3，2}直接去重掉。**
```cpp
vector<vector<int>> result;
vector<int> path;
void backtracking(vector<int>& nums, int startIndex, vector<bool>& used) {
    result.push_back(path);
    for (int i = startIndex; i < nums.size(); i++) {
        // used[i - 1] == true，说明同一树枝candidates[i - 1]使用过
        // used[i - 1] == false，说明同一树层candidates[i - 1]使用过
        // 而我们要对同一树层使用过的元素进行跳过
        if (i > 0 && nums[i] == nums[i - 1] && used[i - 1] == false) {
            continue;
        }
        path.push_back(nums[i]);
        used[i] = true;
        backtracking(nums, i + 1, used);
        used[i] = false;
        path.pop_back();
    }
}
vector<vector<int>> subsetsWithDup(vector<int>& nums) {
    result.clear();
    path.clear();
    vector<bool> used(nums.size(), false);
    sort(nums.begin(), nums.end()); // 去重需要排序
    backtracking(nums, 0, used);
    return result;
}
```

### 递增子序列
Leetcode 491. 给定一个整型数组, 你的任务是找到所有该数组的递增子序列，递增子序列的长度至少是2。

**这里要注意不能进行排序，但是有一点好处是它要求递增子序列，相对于组合问题，隐含着做了一步去重，这里我们只需要在每一层定义一次unordered_set，确保同一父节点下不能重复使用即可。**

```cpp
vector<vector<int>> result;
vector<int> path;
void backtracking(vector<int>& nums, int startIndex) {
    if (path.size() > 1) {
        result.push_back(path);
        // 注意这里不要加return，要取树上的节点
    }
    unordered_set<int> uset; // 使用set对本层元素进行去重
    for (int i = startIndex; i < nums.size(); i++) {
        if ((!path.empty() && nums[i] < path.back())
                || uset.find(nums[i]) != uset.end()) {
                continue;
        }
        uset.insert(nums[i]); // 记录这个元素在本层用过了，本层后面不能再用了
        path.push_back(nums[i]);
        backtracking(nums, i + 1);
        path.pop_back();
    }
}
vector<vector<int>> findSubsequences(vector<int>& nums) {
    result.clear();
    path.clear();
    backtracking(nums, 0);
    return result;
}
```

## 排列问题

### 全排列I
Leetcode46. 给定一个 没有重复 数字的序列，返回其所有可能的全排列。

**处理排列问题就不用使用startIndex了。但排列问题需要一个used数组，标记已经选择的元素。**

```cpp
vector<vector<int>> result;
vector<int> path;
void backtracking (vector<int>& nums, vector<bool>& used) {
    // 此时说明找到了一组
    if (path.size() == nums.size()) {
        result.push_back(path);
        return;
    }
    for (int i = 0; i < nums.size(); i++) {
        if (used[i] == true) continue; // path里已经收录的元素，直接跳过
        used[i] = true;
        path.push_back(nums[i]);
        backtracking(nums, used);
        path.pop_back();
        used[i] = false;
    }
}
vector<vector<int>> permute(vector<int>& nums) {
    result.clear();
    path.clear();
    vector<bool> used(nums.size(), false);
    backtracking(nums, used);
    return result;
}
```

### 全排列II
Leetcode 47. 给定一个可包含重复数字的序列 nums ，按任意顺序 返回所有不重复的全排列。

**注意去重一定要对元素进行排序，这样我们才方便通过相邻的节点来判断是否重复使用了。**

```cpp
vector<vector<int>> result;
vector<int> path;
void backtracking (vector<int>& nums, vector<bool>& used) {
    // 此时说明找到了一组
    if (path.size() == nums.size()) {
        result.push_back(path);
        return;
    }
    for (int i = 0; i < nums.size(); i++) {
        // used[i - 1] == true，说明同一树枝nums[i - 1]使用过
        // used[i - 1] == false，说明同一树层nums[i - 1]使用过
        // 如果同一树层nums[i - 1]使用过则直接跳过
        if (i > 0 && nums[i] == nums[i - 1] && used[i - 1] == false) {
            continue;
        }
        if (used[i] == false) {
            used[i] = true;
            path.push_back(nums[i]);
            backtracking(nums, used);
            path.pop_back();
            used[i] = false;
        }
    }
}
vector<vector<int>> permuteUnique(vector<int>& nums) {
    result.clear();
    path.clear();
    sort(nums.begin(), nums.end()); // 排序
    vector<bool> used(nums.size(), false);
    backtracking(nums, used);
    return result;
}
```

**其实以上所有问题在涉及到去重的时候都可以使用unordered_set来实现，但是这样做的话空间复杂度就从用used数组的$O(n)$变为$O(n^2)$了。**

## 棋盘问题

这里涉及到什么时候backTrack需要返回值，什么时候不需要：**当需要遍历整棵树，找到所有可行的叶子节点时不需要返回值，而如果只需要找到一种可行树枝的话就需要一个bool返回值，如果不及时return的话回溯会把找到的结果清空。** 以下三个问题可以作为例子

### n皇后问题
Leetcode 51. n皇后问题研究的是如何将n个皇后放置在n×n的棋盘上，并且使皇后彼此之间不能相互攻击。

**本质上就是回溯算法，只不过每一步需要isValid进行一次判断：**
```cpp
vector<vector<string>> result;
// n 为输入的棋盘大小
// row 是当前递归到棋盘的第几行了
void backtracking(int n, int row, vector<string>& chessboard) {
    if (row == n) {
        result.push_back(chessboard);
        return;
    }
    for (int col = 0; col < n; col++) {
        if (isValid(row, col, chessboard, n)) { // 验证合法就可以放
            chessboard[row][col] = 'Q'; // 放置皇后
            backtracking(n, row + 1, chessboard);
            chessboard[row][col] = '.'; // 回溯，撤销皇后
        }
    }
}
bool isValid(int row, int col, vector<string>& chessboard, int n) {
    // 检查列
    for (int i = 0; i < row; i++) { // 这是一个剪枝
        if (chessboard[i][col] == 'Q') {
            return false;
        }
    }
    // 检查 45度角是否有皇后
    for (int i = row - 1, j = col - 1; i >=0 && j >= 0; i--, j--) {
        if (chessboard[i][j] == 'Q') {
            return false;
        }
    }
    // 检查 135度角是否有皇后
    for(int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++) {
        if (chessboard[i][j] == 'Q') {
            return false;
        }
    }
    return true;
}
vector<vector<string>> solveNQueens(int n) {
    result.clear();
    std::vector<std::string> chessboard(n, std::string(n, '.'));
    backtracking(n, 0, chessboard);
    return result;
}
```

### 数独问题
Leetcode 37. 编写一个程序，通过填充空格来解决数独问题。

棋盘搜索问题可以使用回溯法暴力搜索，只不过这次我们要做的是二维递归。N皇后问题是一维递归，因为每一行每一列只放一个皇后，只需要一层for循环遍历一行，递归来遍历列，然后一行一列确定皇后的唯一位置。

这个问题中递归函数的返回值需要是bool类型，因为解数独找到一个符合的条件（就在树的叶子节点上）立刻就返回，相当于找**从根节点到叶子节点一条唯一路径，所以需要使用bool返回值。**

```cpp
bool backtracking(vector<vector<char>>& board) {
    for (int i = 0; i < board.size(); i++) {        // 遍历行
        for (int j = 0; j < board[0].size(); j++) { // 遍历列
            if (board[i][j] == '.') {
                for (char k = '1'; k <= '9'; k++) {     // (i, j) 这个位置放k是否合适
                    if (isValid(i, j, k, board)) {
                        board[i][j] = k;                // 放置k
                        if (backtracking(board)) return true; // 如果找到合适一组立刻返回
                        board[i][j] = '.';              // 回溯，撤销k
                    }
                }
                return false;  // 9个数都试完了，都不行，那么就返回false
            }
        }
    }
    return true; // 遍历完没有返回false，说明找到了合适棋盘位置了
}
bool isValid(int row, int col, char val, vector<vector<char>>& board) {
    for (int i = 0; i < 9; i++) { // 判断行里是否重复
        if (board[row][i] == val) {
            return false;
        }
    }
    for (int j = 0; j < 9; j++) { // 判断列里是否重复
        if (board[j][col] == val) {
            return false;
        }
    }
    int startRow = (row / 3) * 3;
    int startCol = (col / 3) * 3;
    for (int i = startRow; i < startRow + 3; i++) { // 判断9方格里是否重复
        for (int j = startCol; j < startCol + 3; j++) {
            if (board[i][j] == val ) {
                return false;
            }
        }
    }
    return true;
}
void solveSudoku(vector<vector<char>>& board) {
    backtracking(board);
}
```

### 重新规划行程
Leetcode 332. 给定一个机票的字符串二维数组 [from, to]，子数组中的两个成员分别表示飞机出发和降落的机场地点，对该行程进行重新规划排序，如果有多种排序方法只返回字典序最大的一个。所有这些机票都属于一个从 JFK（肯尼迪国际机场）出发的先生，所以该行程必须从 JFK 开始。

这道题目有几个难点：

1. 一个行程中，如果航班处理不好容易变成一个圈，成为死循环。
    例如[ ["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"] ]构成了一个环，如果在回溯遍历一个机票的时候不去将其删除就会构成死循环。
2. 有多种解法，字母序靠前排在前面，如何该记录映射关系呢？
    一个机场映射多个机场，机场之间要靠字母序排列，一个机场映射多个机场，可以使用std::unordered_map，如果让多个机场之间再有顺序的话，就是用std::map 或者std::multimap 或者 std::multiset。但是multiset在删除元素的时候会导致迭代器失效，因此**我们还是要选用unordered_map<string,map<string,int>>。**
3. 使用回溯法（也可以说深搜） 的话，那么终止条件是什么呢？
   终止条件就是ans数组的大小等于机票个数+1了。
4. 搜索的过程中，如何遍历一个机场所对应的所有机场
   记unordered_map为Map，那么遍历Map[from]中的每一个pair就可以了。

**这里还要注意我们只要找一个分支即可，因此递归函数需要一个bool返回值。**

```cpp
// unordered_map<出发机场, map<到达机场, 航班次数>> targets
unordered_map<string, map<string, int>> targets;
bool backtracking(int ticketNum, vector<string>& result) {
    if (result.size() == ticketNum + 1) {
        return true;
    }
    for (pair<const string, int>& target : targets[result[result.size() - 1]]) {
        if (target.second > 0 ) { // 记录到达机场是否飞过了
            result.push_back(target.first);
            target.second--;
            if (backtracking(ticketNum, result)) return true;
            result.pop_back();
            target.second++;
        }
    }
    return false;
}
vector<string> findItinerary(vector<vector<string>>& tickets) {
    targets.clear();
    vector<string> result;
    for (const vector<string>& vec : tickets) {
        targets[vec[0]][vec[1]]++; // 记录映射关系
    }
    result.push_back("JFK"); // 起始机场
    backtracking(tickets.size(), result);
    return result;
}
```
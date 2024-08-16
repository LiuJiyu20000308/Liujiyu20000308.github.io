---
layout: post
title: Leetcode记录：单调栈
date: 2024-8-12 14:00 +0800
tags: [数据结构与算法]
toc: true
---

单调栈通常是一维数组，**要寻找任一个元素的右边或者左边第一个比自己大或者小的元素的位置**，此时我们就要想到可以用单调栈了。时间复杂度为O(n)。

单调栈的本质是**空间换时间**，因为在遍历的过程中需要用一个栈来记录右边第一个比当前元素高的元素，优点是整个数组只需要遍历一次。
更直白来说，就是用一个栈来记录我们遍历过的元素，因为我们遍历数组的时候，我们不知道之前都遍历了哪些元素，以至于遍历一个元素找不到是不是之前遍历过一个更小的，所以我们需要用一个容器（这里用单调栈）来记录我们遍历过的元素。

在使用单调栈的时候首先要明确如下几点：

1. 单调栈里存放的元素是什么？
    单调栈里只需要存放元素的下标i就可以了，如果需要使用对应的元素，直接T[i]就可以获取。
2. 单调栈里元素是递增呢？ 还是递减呢？
   如果求一个元素右边第一个更大元素，单调栈就是（栈顶到栈底）递增的，如果求一个元素右边第一个更小元素，单调栈就是递减的。

使用单调栈主要有三个判断条件:

1. 当前遍历的元素T[i]小于栈顶元素T[st.top()]的情况
2. 当前遍历的元素T[i]等于栈顶元素T[st.top()]的情况
3. 当前遍历的元素T[i]大于栈顶元素T[st.top()]的情况

我们以下题为例讲解：
### 每日温度
给定一个整数数组 temperatures ，表示每天的温度，返回一个数组 answer ，其中 answer[i] 是指对于第 i 天，下一个更高温度出现在几天后。如果气温在这之后都不会升高，请在该位置用 0 来代替。

我们用temperatures = [73, 74, 75, 71, 71, 72, 76, 73]为例来逐步分析：
1. 首先将第一个遍历元素加入单调栈
2. 加入T[1] = 74，因为T[1] > T[0]（**当前遍历的元素T[i]大于栈顶元素T[st.top()]的情况**）。
    我们要保持一个递增单调栈（从栈头到栈底），所以将T[0]弹出，T[1]加入，此时result数组可以记录了，result[0] = 1，即T[0]右面第一个比T[0]大的元素是T[1]。
3. 加入T[2]，同理，T[1]弹出
4. 加入T[3]，T[3] < T[2] （**当前遍历的元素T[i]小于栈顶元素T[st.top()]的情况**），加T[3]加入单调栈。
5. 加入T[4]，T[4] == T[3] （当前遍历的元素T[i]等于栈顶元素T[st.top()]的情况），此时依然要加入栈，不用计算距离，**因为我们要求的是右面第一个大于本元素的位置，而不是大于等于！**
6. 加入T[5]，T[5] > T[4] （当前遍历的元素T[i]大于栈顶元素T[st.top()]的情况），将T[4]弹出，同时计算距离，更新result
7. T[4]弹出之后， T[5] > T[3] （当前遍历的元素T[i]大于栈顶元素T[st.top()]的情况），将T[3]继续弹出，同时计算距离，更新result
8. 直到发现T[5]小于T[st.top()]，终止弹出，将T[5]加入单调栈
9. 加入T[6]，同理，需要将栈里的T[5]，T[2]弹出
10. 加入T[7]， T[7] < T[6] 直接入栈，这就是最后的情况，result数组也更新完了。
11. 注意：**定义result数组的时候，就应该直接初始化为0，如果result没有更新，说明这个元素右面没有更大的了，也就是为0。**

逻辑详细版本的C++代码如下：
```cpp
vector<int> dailyTemperatures(vector<int>& T) {
    // 递增栈
    stack<int> st;
    vector<int> result(T.size(), 0);
    st.push(0);
    for (int i = 1; i < T.size(); i++) {
        if (T[i] < T[st.top()]) {                       // 情况一
            st.push(i);
        } else if (T[i] == T[st.top()]) {               // 情况二
            st.push(i);
        } else {
            while (!st.empty() && T[i] > T[st.top()]) { // 情况三
                result[st.top()] = i - st.top();
                st.pop();
            }
            st.push(i);
        }
    }
    return result;
}
```
精简后为：
```cpp
vector<int> dailyTemperatures(vector<int>& T) {
    stack<int> st; // 递增栈
    vector<int> result(T.size(), 0);
    for (int i = 0; i < T.size(); i++) {
        while (!st.empty() && T[i] > T[st.top()]) { // 注意栈不能为空
            result[st.top()] = i - st.top();
            st.pop();
        }
        st.push(i);

    }
    return result;
}
```

### 最大二叉树

Leetcode 654. 给定一个不重复的整数数组 nums 。 最大二叉树 可以用下面的算法从 nums 递归地构建:
1. 创建一个根节点，其值为 nums 中的最大值。
2. 递归地在最大值 左边 的 子数组前缀上 构建左子树。
3. 递归地在最大值 右边 的 子数组后缀上 构建右子树。

返回 nums 构建的 最大二叉树 。

递归方法很好理解：
```cpp
TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
    return construct(nums, 0, nums.size() - 1);
}

TreeNode* construct(const vector<int>& nums, int left, int right) {
    if (left > right) {
        return nullptr;
    }
    int best = left;
    for (int i = left + 1; i <= right; ++i) {
        if (nums[i] > nums[best]) {
            best = i;
        }
    }
    TreeNode* node = new TreeNode(nums[best]);
    node->left = construct(nums, left, best - 1);
    node->right = construct(nums, best + 1, right);
    return node;
}
```

第二种方法可以使用单调栈：在最终构造出的树上，以 nums[i] 为根节点的子树，在原数组中对应的区间，**左边界为 nums[i] 左侧第一个比它大的元素所在的位置，右边界为 nums[i] 右侧第一个比它大的元素所在的位置。左右边界均为开边界。**
如果某一侧边界不存在，则那一侧边界为数组的边界。如果两侧边界均不存在，说明其为最大值，即根节点。

因此，我们的任务变为：**找出每一个元素左侧和右侧第一个比它大的元素所在的位置。**

**如何求左侧第一个更大的元素的位置也体现在此了**
```cpp
TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
    int n = nums.size();
    vector<int> stk;
    vector<int> left(n, -1), right(n, -1);
    vector<TreeNode*> tree(n);
    for (int i = 0; i < n; ++i) {
        tree[i] = new TreeNode(nums[i]);
        while (!stk.empty() && nums[i] > nums[stk.back()]) {
            right[stk.back()] = i;
            stk.pop_back();
        }
        if (!stk.empty()) {
            left[i] = stk.back();
        }
        stk.push_back(i);
    }

    TreeNode* root = nullptr;
    for (int i = 0; i < n; ++i) {
        if (left[i] == -1 && right[i] == -1) {
            root = tree[i];
        }
        else if (right[i] == -1 || (left[i] != -1 && nums[left[i]] < nums[right[i]])) {
            tree[left[i]]->right = tree[i];
        }
        else {
            tree[right[i]]->left = tree[i];
        }
    }
    return root;
}
```
---
layout: post
title: Leetcode记录：二叉树遍历
date: 2024-8-10 14:00 +0800
tags: [数据结构与算法]
toc: true
---
## 二叉树理论基础

### 二叉树的种类

二叉树有两种主要的形式：满二叉树和完全二叉树：

1. 满二叉树：如果一棵二叉树只有度为0的结点和度为2的结点，并且度为0的结点在同一层上，则这棵二叉树为满二叉树。
    如图，这棵二叉树为满二叉树，也可以说深度为$k$，有$2^k-1$个节点的二叉树:

    <div align="center"> <img src="/pic/DS/FullBinaryTree.png" width = 300/> </div>

2. 完全二叉树：在完全二叉树中，除了最底层节点可能没填满外，其余每层节点数都达到最大值，并且最下面一层的节点都集中在该层最左边的若干位置。若最底层为第$h$层（$h$从1开始），则该层包含 $1 \sim 2^(h-1)$ 个节点。 
   <div align="center"> <img src="/pic/DS/CompleteBinaryTree.png" width = 400/> </div>

    之前我们刚刚讲过优先级队列其实是一个堆，**堆就是一棵完全二叉树，同时保证父子节点的顺序关系。**

### 二叉搜索树

二叉搜索树是一个有序树：

1. 若它的左子树不空，则左子树上所有结点的值均小于它的根结点的值；
2. 若它的右子树不空，则右子树上所有结点的值均大于它的根结点的值；
3. 它的左、右子树也分别为二叉搜素树

<div align="center"> <img src="/pic/DS/BST.png" width = 400/> </div>

根据二叉搜索树的性质，左节点的值 < 根节点的值 < 右节点的值，我们很容易想到，对二叉搜索树进行中序遍历就可以得到一个节点值递增的序列。因此我们也将二叉搜索树称为二叉排序树。

寻找插入删除可以见下文：[BST理论+代码](https://blog.csdn.net/smf12138/article/details/126092714)，注意它代码第67行写错了，应该是
```cpp
cur->_key = minRight->_key;
```

### 平衡二叉搜索树

平衡二叉搜索树：又被称为AVL（Adelson-Velsky and Landis）树，且具有以下性质：它是一棵空树或它的左右两个子树的高度差的绝对值不超过1，并且左右两个子树都是一棵平衡二叉树。

<div align="center"> <img src="/pic/DS/AVLTree.png" width = 400/> </div>

**C++中map、set、multimap，multiset的底层实现都是平衡二叉搜索树，所以map、set的增删操作时间时间复杂度是O(logn).**

### 二叉树存储方式

二叉树可以链式存储，也可以顺序存储。链式存储方式就用指针， 顺序存储的方式就是用数组，如果父节点的数组下标是$i$，那么它的左孩子就是$2i +1$，右孩子就是 $2i + 2$。


### 二叉树遍历方式

二叉树遍历方式有：
1. 深度优先遍历
   1. 前序遍历：中左右（递归法，迭代法）
   2. 中序遍历：左中右（递归法，迭代法）
   3. 后序遍历：左右中（递归法，迭代法）
2. 广度优先遍历
   1. 层次遍历（迭代法）

我们经常会使用递归的方式来实现深度优先遍历，也就是实现前中后序遍历，使用递归是比较方便的。
之前讲栈与队列的时候，就说过栈其实就是递归的一种实现结构，也就说前中后序遍历的逻辑其实都是可以借助栈使用递归的方式来实现的。

而广度优先遍历的实现一般使用队列来实现，这也是队列先进先出的特点所决定的，因为需要先进先出的结构，才能一层一层的来遍历二叉树。


## 二叉树代码实现

### 二叉树递归遍历

每次写递归，都按照这三要素来写，可以保证大家写出正确的递归算法：
1. **确定递归函数的参数和返回值：** 确定哪些参数是递归的过程中需要处理的，那么就在递归函数里加上这个参数， 并且还要明确每次递归的返回值是什么进而确定递归函数的返回类型。

2. **确定终止条件：** 写完了递归算法, 运行的时候，经常会遇到栈溢出的错误，就是没写终止条件或者终止条件写的不对，操作系统也是用一个栈的结构来保存每一层递归的信息，如果递归没有终止，操作系统的内存栈必然就会溢出。

3. **确定单层递归的逻辑：** 确定每一层递归需要处理的信息。在这里也就会重复调用自己来实现递归的过程。

#### 前序遍历

```cpp
void traversal(TreeNode* cur, vector<int>& vec) {
    if (cur == NULL) return;
    vec.push_back(cur->val);    // 中
    traversal(cur->left, vec);  // 左
    traversal(cur->right, vec); // 右
}
vector<int> preorderTraversal(TreeNode* root) {
    vector<int> result;
    traversal(root, result);
    return result;
}
```

#### 中序遍历

```cpp
void traversal(TreeNode* cur, vector<int>& vec) {
    if (cur == NULL) return;
    traversal(cur->left, vec);  // 左
    vec.push_back(cur->val);    // 中
    traversal(cur->right, vec); // 右
}
```

#### 后序遍历

```cpp
void traversal(TreeNode* cur, vector<int>& vec) {
    if (cur == NULL) return;
    traversal(cur->left, vec);  // 左
    traversal(cur->right, vec); // 右
    vec.push_back(cur->val);    // 中
}
```

### 二叉树迭代遍历

为什么可以用迭代法（非递归的方式）来实现二叉树的前后中序遍历呢？我们提到过：递归的实现就是**每一次递归调用都会把函数的局部变量、参数值和返回地址等压入调用栈中，然后递归返回的时候，从栈顶弹出上一次递归的各项参数**，所以这就是递归为什么可以返回上一层位置的原因。因此用栈也可以实现二叉树的前后中序遍历。

#### 前序遍历
前序遍历是中左右，每次先处理的是中间节点，那么先将根节点放入栈中，然后将右孩子加入栈，再加入左孩子。
**注意要先放入右子节点，因为这样出栈的时候才是先左后右。**

<div align="center"> <img src="/pic/DS/preOrderBST.gif" width = 400/> </div>

```cpp
vector<int> preorderTraversal(TreeNode* root) {
    stack<TreeNode*> st;
    vector<int> result;
    if (root == NULL) return result;
    st.push(root);
    while (!st.empty()) {
        TreeNode* node = st.top();                       // 中
        st.pop();
        result.push_back(node->val);
        if (node->right) st.push(node->right);           // 右（空节点不入栈）
        if (node->left) st.push(node->left);             // 左（空节点不入栈）
    }
    return result;
}
```
但接下来，再用迭代法写中序遍历的时候，会发现套路又不一样了，目前的前序遍历的逻辑无法直接应用到中序遍历上。

#### 中序遍历

因为前序遍历的顺序是中左右，先访问的元素是中间节点，要处理的元素也是中间节点，所以刚刚才能写出相对简洁的代码，**因为要访问的元素和要处理的元素顺序是一致的，都是中间节点。**

那么再看看中序遍历，中序遍历是左中右，先访问的是二叉树顶部的节点，然后一层一层向下访问，**直到到达树左面的最底部，再开始处理节点**（也就是在把节点的数值放进result数组中），这就造成了**处理顺序和访问顺序是不一致的。**

那么在使用迭代法写中序遍历，就需要**借用指针的遍历来帮助访问节点，栈则用来处理节点上的元素。**

<div align="center"> <img src="/pic/DS/inOrderBST.gif" width = 400/> </div>

```cpp
vector<int> inorderTraversal(TreeNode* root) {
    vector<int> result;
    stack<TreeNode*> st;
    TreeNode* cur = root;
    while (cur != NULL || !st.empty()) {
        if (cur != NULL) { // 指针来访问节点，访问到最底层
            st.push(cur); // 将访问的节点放进栈
            cur = cur->left;                // 左
        } else {
            cur = st.top(); // 从栈里弹出的数据，就是要处理的数据（放进result数组里的数据）
            st.pop();
            result.push_back(cur->val);     // 中
            cur = cur->right;               // 右
        }
    }
    return result;
}
```

#### 后序遍历

先序遍历是中左右，后序遍历是左右中，**那么我们只需要调整一下先序遍历的代码顺序，就变成中右左的遍历顺序，然后在反转result数组**，输出的结果顺序就是左右中了。

```cpp
vector<int> postorderTraversal(TreeNode* root) {
    stack<TreeNode*> st;
    vector<int> result;
    if (root == NULL) return result;
    st.push(root);
    while (!st.empty()) {
        TreeNode* node = st.top();
        st.pop();
        result.push_back(node->val);
        if (node->left) st.push(node->left); // 相对于前序遍历，这更改一下入栈顺序 （空节点不入栈）
        if (node->right) st.push(node->right); // 空节点不入栈
    }
    reverse(result.begin(), result.end()); // 将结果反转之后就是左右中的顺序了
    return result;
}
```

### 统一格式迭代

我们以中序遍历为例，使用栈的话，无法同时解决访问节点（遍历节点）和处理节点（将元素放进结果集）不一致的情况。那我们就将访问的节点放入栈中，把要处理的节点也放入栈中但是要做标记：就是**要处理的节点放入栈之后，紧接着放入一个空指针作为标记**，这种方法也可以叫做**标记法。**

如图所示，可以看出我们将访问的节点直接加入到栈中，但如果是处理的节点则后面放入一个空节点， 这样只有空节点弹出的时候，才将下一个节点放进结果集：

<div align="center"> <img src="/pic/DS/inOrderBST_Iter.gif" width = 400/> </div>

**同时要注意栈是先进后出：**
```cpp
vector<int> inorderTraversal(TreeNode* root) {
    vector<int> result;
    stack<TreeNode*> st;
    if (root != NULL) st.push(root);
    while (!st.empty()) {
        TreeNode* node = st.top();
        if (node != NULL) {
            st.pop(); // 将该节点弹出，避免重复操作，下面再将右中左节点添加到栈中
            if (node->right) st.push(node->right);  // 添加右节点（空节点不入栈）

            st.push(node);                          // 添加中节点
            st.push(NULL); // 中节点访问过，但是还没有处理，加入空节点做为标记。

            if (node->left) st.push(node->left);    // 添加左节点（空节点不入栈）
        } else { // 只有遇到空节点的时候，才将下一个节点放进结果集
            st.pop();           // 将空节点弹出
            node = st.top();    // 重新取出栈中元素
            st.pop();
            result.push_back(node->val); // 加入到结果集
        }
    }
    return result;
}

vector<int> preorderTraversal(TreeNode* root) {
    vector<int> result;
    stack<TreeNode*> st;
    if (root != NULL) st.push(root);
    while (!st.empty()) {
        TreeNode* node = st.top();
        if (node != NULL) {
            st.pop();
            if (node->right) st.push(node->right);  // 右
            if (node->left) st.push(node->left);    // 左
            st.push(node);                          // 中
            st.push(NULL);
        } else {
            st.pop();
            node = st.top();
            st.pop();
            result.push_back(node->val);
        }
    }
    return result;
}

vector<int> postorderTraversal(TreeNode* root) {
    vector<int> result;
    stack<TreeNode*> st;
    if (root != NULL) st.push(root);
    while (!st.empty()) {
        TreeNode* node = st.top();
        if (node != NULL) {
            st.pop();
            st.push(node);                          // 中
            st.push(NULL);

            if (node->right) st.push(node->right);  // 右
            if (node->left) st.push(node->left);    // 左

        } else {
            st.pop();
            node = st.top();
            st.pop();
            result.push_back(node->val);
        }
    }
    return result;
}
```

### 二叉树层级遍历

需要借用一个辅助数据结构即队列来实现，队列先进先出，符合一层一层遍历的逻辑，而用栈先进后出适合模拟深度优先遍历也就是递归的逻辑。而这种层序遍历方式就是图论中的广度优先遍历，只不过我们应用在二叉树上。

使用队列实现二叉树广度优先遍历，动画如下：

<div align="center"> <img src="/pic/DS/levelOrder.gif" width = 400/> </div>

```cpp
vector<vector<int>> levelOrder(TreeNode* root) {
    queue<TreeNode*> que;
    if (root != NULL) que.push(root);
    vector<vector<int>> result;
    while (!que.empty()) {
        int size = que.size();
        vector<int> vec;
        // 这里一定要使用固定大小size，不要使用que.size()，因为que.size是不断变化的
        for (int i = 0; i < size; i++) {
            TreeNode* node = que.front();
            que.pop();
            vec.push_back(node->val);
            if (node->left) que.push(node->left);
            if (node->right) que.push(node->right);
        }
        result.push_back(vec);
    }
    return result;
}
```

迭代法如下，这样做很多人也叫先序遍历以及dfs：
```cpp
void order(TreeNode* cur, vector<vector<int>>& result, int depth)
{
    if (cur == nullptr) return;
    if (result.size() == depth) result.push_back(vector<int>());
    result[depth].push_back(cur->val);
    order(cur->left, result, depth + 1);
    order(cur->right, result, depth + 1);
}
vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> result;
    int depth = 0;
    order(root, result, depth);
    return result;
}
```

### 相关题目


Leetcode 144.二叉树的前序遍历
Leetcode 94. 二叉树的中序遍历
Leetcode 145.二叉树的后序遍历
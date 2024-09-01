---
layout: post
title: Leetcode记录：并查集
date: 2024-8-30 8:00 +0800
tags: [数据结构与算法]
toc: true
---

## 并查集原理

并查集常用来解决连通性问题。并查集主要有两个功能：
1. 将两个元素添加到一个集合中；
2. 判断两个元素在不在同一个集合。

我们用一个一维数组来表示，假设我们将三个元素A，B，C （分别是数字）放在同一个集合，**那么只需要用father[A] = B，father[B] = C 这样就表述 A 与 B 与 C连通了（有向连通图）：**

```cpp
// 将v->u 这条边加入并查集
void join(int u, int v) {
    u = find(u); // 寻找u的根
    v = find(v); // 寻找v的根
    if (u == v) return; // 如果发现根相同，则说明在一个集合，不用两个节点相连直接返回
    father[v] = u;
}
// 并查集里寻根的过程
int find(int u) {
    if (u == father[u]) return u; // 如果根就是自己，直接返回
    else return find(father[u]); // 如果根不是自己，就根据数组下标一层一层向下找
}
```

如何表示 C 也在同一个元素里呢？ 我们需要 father[C] = C，即C的根也为C，这样就方便表示 A，B，C 都在同一个集合里了。所以**father数组初始化的时候要 father[i] = i，默认自己指向自己。**
```cpp
// 并查集初始化
void init() {
    for (int i = 0; i < n; ++i) {
        father[i] = i;
    }
}
```

最后我们如何判断两个元素是否在同一个集合里，如果通过 find函数 找到 两个元素属于同一个根的话，那么这两个元素就是同一个集合，代码如下：
```cpp
// 判断 u 和 v是否找到同一个根
bool isSame(int u, int v) {
    u = find(u);
    v = find(v);
    return u == v;
}
```

在实现find函数的过程中，我们知道，通过递归的方式，不断获取father数组下标对应的数值，最终找到这个集合的根。**搜索过程像是一个多叉树中从叶子到根节点的过程。**如果这棵多叉树高度很深的话，每次find函数去寻找根的过程就要递归很多次。我们的目的**只需要知道这些节点在同一个根下就可以，所以对这棵多叉树的构造只需要除了根节点其他所有节点都挂载根节点下就可以了，这样就是路径压缩：**
```cpp
int find(int u) {
    return u == father[u] ? u : father[u] = find(father[u]);
}
```

## 例题

### 寻找存在的路径
Leetcode 1971. 给定一个包含 n 个节点的无向图中，节点编号从 1 到 n （含 1 和 n ）。你的任务是判断是否有一条从节点 source 出发到节点 destination 的路径存在。

```cpp
vector<int> father;

int find(int a){
    return father[a] == a ? a : father[a] = find(father[a]);
}
void join(int a, int b){
    a = find(a);
    b = find(b);
    if (a == b) return;
    father[b] = a;
}

bool isSame(int a, int b){
    a = find(a);
    b = find(b);
    return a==b;
}
bool validPath(int n, vector<vector<int>>& edges, int source, int destination) {
    father.resize(n,0);
    for(int i = 0; i < n; ++i)
        father[i] = i;
    
    for(auto edge : edges){
        join(edge[0], edge[1]);
    }
    return isSame(source, destination);
}
```

### 冗余连接
Leetcode 684. 树可以看成是一个图（拥有 n 个节点和 n - 1 条边的连通无环无向图）。现给定一个拥有 n 个节点（节点编号从 1 到 n）和 n 条边的连通无向图，请找出一条可以删除的边，删除后图可以变成一棵树。

**只要加入的边的两个顶点已经在同一个集合里了，就说明这条边的加入会构成环，删掉即可：**
```cpp
vector<int> findRedundantConnection(vector<vector<int>>& edges) {
    int n = edges.size();
    father.resize(n+1,0);
    for(int i = 0; i <= n; ++i){
        father[i] = i;
    }
    for(auto edge : edges){
        if(isSame(edge[0], edge[1]))
            return edge;
        else
            join(edge[0], edge[1]);
    }
    return vector<int>{};
}
```

### 冗余连接II
Leetcode 685. 有一种有向树,该树只有一个根节点，所有其他节点都是该根节点的后继。该树除了根节点之外的每一个节点都有且只有一个父节点，而根节点没有父节点。有向树拥有 n 个节点和 n - 1 条边。

现在有一个有向图，有向图是在有向树中的两个没有直接链接的节点中间添加一条有向边。输入一个有向图，该图由一个有着 n 个节点(节点编号 从 1 到 n)，n 条边，请返回一条可以删除的边，使得删除该条边之后该有向图可以被当作一颗有向树。若有多个答案，返回最后出现在给定二维数组的答案。

思路：首先要找入度为2的顶点，如果存在的话尝试删除其中一边，看看是否可行，这里就**需要判断删除之后是否满足有向树的条件**；如果不存在的话则说明图中有环，**把环的最后一条边删除即可**。这两步都需要并查集来进行实现：

```cpp
vector<int> father;
int find(int a){
    return father[a] == a ? a : father[a] = find(father[a]);
}
void join(int a, int b){
    a = find(a);
    b = find(b);
    if (a == b) return;
    father[b] = a;
}

bool isSame(int a, int b){
    a = find(a);
    b = find(b);
    return a==b;
}

bool isTreeAfterDelete(vector<vector<int>>& edges, int deleteEdge){
    int n = edges.size();
    father.resize(n+1,0);
    for(int i = 0; i <= n; i++){
        father[i] = i;
    }
    for(int i = 0; i < n; ++i){
        if(i == deleteEdge)
            continue;
        if(isSame(edges[i][0], edges[i][1]))
            return false;
        join(edges[i][0], edges[i][1]);
    }
    return true;
}

vector<int> getRemove(vector<vector<int>>& edges){
    int n = edges.size();
    father.resize(n+1,0);
    for(int i = 0; i <= n; i++){
        father[i] = i;
    }
    for(int i = 0; i < n; ++i){
        if(isSame(edges[i][0], edges[i][1]))
            return edges[i];
        join(edges[i][0], edges[i][1]);
    }
    return {-1,-1};
}
vector<int> findRedundantDirectedConnection(vector<vector<int>>& edges) {
    int n = edges.size();
    vector<int> inDegree(n+1,0);
    for(auto &edge : edges){
        inDegree[edge[1]]++;
    } 
    vector<int> vec;
    for(int i = n-1; i >= 0; --i){
        if(inDegree[edges[i][1]] == 2){
            vec.push_back(i);
        }
    }
    if(!vec.empty()){
        if(isTreeAfterDelete(edges, vec[0])){
            return edges[vec[0]];
        }else
            return edges[vec[1]];
    }

    auto result = getRemove(edges);
    return result;
}
```
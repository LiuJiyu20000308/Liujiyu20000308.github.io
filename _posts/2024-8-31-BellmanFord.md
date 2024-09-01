---
layout: post
title: Leetcode记录：Bellman-ford算法
date: 2024-8-30 4:00 +0800
tags: [数据结构与算法]
toc: true
---

## Bellman-ford算法

某国为促进城市间经济交流，决定对货物运输提供补贴。共有 n 个编号为 1 到 n 的城市，通过道路网络连接，网络中的道路仅允许从某个城市单向通行到另一个城市，不能反向通行。网络中的道路都有各自的运输成本和政府补贴，**道路的权值计算方式为：运输成本 - 政府补贴。**权值为正表示扣除了政府补贴后运输货物仍需支付的费用；权值为负则表示政府的补贴超过了支出的运输成本，实际表现为运输过程中还能赚取一定的收益。

请找出从城市 1 到城市 n 的所有可能路径中，综合政府补贴后的最低运输成本。如果最低运输成本是一个负数，它表示在遵循最优路径的情况下，运输过程中反而能够实现盈利。城市 1 到城市 n 之间可能会出现没有路径的情况，**同时保证道路网络中不存在任何负权回路。**


Bellman-ford算法专门处理带负权值的单源最短路问题。**Bellman_ford算法的核心思想是 对所有边进行松弛n-1次操作（n为节点数量），从而求得目标最短路。**

### 松弛

假设minDist[B]表示到达B节点最小权值，A可以花费value到达B，minDist[B] 有哪些状态可以推出来？

状态一： minDist[A] + value 可以推出 minDist[B] 状态二： minDist[B]本身就有权值 （可能是其他边链接的节点B 例如节点C，以至于 minDist[B]记录了其他边到minDist[B]的权值）。

那么minDist[B]应做出取舍，应该取二者最小值，这就是松弛操作。**其实 Bellman_ford算法 也是采用了动态规划的思想，即：将一个问题分解成多个决策阶段，通过状态之间的递归关系最后计算出全局最优解。**

**对所有边松弛一次，相当于计算起点 到达 与起点一条边相连的 节点的最短距离**，因此要对所有边松弛n-1次，才能得到起点到达所有节点的最短距离。

```cpp
int main() {
    int n, m, p1, p2, val;
    cin >> n >> m;

    vector<vector<int>> grid;

    // 将所有边保存起来
    for(int i = 0; i < m; i++){
        cin >> p1 >> p2 >> val;
        // p1 指向 p2，权值为 val
        grid.push_back({p1, p2, val});

    }
    int start = 1;  // 起点
    int end = n;    // 终点

    vector<int> minDist(n + 1 , INT_MAX);
    minDist[start] = 0;
    for (int i = 1; i < n; i++) { // 对所有边 松弛 n-1 次
        for (vector<int> &side : grid) { // 每一次松弛，都是对所有边进行松弛
            int from = side[0]; // 边的出发点
            int to = side[1]; // 边的到达点
            int price = side[2]; // 边的权值
            // 松弛操作 
            // minDist[from] != INT_MAX 防止从未计算过的节点出发
            if (minDist[from] != INT_MAX && minDist[to] > minDist[from] + price) { 
                minDist[to] = minDist[from] + price;  
            }
        }
    }
    if (minDist[end] == INT_MAX) cout << "unconnected" << endl; // 不能到达终点
    else cout << minDist[end] << endl; // 到达终点最短路径

}
```
时间复杂度： O(N * E) , N为节点数量，E为图中边的数量
空间复杂度： O(N) ，即 minDist 数组所开辟的空间


## Bellman-ford队列优化算法

Bellman_ford 算法 每次都是对所有边进行松弛，其实是多做了一些无用功。**只需要对上一次松弛的时候更新过的节点作为出发节点所连接的边 进行松弛就够了。**

```cpp
struct Edge { //邻接表
    int to;  // 链接的节点
    int val; // 边的权重

    Edge(int t, int w): to(t), val(w) {}  // 构造函数
};


int main() {
    int n, m, p1, p2, val;
    cin >> n >> m;

    vector<list<Edge>> grid(n + 1); 

    vector<bool> isInQueue(n + 1); // 加入优化，已经在队里里的元素不用重复添加

    // 将所有边保存起来
    for(int i = 0; i < m; i++){
        cin >> p1 >> p2 >> val;
        // p1 指向 p2，权值为 val
        grid[p1].push_back(Edge(p2, val));
    }
    int start = 1;  // 起点
    int end = n;    // 终点

    vector<int> minDist(n + 1 , INT_MAX);
    minDist[start] = 0;

    queue<int> que;
    que.push(start); 

    while (!que.empty()) {

        int node = que.front(); que.pop();
        isInQueue[node] = false; // 从队列里取出的时候，要取消标记，我们只保证已经在队列里的元素不用重复加入
        for (Edge edge : grid[node]) {
            int from = node;
            int to = edge.to;
            int value = edge.val;
            if (minDist[to] > minDist[from] + value) { // 开始松弛
                minDist[to] = minDist[from] + value; 
                if (isInQueue[to] == false) { // 已经在队列里的元素不用重复添加
                    que.push(to);
                    isInQueue[to] = true;
                }
            }
        }

    }
    if (minDist[end] == INT_MAX) cout << "unconnected" << endl; // 不能到达终点
    else cout << minDist[end] << endl; // 到达终点最短路径
}
```

队列优化版Bellman_ford 的时间复杂度 并不稳定，效率高低依赖于图的结构。一般来说，SPFA 的时间复杂度为 O(K * N) K 为不定值，因为 节点需要计入几次队列取决于图的稠密度。如果图是一条线形图且单向的话，每个节点的入度为1，那么只需要加入一次队列，这样时间复杂度就是 O(N)。

所以 SPFA 在最坏的情况下是 O(N * E)，但 一般情况下 时间复杂度为 O(K * N)。

在有环且只有正权回路的情况下，即使元素重复加入队列，最后，也会因为 所有边都松弛后，节点数值（minDist数组）不在发生变化了 而终止。（而且有重复元素加入队列是正常的，多条路径到达同一个节点，节点必要要选择一个最短的路径，而这个节点就会重复加入队列进行判断，选一个最短的）

**但是如果有负权回路的话，就会出现死循环！**

## Bellman-ford判断负权回路

可以在做完n-1次松弛之后再做一次，查看minDist是否会继续变化，如果发生变化则代表存在负权回路：

```cpp
int main() {
    int n, m, p1, p2, val;
    cin >> n >> m;

    vector<vector<int>> grid;

    for(int i = 0; i < m; i++){
        cin >> p1 >> p2 >> val;
        // p1 指向 p2，权值为 val
        grid.push_back({p1, p2, val});

    }
    int start = 1;  // 起点
    int end = n;    // 终点

    vector<int> minDist(n + 1 , INT_MAX);
    minDist[start] = 0;
    bool flag = false;
    for (int i = 1; i <= n; i++) { // 这里我们松弛n次，最后一次判断负权回路
        for (vector<int> &side : grid) {
            int from = side[0];
            int to = side[1];
            int price = side[2];
            if (i < n) {
                if (minDist[from] != INT_MAX && minDist[to] > minDist[from] + price) minDist[to] = minDist[from] + price;
            } else { // 多加一次松弛判断负权回路
                if (minDist[from] != INT_MAX && minDist[to] > minDist[from] + price) flag = true;

            }
        }

    }

    if (flag) cout << "circle" << endl;
    else if (minDist[end] == INT_MAX) {
        cout << "unconnected" << endl;
    } else {
        cout << minDist[end] << endl;
    }
}
```

也可以用SFPA来做，多加一个count数组判断每个节点被加入了多少次队列，如果达到n次则说明存在负权回路：

```cpp
struct Edge { //邻接表
    int to;  // 链接的节点
    int val; // 边的权重

    Edge(int t, int w): to(t), val(w) {}  // 构造函数
};


int main() {
    int n, m, p1, p2, val;
    cin >> n >> m;

    vector<list<Edge>> grid(n + 1); // 邻接表

    // 将所有边保存起来
    for(int i = 0; i < m; i++){
        cin >> p1 >> p2 >> val;
        // p1 指向 p2，权值为 val
        grid[p1].push_back(Edge(p2, val));
    }
    int start = 1;  // 起点
    int end = n;    // 终点

    vector<int> minDist(n + 1 , INT_MAX);
    minDist[start] = 0;

    queue<int> que;
    que.push(start); // 队列里放入起点 
    
    vector<int> count(n+1, 0); // 记录节点加入队列几次
    count[start]++;

    bool flag = false;
    while (!que.empty()) {

        int node = que.front(); que.pop();

        for (Edge edge : grid[node]) {
            int from = node;
            int to = edge.to;
            int value = edge.val;
            if (minDist[to] > minDist[from] + value) { // 开始松弛
                minDist[to] = minDist[from] + value;
                que.push(to);
                count[to]++; 
                if (count[to] == n) {// 如果加入队列次数超过 n-1次 就说明该图与负权回路
                    flag = true;
                    while (!que.empty()) que.pop();
                    break;
                }
            }
        }
    }

    if (flag) cout << "circle" << endl;
    else if (minDist[end] == INT_MAX) {
        cout << "unconnected" << endl;
    } else {
        cout << minDist[end] << endl;
    }
}
```


## Bellman-ford之单源有限最短路径

例题：请计算在最多经过 k 个城市的条件下，从城市 src 到城市 dst 的最低运输成本。

前面讲Bellman-ford的时候提到过，对所有边松弛一次，相当于计算 起点到达 与起点**一条边**相连的节点 的最短距离。
最多经过k个城市就是k + 1条边相连的节点。因此我们对所有边松弛k+1次即可。

但是这道题可能会出现负权回路，如果我们只是简单地把松弛n-1次改成k+1次会出现错误：

<div align="center"> <img src="/pic/DS/Bellman-Ford.png" width = 400/> </div>

第二次松弛开始，每次松弛都会导致所有节点距离-1，这里要做的就是每次更新都要根据上一次的minDist来更新：

```cpp
int main() {
    int src, dst,k ,p1, p2, val ,m , n;
    
    cin >> n >> m;

    vector<vector<int>> grid;

    for(int i = 0; i < m; i++){
        cin >> p1 >> p2 >> val;
        grid.push_back({p1, p2, val});
    }

    cin >> src >> dst >> k;

    vector<int> minDist(n + 1 , INT_MAX);
    minDist[src] = 0;
    vector<int> minDist_copy(n + 1); // 用来记录上一次遍历的结果
    for (int i = 1; i <= k + 1; i++) {
        minDist_copy = minDist; // 获取上一次计算的结果
        for (vector<int> &side : grid) {
            int from = side[0];
            int to = side[1];
            int price = side[2];
            // 注意使用 minDist_copy 来计算 minDist 
            if (minDist_copy[from] != INT_MAX && minDist[to] > minDist_copy[from] + price) {  
                minDist[to] = minDist_copy[from] + price;
            }
        }
    }
    if (minDist[dst] == INT_MAX) cout << "unreachable" << endl; // 不能到达终点
    else cout << minDist[dst] << endl; // 到达终点最短路径

}
```


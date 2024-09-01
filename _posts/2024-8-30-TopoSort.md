---
layout: post
title: Leetcode记录：拓扑排序
date: 2024-8-30 9:00 +0800
tags: [数据结构与算法]
toc: true
---


例子：某个大型软件项目的构建系统拥有 N 个文件，文件编号从 0 到 N - 1，在这些文件中，某些文件依赖于其他文件的内容，这意味着如果文件 A 依赖于文件 B，则必须在处理文件 A 之前处理文件 B （0 <= A, B <= N - 1）。请编写一个算法，用于确定文件处理的顺序。

概括来说，**给出一个有向图，把这个有向图转成线性的排序就叫拓扑排序。**

当然拓扑排序也要检测这个有向图 是否有环，即存在循环依赖的情况，因为这种情况是不能做线性排序的。**所以拓扑排序也是图论中判断有向无环图的常用方法。**

实现拓扑排序的算法有两种：**卡恩算法（BFS）和DFS**，接下来我们来讲解BFS的实现思路：当我们做拓扑排序的时候，应该优先找入度为0的节点，只有入度为0，它才是出发节点。之后将其去掉，再寻找入度为0的节点即可。

因此，拓扑排序的过程，其实就两步：
1. 找到入度为0 的节点，加入结果集
2. 将该节点从图中移除

**如果我们发现结果集元素个数不等于图中节点个数，我们就可以认定图中一定有有向环！这也是拓扑排序判断有向环的方法。**

```cpp
int main() {
    int m, n, s, t;
    cin >> n >> m;
    vector<int> inDegree(n, 0); // 记录每个文件的入度

    unordered_map<int, vector<int>> umap;// 记录文件依赖关系
    vector<int> result; // 记录结果

    while (m--) {
        // s->t，先有s才能有t
        cin >> s >> t;
        inDegree[t]++; // t的入度加一
        umap[s].push_back(t); // 记录s指向哪些文件
    }

    queue<int> que;
    for (int i = 0; i < n; i++) {
        // 入度为0的文件，可以作为开头，先加入队列
        if (inDegree[i] == 0) que.push(i);
        //cout << inDegree[i] << endl;
    }
    // int count = 0;
    while (que.size()) {
        int cur = que.front(); // 当前选中的文件
        que.pop();
        //count++;
        result.push_back(cur);
        vector<int> files = umap[cur]; //获取该文件指向的文件
        if (files.size()) { // cur有后续文件
            for (int i = 0; i < files.size(); i++) {
                inDegree[files[i]] --; // cur的指向的文件入度-1
                if(inDegree[files[i]] == 0) que.push(files[i]);
            }
        }
    }
    if (result.size() == n) {
        for (int i = 0; i < n - 1; i++) cout << result[i] << " ";
        cout << result[n - 1];
    } else cout << -1 << endl;
}
```
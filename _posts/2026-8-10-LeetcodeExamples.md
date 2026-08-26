---
layout: post
title: Leetcode记录：全部例题直观解释与代码执行步骤
date: 2026-8-10 15:20 +0800
tags: [数据结构与算法]
toc: true
---

这篇文章汇总现有 LeetCode 专题的全部例题。每道题先给出直观理解和执行步骤，再直接附上完整的 C++ 与 Python 实现；原专题文章仍保留更详细的推导。

## 原代码索引

- 基础数据结构：[二分与双指针]({% post_url 2024-7-30-DS %})、[滑动窗口与前缀和]({% post_url 2024-8-1-DS %})、[模拟过程]({% post_url 2024-8-1-DS-2 %})、[链表]({% post_url 2024-8-2-DS %})、[哈希表 I]({% post_url 2024-8-3-DS %})、[哈希表 II]({% post_url 2024-8-4-DS %})、[字符串]({% post_url 2024-8-6-DS %})、[KMP]({% post_url 2024-8-7-KMP %})、[栈与队列]({% post_url 2024-8-8-DS %})、[单调队列]({% post_url 2024-8-9-Queue %})。
- 排序与缓存：[排序算法]({% post_url 2024-8-10-Sort %})、[缓存设计]({% post_url 2026-8-10-Cache %})。
- 树：[二叉树遍历]({% post_url 2024-8-11-BinaryTree %})、[二叉树例题]({% post_url 2024-8-12-BT %})、[二叉搜索树]({% post_url 2024-8-15-BST %})、[单调栈]({% post_url 2024-8-16-MonotonicStack %})。
- 搜索与规划：[回溯]({% post_url 2024-8-18-BackTracking %})、[贪心]({% post_url 2024-8-21-Greedy %})、[基础 DP、打家劫舍与股票]({% post_url 2024-8-27-DP %})、[背包]({% post_url 2024-8-28-01Package %})、[序列与字符串 DP]({% post_url 2024-8-29-DP %})。
- 图论：[并查集]({% post_url 2024-8-30-DisjointSet %})、[DFS 与 BFS]({% post_url 2024-8-30-Graph %})、[最小生成树]({% post_url 2024-8-30-MST %})、[Dijkstra]({% post_url 2024-8-30-MinDist %})、[拓扑排序]({% post_url 2024-8-30-TopoSort %})、[Bellman-Ford]({% post_url 2024-8-31-BellmanFord %})、[Floyd]({% post_url 2024-8-31-Floyd %})。

## 二分查找和双指针

### 基本二分查找


**形象理解**：像查字典，不从第一页开始翻，而是每次打开中间一页。目标更大就扔掉左半本，更小就扔掉右半本，每次将搜索范围缩小一半。

#### 执行步骤

```text
// 1. 用 left、right 表示当前仍可能包含答案的闭区间。
// 2. 取中点 mid，避免直接写 (left + right) 造成整数溢出。
// 3. nums[mid] 等于目标时立即返回。
// 4. 目标更大就令 left = mid + 1，否则令 right = mid - 1。
// 5. 区间为空仍未找到，返回 -1。
```

#### C++ 实现

```cpp
int search(vector<int>& nums, int target) {
        int left = 0;
        int right = nums.size() - 1; // 定义target在左闭右闭的区间里，[left, right]
        while (left <= right) { // 当left==right，区间[left, right]依然有效，所以用 <=
            int midd = left + ((right - left) / 2);// 防止溢出 等同于(left + right)/2
            if (nums[mid] > target) {
                right = mid - 1; // target 在左区间，所以[left, middle - 1]
            } else if (nums[mid] < target) {
                left = mid + 1; // target 在右区间，所以[middle + 1, right]
            } else { // nums[middle] == target
                return mid; // 数组中找到目标值，直接返回下标
            }
        }
        // 未找到目标值
        return -1;
    }
// Another version:
int search(vector<int>& nums, int target) {
        int left = 0;
        int right = nums.size(); // 定义target在左闭右开的区间里，即：[left, right)
        while (left < right) { // 因为left == right的时候，在[left, right)是无效的空间，所以使用 <
            int middle = left + ((right - left) >> 1);
            if (nums[middle] > target) {
                right = middle; // target 在左区间，在[left, middle)中
            } else if (nums[middle] < target) {
                left = middle + 1; // target 在右区间，在[middle + 1, right)中
            } else { // nums[middle] == target
                return middle; // 数组中找到目标值，直接返回下标
            }
        }
        // 未找到目标值
        return -1;
    }
```

#### Python 实现

```python
def search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```


### 二分查找左右边界


**形象理解**：普通二分只要“撞见一个目标”就结束；边界二分像在一排相同书名中寻找最左或最右一本，找到后仍继续向对应方向挤压。

#### 执行步骤

```text
// 1. 左边界：nums[mid] >= target 时收缩右侧，逼近第一个 target。
// 2. 循环结束后检查 left 是否越界以及 nums[left] 是否真等于 target。
// 3. 右边界：nums[mid] <= target 时收缩左侧，逼近最后一个 target。
// 4. 循环结束后检查 right 的合法性，不能只返回插入位置。
```

#### C++ 实现

```cpp
int left_bound(vector<int>& nums, int target) {
    if (nums.size() == 0) return -1;
    int left = 0;
    int right = nums.size() -1; // 注意

    while (left <= right) { // 注意
        int mid = (left + right) / 2;
        if (nums[mid] == target) {
            right = mid - 1;
        } else if (nums[mid] < target) {
            left = mid + 1;
        } else if (nums[mid] > target) {
            right = mid - 1; // 注意
        }
    }
    // target 比所有数都大
    if (left == nums.size()) return -1;
    return nums[left] == target ? left : -1;
}

int right_bound(vector<int>& nums, int target) {
    if (nums.size() == 0) return -1;
    int left = 0, right = nums.size()-1;

    while (left <= right) {
        int mid = (left + right) / 2;
        if (nums[mid] == target) {
            left = mid + 1; // 注意
        } else if (nums[mid] < target) {
            left = mid + 1;
        } else if (nums[mid] > target) {
            right = mid -1;
        }
    }
    if (right == -1) return -1;
    return nums[right] == target ? right : -1; // 注意
}
```

#### Python 实现

```python
from bisect import bisect_left, bisect_right

def search_range(nums, target):
    left = bisect_left(nums, target)
    if left == len(nums) or nums[left] != target:
        return [-1, -1]
    return [left, bisect_right(nums, target) - 1]
```


### 两个有序数组的中位数


**形象理解**：不是把两副排好序的牌全部合并，而是寻找第 k 小。比较两副牌各自第 `k/2` 张，较小一侧的前半段肯定不可能包含第 k 小，可以整段丢弃。

#### 执行步骤

```text
// 1. 将中位数转换为求第 k 小；偶数长度时求中间两个数再取平均。
// 2. 保证较短数组耗尽时，直接从另一个数组取剩余的第 k 个。
// 3. k == 1 时返回两个当前元素中的较小值。
// 4. 比较两数组各自向后 k/2 位置的值。
// 5. 丢弃较小值所在数组的一段，并把 k 减去实际丢弃数量。
// 6. 重复直到命中边界条件。
```

#### C++ 实现

```cpp
int getKthElement(const vector<int> &nums1, const vector<int> &nums2, int k){
    int m = nums1.size(), n = nums2.size();
    int index1 = 0, index2 = 0;

    while(true){
        if(index1 == m){
            return nums2[index2 + k - 1];
        }
        if(index2 == n){
            return nums1[index1 + k - 1];
        }
        if(k == 1)
            return min(nums1[index1], nums2[index2]);

        int newIndex1 = min(index1 + k/2 - 1, m-1);
        int newIndex2 = min(index2 + k/2 - 1, n-1);
        int pivot1 = nums1[newIndex1], pivot2 = nums2[newIndex2];
        if(pivot1 <= pivot2){
            k-= newIndex1 - index1 + 1;
            index1 = newIndex1 + 1;
        } else{
            k -= newIndex2 - index2 + 1;
            index2 = newIndex2 + 1;
        }
    }
}
double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
    int totalLength = nums1.size() + nums2.size();
    if(totalLength % 2 == 1)
        return getKthElement(nums1, nums2, (totalLength+1)/2);
    else
        return (getKthElement(nums1,nums2,totalLength/2) + getKthElement(nums1, nums2, totalLength/2+1)) / 2.0;
}
```

#### Python 实现

```python
def find_median_sorted_arrays(a, b):
    if len(a) > len(b):
        a, b = b, a
    m, n = len(a), len(b)
    left, right = 0, m
    while left <= right:
        i = (left + right) // 2
        j = (m + n + 1) // 2 - i
        a_left = float("-inf") if i == 0 else a[i - 1]
        a_right = float("inf") if i == m else a[i]
        b_left = float("-inf") if j == 0 else b[j - 1]
        b_right = float("inf") if j == n else b[j]
        if a_left <= b_right and b_left <= a_right:
            if (m + n) % 2:
                return max(a_left, b_left)
            return (max(a_left, b_left) + min(a_right, b_right)) / 2
        if a_left > b_right:
            right = i - 1
        else:
            left = i + 1
```


### 链表的中间结点


**形象理解**：两个人同时沿链表走，一个每次一步，一个每次两步。快的人走到终点时，慢的人刚好走完整条路的一半。

#### 执行步骤

```text
// 1. slow 和 fast 都从头结点出发。
// 2. 每轮 slow 走一步，fast 走两步。
// 3. fast 到达末尾或越过末尾时停止。
// 4. 返回 slow；偶数长度时自然落在第二个中间结点。
```

#### C++ 实现

```cpp
ListNode middleNode(ListNode head) {
    ListNode fast = head;
    ListNode slow = head;
    while (fast != null && fast.next != null) {
        fast = fast.next.next;
        slow = slow.next;
    }
    return slow;
}
```

#### Python 实现

```python
def middle_node(head):
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
    return slow
```


### 删除有序数组中的重复项


**形象理解**：慢指针是“已整理区域的最后一个位置”，快指针像质检员向后扫描。看到新数字，就把它搬到已整理区域后面。

#### 执行步骤

```text
// 1. 慢指针指向当前最后一个不重复元素。
// 2. 快指针从第二个元素开始扫描。
// 3. 当 nums[fast] != nums[slow] 时发现新元素。
// 4. 先移动 slow，再把新元素写到 nums[slow]。
// 5. 返回 slow + 1，表示唯一元素数量。
```

#### C++ 实现

```cpp
int removeElement(vector<int>& nums, int val) {
    int slowIndex=0;
    for(int fastIndex =0;fastIndex<nums.size();fastIndex++)
    {
        if(val!=nums[fastIndex])
        {
            nums[slowIndex++]=nums[fastIndex];
        }
    }
    return slowIndex;
}
```

#### Python 实现

```python
def remove_duplicates(nums):
    write = 0
    for value in nums:
        if write == 0 or value != nums[write - 1]:
            nums[write] = value
            write += 1
    return write
```


### 有序数组的平方


**形象理解**：平方后的最大值一定来自原数组两端，因为最负的数平方后也可能最大。左右两端像两名候选人，每次把平方更大的放到答案末尾。

#### 执行步骤

```text
// 1. left 指向最左端，right 指向最右端，write 指向结果末尾。
// 2. 比较 nums[left]^2 和 nums[right]^2。
// 3. 将较大值写入 result[write]，移动对应端点。
// 4. write 从后向前移动，直到所有位置填满。
```

#### C++ 实现

```cpp
vector<int> sortedSquares(vector<int>& A) {
        int k = A.size() - 1;
        vector<int> result(A.size(), 0);
        for (int i = 0, j = A.size() - 1; i <= j;) { // 注意这里要i <= j，因为最后要处理两个元素
            if (A[i] * A[i] < A[j] * A[j])  {
                result[k--] = A[j] * A[j];
                j--;
            }
            else {
                result[k--] = A[i] * A[i];
                i++;
            }
        }
        return result;
    }
```

#### Python 实现

```python
def sorted_squares(nums):
    answer = [0] * len(nums)
    left, right = 0, len(nums) - 1
    for write in range(len(nums) - 1, -1, -1):
        if abs(nums[left]) > abs(nums[right]):
            answer[write] = nums[left] ** 2
            left += 1
        else:
            answer[write] = nums[right] ** 2
            right -= 1
    return answer
```


### 反转字符串


**形象理解**：像交换一排人的座位，最左和最右互换，然后向中间靠拢，直到两人相遇。

#### 执行步骤

```text
// 1. left 从开头出发，right 从末尾出发。
// 2. 交换 s[left] 和 s[right]。
// 3. left++、right--，继续处理内部区间。
// 4. left >= right 时完成原地反转。
```

#### C++ 实现

```cpp
void reverseString(char[] s) {
    int left = 0;
    int right = s.length - 1;
    while (left < right) {
        char tmp = s[left];
        s[left] = s[right];
        s[right] = tmp;
        left++;
        right--;
    }
}
```

#### Python 实现

```python
def reverse_string(chars):
    left, right = 0, len(chars) - 1
    while left < right:
        chars[left], chars[right] = chars[right], chars[left]
        left, right = left + 1, right - 1
```


### 最长回文子串


**形象理解**：把每个字符或两个字符之间的缝隙都当成“折纸中心”，同时向左右展开；左右字符相同就继续展开，断开时记录最大半径。

#### 执行步骤

```text
// 1. 枚举每个位置作为奇数回文中心。
// 2. 再枚举相邻位置之间作为偶数回文中心。
// 3. 左右指针在边界内且字符相同就同步向外移动。
// 4. 每次扩展成功后比较并更新最长区间。
// 5. 最后根据最佳起点和长度截取答案。
```

#### C++ 实现

```cpp
string expandCenter(string s, int left, int right) {
    int len = s.length();
    // 要时刻注意避免越界访问
    while (left >= 0 && right < len
           && s[left] == s[right]) {
        left--;
        right++;
    }
    return s.substr(left + 1, right - left - 1);
}

string longestPalindrome(string s) {
    int len = s.length();
    string res;
    for (int i = 0; i < len; i++) {
        string sub1 = expandCenter(s, i, i);
        string sub2 = expandCenter(s, i, i + 1);
        res = res.length() >= sub1.length() ? res : sub1;
        res = res.length() >= sub2.length() ? res : sub2;
    }
    return res;
}
```

#### Python 实现

```python
def longest_palindrome(s):
    best_left = best_right = 0
    for center in range(len(s)):
        for left, right in ((center, center), (center, center + 1)):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                if right - left > best_right - best_left:
                    best_left, best_right = left, right
                left, right = left - 1, right + 1
    return s[best_left:best_right + 1]
```


## 滑动窗口和前缀和

### 长度最小的子数组


**形象理解**：窗口像一根可以伸缩的尺子。右端不断向前吸收数字，和达到 target 后，左端尽量收缩，找出仍满足条件的最短尺子。

#### 执行步骤

```text
// 1. right 向右移动，并把 nums[right] 加入 windowSum。
// 2. 当 windowSum >= target 时，当前窗口是可行答案。
// 3. 更新最短长度，再移除 nums[left] 并令 left++。
// 4. 继续收缩，直到窗口重新不满足条件。
// 5. 扫描结束后，无答案返回 0。
```

#### C++ 实现

```cpp
int minSubArrayLen(int s, vector<int>& nums) {
        int result = INT32_MAX;
        int sum = 0; // 滑动窗口数值之和
        int i = 0; // 滑动窗口起始位置
        int subLength = 0; // 滑动窗口的长度
        for (int j = 0; j < nums.size(); j++) {
            sum += nums[j];
            // 注意这里使用while，每次更新 i（起始位置），并不断比较子序列是否符合条件
            while (sum >= s) {
                subLength = (j - i + 1); // 取子序列的长度
                result = result < subLength ? result : subLength;
                sum -= nums[i++]; // 这里体现出滑动窗口的精髓之处，不断变更i（子序列的起始位置）
            }
        }
        // 如果result没有被赋值的话，就返回0，说明没有符合条件的子序列
        return result == INT32_MAX ? 0 : result;
    }
```

#### Python 实现

```python
def min_sub_array_len(target, nums):
    left = window_sum = 0
    answer = len(nums) + 1
    for right, value in enumerate(nums):
        window_sum += value
        while window_sum >= target:
            answer = min(answer, right - left + 1)
            window_sum -= nums[left]
            left += 1
    return 0 if answer > len(nums) else answer
```


### 无重复字符的最长子串


**形象理解**：窗口像一段只允许每种卡片出现一次的队伍。右边加入重复卡片时，从左边逐个赶人，直到重复被消除。

#### 执行步骤

```text
// 1. 右指针加入当前字符并更新出现次数或最后位置。
// 2. 若当前字符重复，移动 left 越过造成重复的旧位置。
// 3. 窗口恢复无重复后，用 right - left + 1 更新答案。
// 4. right 扫完整个字符串后返回最大长度。
```

#### C++ 实现

```cpp
int lengthOfLongestSubstring(string s) {
    int n = s.size();
    int ans = 0;
    unordered_map<char, int> map;//记录字符上一次出现的位置,字符为key，位置为value
    for(int i = 0, j = 0; j < n; j++){//i表示子串的起始位置，j表示子串的结束位置
        if(map.find(s[j]) != map.end()){//如果字符上一次出现的位置在i之后，更新i
            i = max(map[s[j]], i);//map[s[j]]表示字符s[j]上一次出现的位置
        }
        ans = max(ans, j - i + 1);//更新结果
        map[s[j]] = j + 1;		//更新字符s[j]上一次出现的位置
    }
    return ans;
}
```

#### Python 实现

```python
def length_of_longest_substring(s):
    last, left, answer = {}, 0, 0
    for right, char in enumerate(s):
        left = max(left, last.get(char, -1) + 1)
        last[char] = right
        answer = max(answer, right - left + 1)
    return answer
```


### 最大连续 1 的个数 III


**形象理解**：把最多 k 次翻转看成 k 张“0 的通行证”。窗口里 0 的数量超过 k，就从左边收回通行证，直到重新合法。

#### 执行步骤

```text
// 1. right 扫描数组，遇到 0 就增加 zeroCount。
// 2. zeroCount > k 时移动 left，并在移出 0 时减少计数。
// 3. 窗口合法后更新最大长度。
// 4. 每个元素最多进出窗口一次，整体 O(n)。
```

#### C++ 实现

```cpp
int longestOnes(vector<int>& nums, int k) {
    int n = nums.size();
    int left = 0, lsum = 0, rsum = 0;
    int ans = 0;
    for (int right = 0; right < n; ++right) {
        rsum += 1 - nums[right]; //[0，right]中0的个数.
        while (lsum < rsum - k) {
            lsum += 1 - nums[left];
            ++left;
        }
        ans = max(ans, right - left + 1);
    }
    return ans;
}
```

#### Python 实现

```python
def longest_ones(nums, k):
    left = zeros = 0
    for right, value in enumerate(nums):
        zeros += value == 0
        if zeros > k:
            zeros -= nums[left] == 0
            left += 1
    return len(nums) - left
```


### 最小覆盖子串


**形象理解**：要凑齐一张购物清单。右端不断把商品放进购物车，凑齐后左端开始退掉多余商品，直到再退一步就不完整，此时得到局部最小答案。

#### 执行步骤

```text
// 1. 统计 t 中每个字符需要的数量。
// 2. right 扩张窗口，更新窗口计数和已经满足的字符种类。
// 3. 当所有需求满足时，反复移动 left 删除多余字符。
// 4. 每次收缩前记录更短的合法窗口。
// 5. 若从未合法返回空串，否则截取最佳区间。
```

#### C++ 实现

```cpp
unordered_map<char,int> ori, cnt;

bool check() {
    for (const auto &p : ori) {
        if (cnt[p.first] < p.second) {
            return false;
        }
    }
    return true;
}

string minWindow(string s, string t) {
    for (auto & c : t){
        ori[c]++;
    }
    int length = INT_MAX;
    int left = 0;
    int right = 0;
    int ansLeft = 0;
    while (right <= s.length()) {
        if (ori.find(s[right]) != ori.end()) {
            cnt[s[right]]++;
        }
        while (check() && left <=right) {
            if (right - left + 1 < length){
                ansLeft = left;
                length = right - left + 1;
            }
            if (ori.find(s[left])!= ori.end()) {
                cnt[s[left]]--;
            }
            left++;
        }
        right++;
    }
    return length == INT_MAX ? "" : s.substr(ansLeft, length);
}
```

#### Python 实现

```python
from collections import Counter

def min_window(s, t):
    need = Counter(t)
    missing, left, best = len(t), 0, (0, float("inf"))
    for right, char in enumerate(s, 1):
        if need[char] > 0:
            missing -= 1
        need[char] -= 1
        while missing == 0:
            if right - left < best[1] - best[0]:
                best = (left, right)
            old = s[left]
            need[old] += 1
            if need[old] > 0:
                missing += 1
            left += 1
    return "" if best[1] == float("inf") else s[best[0]:best[1]]
```


### 水果成篮


**形象理解**：只有两个篮子，每个篮子只能装一种水果。走过果树时出现第三种水果，就从最左边开始丢弃，直到篮子里重新只剩两种。

#### 执行步骤

```text
// 1. right 加入水果类型并增加频数。
// 2. 类型数超过 2 时，left 逐个移出水果。
// 3. 某类型频数降为 0 时从哈希表删除。
// 4. 每次窗口合法后更新最大长度。
```

#### C++ 实现

```cpp
int totalFruit(vector<int>& fruits) {
        int n = fruits.size();
        unordered_map<int, int> cnt;

        int left = 0, ans = 0;
        for (int right = 0; right < n; ++right) {
            ++cnt[fruits[right]];
            while (cnt.size() > 2) {
                auto it = cnt.find(fruits[left]);
                --it->second;
                if (it->second == 0) {
                    cnt.erase(it);
                }
                ++left;
            }
            ans = max(ans, right - left + 1);
        }
        return ans;
    }
```

#### Python 实现

```python
from collections import defaultdict

def total_fruit(fruits):
    count, left, answer = defaultdict(int), 0, 0
    for right, fruit in enumerate(fruits):
        count[fruit] += 1
        while len(count) > 2:
            count[fruits[left]] -= 1
            if count[fruits[left]] == 0:
                del count[fruits[left]]
            left += 1
        answer = max(answer, right - left + 1)
    return answer
```


### 找出所有字母异位词


**形象理解**：用一张 26 格的“欠账表”比较固定长度窗口和目标串。窗口每向右滑一格，只需处理离开的字符和新进入的字符，不必重新清点全部字符。

#### 执行步骤

```text
// 1. 统计 p 的字符频数，并建立同长度初始窗口。
// 2. 用 differ 记录频数不相等的字符种类数。
// 3. differ == 0 时记录窗口起点。
// 4. 窗口右移时先移出左字符，再加入右字符。
// 5. 每次修改频数前后更新 differ。
```

#### C++ 实现

```cpp
vector<int> findAnagrams(string s, string p) {
    int sLen = s.size(), pLen = p.size();
    if (sLen < pLen)
        return vector<int>{};
    vector<int> ans;
    vector<int> count(26);
    for(int i = 0; i < pLen; ++i){
        ++count[s[i]-'a'];
        --count[p[i]-'a'];
    }
    int differ = 0;
    for(int i = 0; i < 26; ++i){
        if(count[i]!=0)
            ++differ;
    }
    if (differ == 0) {
        ans.push_back(0);
    }
    for(int i = 0; i < sLen-pLen; ++i) {
        if (count[s[i]-'a'] == 1){
            --differ;
        } else if(count[s[i]-'a'] == 0){
            ++differ;
        }
        --count[s[i]-'a'];

        if (count[s[i+pLen]-'a'] == -1){
            --differ;
        } else if(count[s[i+pLen]-'a'] == 0){
            ++differ;
        }
        ++count[s[i+pLen]-'a'];

        if(differ == 0)
            ans.push_back(i+1);
    }
    return ans;
}
```

#### Python 实现

```python
from collections import Counter

def find_anagrams(s, p):
    need, window, width = Counter(p), Counter(), len(p)
    answer = []
    for right, char in enumerate(s):
        window[char] += 1
        if right >= width:
            old = s[right - width]
            window[old] -= 1
            if window[old] == 0:
                del window[old]
        if window == need:
            answer.append(right - width + 1)
    return answer
```


### 串联所有单词的子串


**形象理解**：字符变成等长积木。按单词长度对字符串切片，共有 `wordLength` 种起始偏移；每种偏移上都运行一次“单词版异位词窗口”。

#### 执行步骤

```text
// 1. 统计 words 中每个单词需要的次数。
// 2. 枚举 0 到 wordLength - 1 的切分偏移。
// 3. right 每次跨过一个完整单词，并加入窗口计数。
// 4. 某单词超量时，left 也按单词长度收缩。
// 5. 窗口单词数等于 words.size() 时记录起点。
```

#### C++ 实现

```cpp
vector<int> findSubstring(string &s, vector<string> &words) {
    vector<int> res;
    int m = words.size(), n = words[0].size(), ls = s.size();
    for (int i = 0; i < n && i + m * n <= ls; ++i) {
        unordered_map<string, int> differ;
        for (int j = 0; j < m; ++j) {
            ++differ[s.substr(i + j * n, n)];
        }
        for (string &word: words) {
            if (--differ[word] == 0) {
                differ.erase(word);
            }
        }
        for (int start = i; start < ls - m * n + 1; start += n) {
            if (start != i) {
                string word = s.substr(start + (m - 1) * n, n);
                if (++differ[word] == 0) {
                    differ.erase(word);
                }
                word = s.substr(start - n, n);
                if (--differ[word] == 0) {
                    differ.erase(word);
                }
            }
            if (differ.empty()) {
                res.emplace_back(start);
            }
        }
    }
    return res;
}
```

#### Python 实现

```python
from collections import Counter, defaultdict

def find_substring(s, words):
    width, count, need = len(words[0]), len(words), Counter(words)
    answer = []
    for offset in range(width):
        left, used, window = offset, 0, defaultdict(int)
        for right in range(offset, len(s) - width + 1, width):
            word = s[right:right + width]
            window[word] += 1
            used += 1
            while window[word] > need[word]:
                old = s[left:left + width]
                window[old] -= 1
                used -= 1
                left += width
            if used == count:
                answer.append(left)
    return answer
```


### 字符串的排列


**形象理解**：与异位词题相同，只是不需要收集所有位置；固定长度窗口的字符账本一旦与 `s1` 完全一致，就可以立即返回 true。

#### 执行步骤

```text
// 1. 统计 s1 的频数，建立 s2 的固定长度窗口。
// 2. 每次滑动只更新进入和离开的两个字符。
// 3. 两份频数一致时立即返回 true。
// 4. 所有窗口都不匹配则返回 false。
```

#### C++ 实现

```cpp
int main() {
    int n, a, b;
    cin >> n;
    vector<int> vec(n);
    vector<int> p(n);
    int presum = 0;
    for (int i = 0; i < n; i++) {
        cin >> vec[i];
        presum += vec[i];
        p[i] = presum;
    }

    while (cin >> a >> b) {
        int sum;
        if (a == 0) sum = p[b];
        else sum = p[b] - p[a - 1];
        cout << sum << endl;
    }
}
```

#### Python 实现

```python
from collections import Counter

def check_inclusion(s1, s2):
    width, need = len(s1), Counter(s1)
    window = Counter(s2[:width])
    if window == need:
        return True
    for right in range(width, len(s2)):
        window[s2[right]] += 1
        old = s2[right - width]
        window[old] -= 1
        if window[old] == 0:
            del window[old]
        if window == need:
            return True
    return False
```


### 除自身以外数组的乘积


**形象理解**：每个位置的答案由“左边所有人的乘积”和“右边所有人的乘积”拼成。先从左向右发左侧成绩，再从右向左乘上右侧成绩。

#### 执行步骤

```text
// 1. 第一遍令 answer[i] 保存 nums[0..i-1] 的乘积。
// 2. 用 rightProduct 从数组末尾向前累乘。
// 3. answer[i] *= rightProduct，得到除自身外的完整乘积。
// 4. 再把 nums[i] 纳入 rightProduct，供更左位置使用。
```

#### C++ 实现

```cpp
vector<int> productExceptSelf(vector<int>& nums) {
    int length = nums.size();
    vector<int> answer(length);

    // answer[i] 表示索引 i 左侧所有元素的乘积
    // 因为索引为 '0' 的元素左侧没有元素， 所以 answer[0] = 1
    answer[0] = 1;
    for (int i = 1; i < length; i++) {
        answer[i] = nums[i - 1] * answer[i - 1];
    }

    // R 为右侧所有元素的乘积
    // 刚开始右边没有元素，所以 R = 1
    int R = 1;
    for (int i = length - 1; i >= 0; i--) {
        // 对于索引 i，左边的乘积为 answer[i]，右边的乘积为 R
        answer[i] = answer[i] * R;
        // R 需要包含右边所有的乘积，所以计算下一个结果时需要将当前值乘到 R 上
        R *= nums[i];
    }
    return answer;
}
```

#### Python 实现

```python
def product_except_self(nums):
    answer, prefix = [1] * len(nums), 1
    for i, value in enumerate(nums):
        answer[i] = prefix
        prefix *= value
    suffix = 1
    for i in range(len(nums) - 1, -1, -1):
        answer[i] *= suffix
        suffix *= nums[i]
    return answer
```


### 和为 K 的子数组


**形象理解**：当前前缀和是 `pre`，要让某段和为 k，只需在过去找前缀和 `pre-k`。哈希表像一本账簿，记录每种历史余额出现过几次。

#### 执行步骤

```text
// 1. 先记录前缀和 0 出现一次，表示子数组可从下标 0 开始。
// 2. 遍历元素并更新当前前缀和 pre。
// 3. 将历史中 pre - k 的出现次数累加到答案。
// 4. 最后增加 pre 的出现次数，供后续位置使用。
```

#### C++ 实现

```cpp
int subarraySum(vector<int>& nums, int k) {
        unordered_map<int, int> mp;
        mp[0] = 1; //注意！
        int count = 0, pre = 0;
        for (auto& x:nums) {
            pre += x;
            if (mp.find(pre - k) != mp.end()) {
                count += mp[pre - k];
            }
            mp[pre]++;
        }
        return count;
    }
```

#### Python 实现

```python
from collections import defaultdict

def subarray_sum(nums, k):
    frequency, prefix, answer = defaultdict(int, {0: 1}), 0, 0
    for value in nums:
        prefix += value
        answer += frequency[prefix - k]
        frequency[prefix] += 1
    return answer
```


### 和可被 K 整除的子数组


**形象理解**：两个前缀和除以 k 的余数相同，它们的差就能被 k 整除。只需要统计过去见过多少次相同余数。

#### 执行步骤

```text
// 1. 初始化余数 0 出现一次。
// 2. 更新前缀和，并计算 ((pre % k) + k) % k 处理负数。
// 3. 相同余数的历史次数就是以当前位置结尾的新答案数。
// 4. 将当前余数计入哈希表。
```

#### C++ 实现

```cpp
int subarraysDivByK(vector<int>& nums, int k)
{
    int ret = 0;
    unordered_map<int, int> hash;
    hash[0] = 1;

    int sum = 0;
    for (auto e : nums)
    {
        sum += e;
        int mod = (sum % k + k) % k; //这里考虑有负数的情况
        if (hash.count(mod)) ret += hash[mod];
        hash[mod]++;
    }

    return ret;
}
```

#### Python 实现

```python
from collections import defaultdict

def subarrays_div_by_k(nums, k):
    frequency, remainder, answer = defaultdict(int, {0: 1}), 0, 0
    for value in nums:
        remainder = (remainder + value) % k
        answer += frequency[remainder]
        frequency[remainder] += 1
    return answer
```


### 连续数组


**形象理解**：把 0 看成 -1，问题就变成找和为 0 的最长子数组。同一个前缀和再次出现，说明两次出现之间的增量为 0。

#### 执行步骤

```text
// 1. 将 1 计为 +1，将 0 计为 -1，维护前缀和。
// 2. 只保存每个前缀和第一次出现的位置。
// 3. 再次遇到相同前缀和时，用当前位置减最早位置更新长度。
// 4. 初始记录 prefix 0 位于 -1，支持答案从下标 0 开始。
```

#### C++ 实现

```cpp
int findMaxLength(vector<int>& nums)
{
    for (auto& e : nums) if (e == 0) e = -1;
    unordered_map<int, int> hash;
    hash[0] = -1;

    int ret = 0;
    int sum = 0;
    for (int i = 0; i < nums.size(); i++)
    {
        sum += nums[i];
        if (hash.count(sum)) ret = max(ret, i - hash[sum]);
        else hash[sum] = i;
    }

    return ret;
}
```

#### Python 实现

```python
def find_max_length(nums):
    first, balance, answer = {0: -1}, 0, 0
    for i, value in enumerate(nums):
        balance += 1 if value else -1
        if balance in first:
            answer = max(answer, i - first[balance])
        else:
            first[balance] = i
    return answer
```


## 模拟过程

### 螺旋矩阵 II


**形象理解**：像沿着一圈圈跑道填数字。每填完外圈，就把上、下、左、右四条边同时向内收缩一格。

#### 执行步骤

```text
// 1. 维护 top、bottom、left、right 四条尚未填充的边界。
// 2. 沿上边从左到右填充，再令 top++。
// 3. 沿右边从上到下填充，再令 right--。
// 4. 边界仍合法时，沿下边从右到左、沿左边从下到上填充。
// 5. 四条边界交错后结束，保证每个格子只填一次。
```

#### C++ 实现

```cpp
vector<vector<int>> generateMatrix(int n) {
    vector<vector<int>> res(n, vector<int>(n, 0)); // 使用vector定义一个二维数组
    int startx = 0, starty = 0; // 定义每循环一个圈的起始位置
    int loop = n / 2; // 每个圈循环几次，例如n为奇数3，那么loop = 1 只是循环一圈
    int mid = n / 2; // 矩阵中间的位置，例如：n为3， 中间的位置就是(1，1)
    int count = 1; // 用来给矩阵中每一个空格赋值
    int offset = 1; // 需要控制每一条边遍历的长度，每次循环右边界收缩一位
    int i,j;
    while (loop --) {
        i = startx;
        j = starty;

        // 下面开始的四个for就是模拟转了一圈
        // 模拟填充上行从左到右(左闭右开)
        for (j; j < n - offset; j++) {
            res[i][j] = count++;
        }
        // 模拟填充右列从上到下(左闭右开)
        for (i; i < n - offset; i++) {
            res[i][j] = count++;
        }
        // 模拟填充下行从右到左(左闭右开)
        for (; j > starty; j--) {
            res[i][j] = count++;
        }
        // 模拟填充左列从下到上(左闭右开)
        for (; i > startx; i--) {
            res[i][j] = count++;
        }

        // 第二圈开始的时候，起始位置要各自加1， 例如：第一圈起始位置是(0, 0)，第二圈起始位置是(1, 1)
        startx++;
        starty++;

        // offset 控制每一圈里每一条边遍历的长度
        offset += 1;
    }

    // 如果n为奇数的话，需要单独给矩阵最中间的位置赋值
    if (n % 2) {
        res[mid][mid] = count;
    }
    return res;
}
```

#### Python 实现

```python
def generate_matrix(n):
    matrix = [[0] * n for _ in range(n)]
    left, right, top, bottom, value = 0, n - 1, 0, n - 1, 1
    while left <= right:
        for col in range(left, right + 1):
            matrix[top][col], value = value, value + 1
        top += 1
        for row in range(top, bottom + 1):
            matrix[row][right], value = value, value + 1
        right -= 1
        if top <= bottom:
            for col in range(right, left - 1, -1):
                matrix[bottom][col], value = value, value + 1
            bottom -= 1
        if left <= right:
            for row in range(bottom, top - 1, -1):
                matrix[row][left], value = value, value + 1
            left += 1
    return matrix
```


## 链表

### 删除链表元素与虚拟头结点


**形象理解**：虚拟头结点像在真实队伍前放一个永远不删除的领队，使删除第一个真实节点和删除中间节点使用完全相同的操作。

#### 执行步骤

```text
// 1. 创建 dummy，并令 dummy->next = head。
// 2. cur 指向待检查节点的前一个节点。
// 3. cur->next 值等于 val 时，跨过并释放该节点。
// 4. 否则 cur 向后移动一步。
// 5. 返回 dummy->next，而不是旧的 head。
```

#### C++ 实现

```cpp
ListNode* removeElements(ListNode* head, int val) {
    ListNode* dummyHead = new ListNode(0); // 设置一个虚拟头结点
    dummyHead->next = head; // 将虚拟头结点指向head，这样方便后面做删除操作
    ListNode* cur = dummyHead;
    while (cur->next != NULL) {
        if(cur->next->val == val) {
            ListNode* tmp = cur->next;
            cur->next = cur->next->next;
            delete tmp;
        } else {
            cur = cur->next;
        }
    }
    head = dummyHead->next;
    delete dummyHead;
    return head;
}
```

#### Python 实现

```python
def remove_elements(head, value):
    dummy = ListNode(0, head)
    current = dummy
    while current.next:
        if current.next.val == value:
            current.next = current.next.next
        else:
            current = current.next
    return dummy.next
```


### 设计链表


**形象理解**：链表类像一列可插拔车厢；虚拟头结点固定不动，`size` 记录真实车厢数，使每次插入和删除都先找到前驱车厢再改两根连接。

#### 执行步骤

```text
// 1. get：检查 index，再从 dummy->next 走 index 步。
// 2. addAtHead：等价于在 index 0 插入。
// 3. addAtTail：等价于在 index size 插入。
// 4. addAtIndex：找到第 index 个位置的前驱，接入新节点并增加 size。
// 5. deleteAtIndex：找到前驱，跨过目标节点，释放内存并减少 size。
```

#### C++ 实现

```cpp
class MyLinkedList {
public:
    // 定义链表节点结构体
    struct LinkedNode {
        int val;
        LinkedNode* next;
        LinkedNode(int val):val(val), next(nullptr){}
    };

    // 初始化链表
    MyLinkedList() {
        _dummyHead = new LinkedNode(0); // 这里定义的头结点 是一个虚拟头结点，而不是真正的链表头结点
        _size = 0;
    }

    // 获取到第index个节点数值，如果index是非法数值直接返回-1， 注意index是从0开始的，第0个节点就是头结点
    int get(int index) {
        if (index > (_size - 1) || index < 0) {
            return -1;
        }
        LinkedNode* cur = _dummyHead->next;
        while(index--){ // 如果--index 就会陷入死循环
            cur = cur->next;
        }
        return cur->val;
    }

    // 在链表最前面插入一个节点，插入完成后，新插入的节点为链表的新的头结点
    void addAtHead(int val) {
        LinkedNode* newNode = new LinkedNode(val);
        newNode->next = _dummyHead->next;
        _dummyHead->next = newNode;
        _size++;
    }

    // 在链表最后面添加一个节点
    void addAtTail(int val) {
        LinkedNode* newNode = new LinkedNode(val);
        LinkedNode* cur = _dummyHead;
        while(cur->next != nullptr){
            cur = cur->next;
        }
        cur->next = newNode;
        _size++;
    }

    // 在第index个节点之前插入一个新节点，例如index为0，那么新插入的节点为链表的新头节点。
    // 如果index 等于链表的长度，则说明是新插入的节点为链表的尾结点
    // 如果index大于链表的长度，则返回空
    // 如果index小于0，则在头部插入节点
    void addAtIndex(int index, int val) {

        if(index > _size) return;
        if(index < 0) index = 0;
        LinkedNode* newNode = new LinkedNode(val);
        LinkedNode* cur = _dummyHead;
        while(index--) {
            cur = cur->next;
        }
        newNode->next = cur->next;
        cur->next = newNode;
        _size++;
    }

    // 删除第index个节点，如果index 大于等于链表的长度，直接return，注意index是从0开始的
    void deleteAtIndex(int index) {
        if (index >= _size || index < 0) {
            return;
        }
        LinkedNode* cur = _dummyHead;
        while(index--) {
            cur = cur ->next;
        }
        LinkedNode* tmp = cur->next;
        cur->next = cur->next->next;
        delete tmp;
        //delete命令指示释放了tmp指针原本所指的那部分内存，
        //被delete后的指针tmp的值（地址）并非就是NULL，而是随机值。也就是被delete后，
        //如果不再加上一句tmp=nullptr,tmp会成为乱指的野指针
        //如果之后的程序不小心使用了tmp，会指向难以预想的内存空间
        tmp=nullptr;
        _size--;
    }

    // 打印链表
    void printLinkedList() {
        LinkedNode* cur = _dummyHead;
        while (cur->next != nullptr) {
            cout << cur->next->val << " ";
            cur = cur->next;
        }
        cout << endl;
    }
private:
    int _size;
    LinkedNode* _dummyHead;

};
```

#### Python 实现

```python
class MyLinkedList:
    def __init__(self):
        self.size = 0
        self.head = ListNode()

    def get(self, index):
        if index < 0 or index >= self.size:
            return -1
        node = self.head.next
        for _ in range(index):
            node = node.next
        return node.val

    def addAtHead(self, val):
        self.addAtIndex(0, val)

    def addAtTail(self, val):
        self.addAtIndex(self.size, val)

    def addAtIndex(self, index, val):
        if index > self.size:
            return
        index = max(index, 0)
        previous = self.head
        for _ in range(index):
            previous = previous.next
        previous.next = ListNode(val, previous.next)
        self.size += 1

    def deleteAtIndex(self, index):
        if index < 0 or index >= self.size:
            return
        previous = self.head
        for _ in range(index):
            previous = previous.next
        previous.next = previous.next.next
        self.size -= 1
```


### 反转链表


**形象理解**：像逐节把火车连接方向掉头。`cur` 拿着当前车厢，`next` 先保存后续车厢，再让当前车厢指回 `prev`。

#### 执行步骤

```text
// 1. prev 初始化为空，cur 指向原头结点。
// 2. 保存 next = cur->next，防止反转后丢失后半段。
// 3. 令 cur->next = prev，完成当前指针反向。
// 4. prev、cur 同时向前推进。
// 5. cur 为空时 prev 就是新头结点。
```

#### C++ 实现

```cpp
ListNode* reverseList(ListNode* head) {
    ListNode* temp; // 保存cur的下一个节点
    ListNode* cur = head;
    ListNode* pre = NULL;
    while(cur) {
        temp = cur->next;  // 保存一下 cur的下一个节点，因为接下来要改变cur->next
        cur->next = pre; // 翻转操作
        // 更新pre 和 cur指针
        pre = cur;
        cur = temp;
    }
    return pre;
}
```

#### Python 实现

```python
def reverse_list(head):
    previous = None
    while head:
        head.next, previous, head = previous, head, head.next
    return previous
```


### K 个一组反转链表


**形象理解**：把链表按每 k 节切成一节节车厢组。先确认凑得齐 k 节，整组掉头后再与前后组重新接轨；最后不足 k 节的部分保持原样。

#### 执行步骤

```text
// 1. dummy 指向 head，pre 指向当前待反转组的前驱。
// 2. 从 pre 向后走 k 步找 tail；不足 k 个就结束。
// 3. 保存 nextGroup = tail->next，避免断链。
// 4. 反转 [head, tail]，得到该组新的头尾。
// 5. pre 接新组头，旧组头接 nextGroup，再进入下一组。
```

#### C++ 实现

```cpp
pair<ListNode*,ListNode*> reverse(ListNode*head, ListNode *tail){
    ListNode *pre = tail->next;
    ListNode *p = head;
    while (pre != tail){
        ListNode *tmp = p->next;
        p->next = pre;
        pre = p;
        p = tmp;
    }
    return {tail, head};
}

ListNode* reverseKGroup(ListNode* head, int k) {
    ListNode *dummyhead = new ListNode(0, head);
    ListNode *pre = dummyhead;
    while(head) {
        ListNode *tail = pre;
        for (int i = 0; i < k; ++i){
            tail = tail-> next;
            if (!tail){
                return dummyhead->next;
            }
        }
        ListNode *tmp =tail->next;
        tie(head, tail) = reverse(head, tail);
        pre->next = head;
        // tail->next = tmp;
        pre = tail;
        head = tmp;
    }
    return dummyhead->next;
}
```

#### Python 实现

```python
def reverse_k_group(head, k):
    dummy = ListNode(0, head)
    group_prev = dummy
    while True:
        kth = group_prev
        for _ in range(k):
            kth = kth.next
            if not kth:
                return dummy.next
        group_next, previous, current = kth.next, kth.next, group_prev.next
        while current != group_next:
            current.next, previous, current = previous, current, current.next
        tail = group_prev.next
        group_prev.next = kth
        group_prev = tail
```


### 两两交换链表节点


**形象理解**：每次处理相邻两节 `first`、`second`，先把前驱接到 second，再让 second 接 first，最后让 first 接回后续链表。

#### 执行步骤

```text
// 1. dummy 统一处理头部交换，prev 指向当前二元组前驱。
// 2. 确认 prev 后至少还有两个节点。
// 3. 保存 first、second 和第二个节点之后的位置。
// 4. 按 prev -> second -> first -> next 的顺序重连。
// 5. prev 移到 first，继续交换下一对。
```

#### C++ 实现

```cpp
ListNode* swapPairs(ListNode* head) {
    ListNode* dummyHead = new ListNode(0); // 设置一个虚拟头结点
    dummyHead->next = head; // 将虚拟头结点指向head，这样方便后面做删除操作
    ListNode* cur = dummyHead;
    while(cur->next != nullptr && cur->next->next != nullptr) {
        ListNode* tmp = cur->next; // 记录临时节点
        ListNode* tmp1 = cur->next->next->next; // 记录临时节点

        cur->next = cur->next->next;    // 步骤一
        cur->next->next = tmp;          // 步骤二
        cur->next->next->next = tmp1;   // 步骤三

        cur = cur->next->next; // cur移动两位，准备下一轮交换
    }
    ListNode* result = dummyHead->next;
    delete dummyHead;
    return result;
}

ListNode* swapPairs(ListNode* head) {
    if (head == nullptr || head->next == nullptr)
        return head;
    ListNode *tmp = head->next->next;
    ListNode *p = head->next;
    p->next = head;
    head->next = swapPairs(tmp);
    return p;
}
```

#### Python 实现

```python
def swap_pairs(head):
    dummy = ListNode(0, head)
    previous = dummy
    while previous.next and previous.next.next:
        first, second = previous.next, previous.next.next
        first.next = second.next
        second.next = first
        previous.next = second
        previous = first
    return dummy.next
```


### 删除链表倒数第 N 个节点


**形象理解**：让 fast 比 slow 领先 n 步，之后两人同速前进。fast 到终点时，slow 恰好站在待删除节点的前一位。

#### 执行步骤

```text
// 1. 使用 dummy 避免删除头结点时单独判断。
// 2. fast 先从 dummy 向前走 n + 1 步，制造间隔。
// 3. fast、slow 同时移动直到 fast 为空。
// 4. slow->next 就是倒数第 n 个节点。
// 5. 跨过并释放该节点，返回 dummy->next。
```

#### C++ 实现

```cpp
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode* dummy = new ListNode(0, head);
        ListNode* first = head;
        ListNode* second = dummy;
        for (int i = 0; i < n; ++i) {
            first = first->next;
        }
        while (first) {
            first = first->next;
            second = second->next;
        }
        second->next = second->next->next;
        ListNode* ans = dummy->next;
        delete dummy;
        return ans;
    }
```

#### Python 实现

```python
def remove_nth_from_end(head, n):
    dummy = ListNode(0, head)
    fast = slow = dummy
    for _ in range(n + 1):
        fast = fast.next
    while fast:
        fast, slow = fast.next, slow.next
    slow.next = slow.next.next
    return dummy.next
```


### 链表相交


**形象理解**：A 走完自己的路后去走 B，B 走完自己的路后去走 A。两人总路程都变成 `A+B`，若有共同尾巴就会在入口相遇。

#### 执行步骤

```text
// 1. pA、pB 分别从两个头结点出发。
// 2. pA 到空后切换到 headB，pB 到空后切换到 headA。
// 3. 两指针每轮各走一步，自动抵消长度差。
// 4. pA == pB 时返回该节点；没有交点时二者会同时为空。
```

#### C++ 实现

```cpp
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        unordered_set<ListNode *> visited;
        ListNode *temp = headA;
        while (temp != nullptr) {
            visited.insert(temp);
            temp = temp->next;
        }
        temp = headB;
        while (temp != nullptr) {
            if (visited.count(temp)) {
                return temp;
            }
            temp = temp->next;
        }
        return nullptr;
    }
```

#### Python 实现

```python
def get_intersection_node(head_a, head_b):
    a, b = head_a, head_b
    while a is not b:
        a = a.next if a else head_b
        b = b.next if b else head_a
    return a
```


### 环形链表入口


**形象理解**：快慢指针像操场上的两名跑者，有环就一定会相遇。相遇后让一人回起点，两人都改成一步一格，再次相遇的位置就是环入口。

#### 执行步骤

```text
// 1. slow 每次一步，fast 每次两步，先判断是否存在环。
// 2. fast 或 fast->next 为空说明无环。
// 3. 首次相遇后，将一个指针重置到 head。
// 4. 两指针都每次走一步。
// 5. 第二次相遇的位置就是入环节点。
```

#### C++ 实现

```cpp
ListNode *detectCycle(ListNode *head) {
    ListNode *slow = head, *fast = head;
    while (fast != nullptr) {
        slow = slow->next;
        if (fast->next == nullptr) {
            return nullptr;
        }
        fast = fast->next->next;
        if (fast == slow) {
            ListNode *ptr = head;
            while (ptr != slow) {
                ptr = ptr->next;
                slow = slow->next;
            }
            return ptr;
        }
    }
    return nullptr;
}
```

#### Python 实现

```python
def detect_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:
            finder = head
            while finder is not slow:
                finder, slow = finder.next, slow.next
            return finder
    return None
```


## 哈希表与多数之和

### 有效的字母异位词


**形象理解**：给 26 个字母各开一个账户，读取 s 时存款，读取 t 时取款；最后所有账户都归零，才说明两串只是排列不同。

#### 执行步骤

```text
// 1. 长度不同可直接返回 false。
// 2. 遍历 s，增加对应字符频数。
// 3. 遍历 t，减少对应字符频数。
// 4. 检查所有频数是否为 0。
```

#### C++ 实现

```cpp
bool isAnagram(string s, string t) {
    if(s.length() != t.length())
        return false;
    unordered_map<char,int> sMap;
    for(auto c : s){
        sMap[c]++;
    }
    for(auto c :t){
        if(sMap[c] <= 0){
            return false;
        }else{
            sMap[c]--;
        }
    }
    return true;
}
```

#### Python 实现

```python
from collections import Counter

def is_anagram(s, t):
    return Counter(s) == Counter(t)
```


### 两个数组的交集


**形象理解**：先把第一个数组的元素登记成会员名单，再扫描第二个数组；查到会员就加入答案集合，集合天然负责去重。

#### 执行步骤

```text
// 1. 将 nums1 放入 unordered_set。
// 2. 遍历 nums2，检查元素是否在集合中。
// 3. 命中时加入结果集合，避免重复答案。
// 4. 将结果集合转换为 vector 返回。
```

#### C++ 实现

```cpp
vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
    sort(nums1.begin(), nums1.end());
    sort(nums2.begin(), nums2.end());
    int length1 = nums1.size(), length2 = nums2.size();
    int index1 = 0, index2 = 0;
    vector<int> intersection;
    while (index1 < length1 && index2 < length2) {
        int num1 = nums1[index1], num2 = nums2[index2];
        if (num1 == num2) {
            // 保证加入元素的唯一性
            if (!intersection.size() || num1 != intersection.back()) {
                intersection.push_back(num1);
            }
            index1++;
            index2++;
        } else if (num1 < num2) {
            index1++;
        } else {
            index2++;
        }
    }
    return intersection;
}
```

#### Python 实现

```python
def intersection(nums1, nums2):
    return list(set(nums1) & set(nums2))
```


### 快乐数


**形象理解**：每个数字都会唯一地变成下一个数字，整个过程像沿单向道路前进；若到不了 1，就一定会进入以前走过的环。

#### 执行步骤

```text
// 1. 编写函数计算各位数字的平方和。
// 2. 用集合记录已经出现过的 n。
// 3. n == 1 时返回 true。
// 4. n 已经在集合中时说明进入循环，返回 false。
// 5. 否则记录 n，并更新为下一次平方和。
```

#### C++ 实现

```cpp
bool isHappy(int n) {
    unordered_set<int> numSet;
    while(n != 1){
        if (numSet.count(n)){
            return false;
        }
        numSet.insert(n);
        int sum = 0;
        while(n!=0){
            int b = n%10;
            sum += b*b;
            n = n/10;
        }
        n= sum;
    }
    return true;
}
```

#### Python 实现

```python
def is_happy(n):
    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = sum(int(digit) ** 2 for digit in str(n))
    return n == 1
```


### 两数之和


**形象理解**：看到当前数字 x 时，不向后盲找，而是在过去的账本中查询它的搭档 `target-x` 是否已经出现。

#### 执行步骤

```text
// 1. 哈希表保存 value -> index。
// 2. 遍历 nums[i]，计算 complement = target - nums[i]。
// 3. complement 已存在时返回两个下标。
// 4. 否则记录 nums[i] 和 i，供后续元素匹配。
```

#### C++ 实现

```cpp
vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int, int> hashtable;
    for (int i = 0; i < nums.size(); ++i) {
        auto it = hashtable.find(target - nums[i]);
        if (it != hashtable.end()) {
            return {it->second, i};
        }
        hashtable[nums[i]] = i;
    }
    return {-1,-1};
}
```

#### Python 实现

```python
def two_sum(nums, target):
    seen = {}
    for i, value in enumerate(nums):
        if target - value in seen:
            return [seen[target - value], i]
        seen[value] = i
```


### 四数相加 II


**形象理解**：把四重循环拆成两支两人队伍。先统计 A+B 的所有成绩，再让 C+D 查询需要多少个相反成绩才能凑成 0。

#### 执行步骤

```text
// 1. 双循环枚举 nums1 和 nums2 的和并统计次数。
// 2. 双循环枚举 nums3 和 nums4 的和 sum。
// 3. 查找哈希表中 -sum 出现的次数。
// 4. 将该次数累加到答案。
```

#### C++ 实现

```cpp
int fourSumCount(vector<int>& A, vector<int>& B, vector<int>& C, vector<int>& D) {
    unordered_map<int, int> countAB;
    for (int u: A) {
        for (int v: B) {
            ++countAB[u + v];
        }
    }
    int ans = 0;
    for (int u: C) {
        for (int v: D) {
            if (countAB.count(-u - v)) {
                ans += countAB[-u - v];
            }
        }
    }
    return ans;
}
```

#### Python 实现

```python
from collections import Counter

def four_sum_count(a, b, c, d):
    pair_sums = Counter(x + y for x in a for y in b)
    return sum(pair_sums[-x - y] for x in c for y in d)
```


### 三数之和


**形象理解**：排序后固定第一个数，剩下两个数像夹子从左右两端向中间夹。和太小就移动左端，太大就移动右端。

#### 执行步骤

```text
// 1. 排序，使双指针移动方向可判断，也方便去重。
// 2. 枚举第一个数 i；与前一个相同就跳过。
// 3. left=i+1、right=n-1，计算三数和。
// 4. 和小于 0 移动 left，大于 0 移动 right。
// 5. 等于 0 时记录答案，并跳过左右重复值后继续。
```

#### C++ 实现

```cpp
vector<vector<int>> threeSum(vector<int>& nums) {
    sort(nums.begin(),nums.end());
    vector<vector<int>> ans;
    for(int i = 0; i <nums.size()-2;++i){
        if(i!=0 && nums[i] == nums[i-1])
            continue;
        int left = i+1;
        int right = nums.size()-1;
        while(left < right){
            if(nums[i] + nums[left] + nums[right] < 0){
                left++;
            } else if(nums[i]+nums[left]+nums[right] > 0){
                right--;
            }else {
                ans.push_back({nums[i],nums[left],nums[right]});
                left++;
                while(nums[left] == nums[left-1] && left < right)
                    left++;
                right--;
                while(nums[right] == nums[right+1] && right > left)
                    right--;
            }
        }
    }
    return ans;
}
```

#### Python 实现

```python
def three_sum(nums):
    nums.sort()
    answer = []
    for i, value in enumerate(nums):
        if i and value == nums[i - 1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = value + nums[left] + nums[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                answer.append([value, nums[left], nums[right]])
                left, right = left + 1, right - 1
                while left < right and nums[left] == nums[left - 1]:
                    left += 1
                while left < right and nums[right] == nums[right + 1]:
                    right -= 1
    return answer
```


### 四数之和


**形象理解**：在三数之和外再固定一个数。前两层像锁定两张牌，后两张仍用左右夹逼寻找目标。

#### 执行步骤

```text
// 1. 排序并枚举第一个下标 i，跳过重复值。
// 2. 枚举第二个下标 j，同样跳过重复值。
// 3. left=j+1、right=n-1 计算四数和，使用更宽整数避免溢出。
// 4. 根据和与 target 的关系移动左右指针。
// 5. 命中后记录并跳过两端重复值。
```

#### C++ 实现

```cpp
vector<vector<int>> fourSum(vector<int>& nums, int target) {
    if(nums.size()<4)
        return vector<vector<int>>{};
    sort(nums.begin(), nums.end());
    vector<vector<int>> ans;
    for(int i = 0; i < nums.size()-3; ++i){
        if(nums[i] > 0 && nums[i]> target)
            break;
        if (i>0 && nums[i] == nums[i-1])
            continue;
        for(int j = i+1; j < nums.size()-2; ++j){
            if(nums[j] > 0 && nums[j]+nums[i] > target){
                break;
            }
            if(j>i+1 && nums[j] == nums[j-1])
                continue;
            int left = j+1;
            int right = nums.size()-1;
            while(left < right){
                int a =nums[i];
                int b = nums[j];
                int c = nums[left];
                int d = nums[right];
                if(c > 0 && a+b+c >= target)
                    break;
                if((long)a+b+c+d < target)
                    left++;
                else if((long)a+b+c+d > target)
                    right--;
                else{
                    ans.push_back({a,b,c,d});
                    right--;
                    left++;
                    while(left < right && nums[right] == nums[right+1])
                    right--;
                    while(left < right && nums[left] == nums[left-1])
                    left++;
                }
            }
        }
    }
    return ans;
}
```

#### Python 实现

```python
def four_sum(nums, target):
    nums.sort()
    answer = []
    for i in range(len(nums) - 3):
        if i and nums[i] == nums[i - 1]:
            continue
        for j in range(i + 1, len(nums) - 2):
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue
            left, right = j + 1, len(nums) - 1
            while left < right:
                total = nums[i] + nums[j] + nums[left] + nums[right]
                if total == target:
                    answer.append([nums[i], nums[j], nums[left], nums[right]])
                    left, right = left + 1, right - 1
                    while left < right and nums[left] == nums[left - 1]: left += 1
                    while left < right and nums[right] == nums[right + 1]: right -= 1
                elif total < target:
                    left += 1
                else:
                    right -= 1
    return answer
```


## 字符串与 KMP

### 反转字符串 II


**形象理解**：把字符串按每 `2k` 个字符划成一组，每组只把前 k 个人掉头；最后一组人数不足时按照同样边界规则处理。

#### 执行步骤

```text
// 1. i 每次增加 2*k，定位下一组开头。
// 2. 反转区间 [i, min(i+k, n))。
// 3. 不足 k 个时 min 自动选择字符串末尾。
// 4. 后 k 个字符保持原顺序。
```

#### C++ 实现

```cpp
string reverseStr(string s, int k) {
    int n = s.length();
    for (int i = 0; i < n; i += 2 * k) {
        reverse(s.begin() + i, s.begin() + min(i + k, n));
    }
    return s;
}
```

#### Python 实现

```python
def reverse_str(s, k):
    chars = list(s)
    for start in range(0, len(chars), 2 * k):
        chars[start:start + k] = reversed(chars[start:start + k])
    return "".join(chars)
```


### 替换数字


**形象理解**：每个数字要膨胀成 `number`。先计算最终长度，再从后向前搬运，就不会覆盖还没读取的原字符。

#### 执行步骤

```text
// 1. 统计数字数量，计算扩展后的新长度。
// 2. resize 一次性准备最终空间。
// 3. oldIndex 从原末尾、newIndex 从新末尾向前移动。
// 4. 遇到字母复制一个字符；遇到数字倒序写入 "number"。
// 5. 两个指针完成后得到原地扩展结果。
```

#### C++ 实现

```cpp
#include <iostream>
using namespace std;
int main() {
    string s;
    while (cin >> s) {
        int sOldIndex = s.size() - 1;
        int count = 0; // 统计数字的个数
        for (int i = 0; i < s.size(); i++) {
            if (s[i] >= '0' && s[i] <= '9') {
                count++;
            }
        }
        // 扩充字符串s的大小，也就是将每个数字替换成"number"之后的大小
        s.resize(s.size() + count * 5);
        int sNewIndex = s.size() - 1;
        // 从后往前将数字替换为"number"
        while (sOldIndex >= 0) {
            if (s[sOldIndex] >= '0' && s[sOldIndex] <= '9') {
                s[sNewIndex--] = 'r';
                s[sNewIndex--] = 'e';
                s[sNewIndex--] = 'b';
                s[sNewIndex--] = 'm';
                s[sNewIndex--] = 'u';
                s[sNewIndex--] = 'n';
            } else {
                s[sNewIndex--] = s[sOldIndex];
            }
            sOldIndex--;
        }
        cout << s << endl;
    }
}
```

#### Python 实现

```python
def replace_digits(s):
    return "".join("number" if char.isdigit() else char for char in s)
```


### 反转字符串中的单词


**形象理解**：先把整句话像磁带一样整体倒放，单词顺序已经反过来；再把每个单词内部单独倒正，最后压缩多余空格。

#### 执行步骤

```text
// 1. 去除首尾和单词间多余空格，保证单词间只留一个空格。
// 2. 反转整个字符串，使单词顺序反转。
// 3. 扫描空格边界，逐个反转单词内部字符。
// 4. 返回整理后的字符串。
```

#### C++ 实现

```cpp
string reverseWords(string s) {
    // 使用双指针
    int m = s.size() - 1;
    string res;
    // 除去尾部空格
    while (s[m] == ' ' && m > 0) m--;
    int n = m; // n是另一个指针
    while (m >= 0) {
        while (m >= 0 && s[m] != ' ') m--;
        res += s.substr(m + 1, n - m) + " "; // 获取单词并加上空格
        while (m >= 0 && s[m] == ' ') m--;
        n = m;
    }
    return res.substr(0, res.size() - 1); // 忽略最后一位的空格
}
```

#### Python 实现

```python
def reverse_words(s):
    return " ".join(reversed(s.split()))
```


### 右旋字符串


**形象理解**：要把尾部 k 个字符搬到前面，可以先整体翻面，再分别把新前半段和后半段翻正，像三次翻转一副分成两摞的牌。

#### 执行步骤

```text
// 1. 令 k %= n，处理 k 大于长度的情况。
// 2. 反转整个字符串。
// 3. 反转前 k 个字符。
// 4. 反转剩余字符，得到右旋结果。
```

#### C++ 实现

```cpp
#include<iostream>
#include<algorithm>
using namespace std;
int main() {
    int n;
    string s;
    cin >> n;
    cin >> s;
    int len = s.size(); //获取长度

    reverse(s.begin(), s.end()); // 整体反转
    reverse(s.begin(), s.begin() + n); // 先反转前一段，长度n
    reverse(s.begin() + n, s.end()); // 再反转后一段
    cout << s << endl;
}
```

#### Python 实现

```python
def rotate_right(s, k):
    if not s:
        return s
    k %= len(s)
    return s[-k:] + s[:-k]
```


### 找出字符串中第一个匹配下标


**形象理解**：暴力算法每次不匹配都回到起点；KMP 像记住模式串自己有哪些相同前后缀，失败时直接滑到仍可能匹配的位置。

#### 执行步骤

```text
// 1. 为 needle 构造前缀表 next。
// 2. i 扫描 haystack，j 表示 needle 已匹配到的位置。
// 3. 字符不匹配时，根据 next[j] 回退 j，而不回退 i。
// 4. 字符相同就推进 j。
// 5. j 到达模式串末尾时返回起始下标。
```

#### C++ 实现

```cpp
int strStr(string haystack, string needle) {
    int hLen = haystack.length();
    int nLen = needle.length();
    if(hLen < nLen)
        return -1;
    int start = 0;
    for(int start = 0; start < hLen - nLen + 1; ++start){
        if(haystack[start] == needle[0]){
            for(int i = 0; i < nLen; ++i){
                if(haystack[i+start] != needle[i])
                    break;
                if(i==nLen-1 && haystack[i+start] == needle[i])
                    return start;
            }
        }
    }
    return -1;
}
```

#### Python 实现

```python
def str_str(haystack, needle):
    if not needle:
        return 0
    prefix = [0] * len(needle)
    j = 0
    for i in range(1, len(needle)):
        while j and needle[i] != needle[j]:
            j = prefix[j - 1]
        if needle[i] == needle[j]:
            j += 1
        prefix[i] = j
    j = 0
    for i, char in enumerate(haystack):
        while j and char != needle[j]:
            j = prefix[j - 1]
        if char == needle[j]:
            j += 1
        if j == len(needle):
            return i - j + 1
    return -1
```


### 重复的子字符串


**形象理解**：如果字符串由某个短模板重复组成，最长相等前后缀会留下一个完整周期；总长度必须能被周期长度整除。

#### 执行步骤

```text
// 1. 构造字符串的前缀表。
// 2. 令 longest 为最长相等真前后缀长度。
// 3. period = n - longest。
// 4. longest > 0 且 n % period == 0 时存在重复模板。
```

#### C++ 实现

```cpp
bool repeatedSubstringPattern(string s) {
        return (s + s).find(s, 1) != s.size();
    }
```

#### Python 实现

```python
def repeated_substring_pattern(s):
    return s in (s + s)[1:-1]
```


## 栈、队列和单调队列

### 用栈实现队列


**形象理解**：一个栈负责收件，另一个栈负责发件。发件栈为空时，把收件栈全部倒过去，两次后进先出就恢复成先进先出。

#### 执行步骤

```text
// 1. push 永远压入 input 栈。
// 2. pop/peek 前若 output 为空，将 input 全部搬到 output。
// 3. output 栈顶就是最早进入队列的元素。
// 4. 两个栈都为空时队列为空。
```

#### C++ 实现

```cpp
class MyQueue {
public:
    stack<int> stIn;
    stack<int> stOut;
    /** Initialize your data structure here. */
    MyQueue() {

    }
    /** Push element x to the back of queue. */
    void push(int x) {
        stIn.push(x);
    }

    /** Removes the element from in front of queue and returns that element. */
    int pop() {
        // 只有当stOut为空的时候，再从stIn里导入数据（导入stIn全部数据）
        if (stOut.empty()) {
            // 从stIn导入数据直到stIn为空
            while(!stIn.empty()) {
                stOut.push(stIn.top());
                stIn.pop();
            }
        }
        int result = stOut.top();
        stOut.pop();
        return result;
    }

    /** Get the front element. */
    int peek() {
        int res = this->pop(); // 直接使用已有的pop函数
        stOut.push(res); // 因为pop函数弹出了元素res，所以再添加回去
        return res;
    }

    /** Returns whether the queue is empty. */
    bool empty() {
        return stIn.empty() && stOut.empty();
    }
};
```

#### Python 实现

```python
class MyQueue:
    def __init__(self):
        self.input, self.output = [], []

    def push(self, x):
        self.input.append(x)

    def _move(self):
        if not self.output:
            while self.input:
                self.output.append(self.input.pop())

    def pop(self):
        self._move()
        return self.output.pop()

    def peek(self):
        self._move()
        return self.output[-1]

    def empty(self):
        return not self.input and not self.output
```


### 用队列实现栈


**形象理解**：新元素入队后，让它前面的所有旧元素依次出队再入队，新元素就被旋转到队首，成为栈顶。

#### 执行步骤

```text
// 1. 记录入栈前队列大小 n。
// 2. 将新元素 push 到队尾。
// 3. 重复 n 次：弹出队首并重新放到队尾。
// 4. 此时新元素位于队首，pop/top 都可直接使用。
```

#### C++ 实现

```cpp
class MyStack {
public:
    queue<int> q;

    /** Initialize your data structure here. */
    MyStack() {

    }

    /** Push element x onto stack. */
    void push(int x) {
        int n = q.size();
        q.push(x);
        for (int i = 0; i < n; i++) {
            q.push(q.front());
            q.pop();
        }
    }

    /** Removes the element on top of the stack and returns that element. */
    int pop() {
        int r = q.front();
        q.pop();
        return r;
    }

    /** Get the top element. */
    int top() {
        int r = q.front();
        return r;
    }

    /** Returns whether the stack is empty. */
    bool empty() {
        return q.empty();
    }
};
```

#### Python 实现

```python
from collections import deque

class MyStack:
    def __init__(self):
        self.queue = deque()

    def push(self, x):
        self.queue.append(x)

    def pop(self):
        self.queue.rotate(1)
        return self.queue.popleft()

    def top(self):
        return self.queue[-1]

    def empty(self):
        return not self.queue
```


### 有效的括号


**形象理解**：左括号像打开的盒子，必须按后开先关的顺序闭合。栈顶永远保存当前最需要被关闭的盒子。

#### 执行步骤

```text
// 1. 遇到左括号时压入对应的期望右括号。
// 2. 遇到右括号时，栈空说明没有可匹配的左括号。
// 3. 右括号不等于栈顶期望值时返回 false。
// 4. 匹配成功就弹栈；扫描结束时栈必须为空。
```

#### C++ 实现

```cpp
bool isValid(string s) {
    if (s.size() % 2 != 0) return false; // 如果s的长度为奇数，一定不符合要求
    stack<char> st;
    for (int i = 0; i < s.size(); i++) {
        if (s[i] == '(') st.push(')');
        else if (s[i] == '{') st.push('}');
        else if (s[i] == '[') st.push(']');
        // 第三种情况：遍历字符串匹配的过程中，栈已经为空了，没有匹配的字符了，说明右括号没有找到对应的左括号 return false
        // 第二种情况：遍历字符串匹配的过程中，发现栈里没有我们要匹配的字符。所以return false
        else if (st.empty() || st.top() != s[i]) return false;
        else st.pop(); // st.top() 与 s[i]相等，栈弹出元素
    }
    // 第一种情况：此时我们已经遍历完了字符串，但是栈不为空，说明有相应的左括号没有右括号来匹配，所以return false，否则就return true
    return st.empty();
}
```

#### Python 实现

```python
def is_valid(s):
    pairs, stack = {")": "(", "]": "[", "}": "{"}, []
    for char in s:
        if char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return False
        else:
            stack.append(char)
    return not stack
```


### 删除相邻重复项


**形象理解**：栈像一块消除游戏棋盘。新字符与栈顶相同就一起消失，否则留下成为新的栈顶。

#### 执行步骤

```text
// 1. 从左到右读取字符。
// 2. 栈非空且当前字符等于栈顶时弹出栈顶。
// 3. 否则把当前字符压栈。
// 4. 最后栈中字符按原顺序组成答案。
```

#### C++ 实现

```cpp
string removeDuplicates(string S) {
    stack<char> st;
    for (char s : S) {
        if (st.empty() || s != st.top()) {
            st.push(s);
        } else {
            st.pop(); // s 与 st.top()相等的情况
        }
    }
    string result = "";
    while (!st.empty()) { // 将栈中元素放到result字符串汇总
        result += st.top();
        st.pop();
    }
    reverse (result.begin(), result.end()); // 此时字符串需要反转一下
    return result;

}
```

#### Python 实现

```python
def remove_duplicates_string(s):
    stack = []
    for char in s:
        if stack and stack[-1] == char:
            stack.pop()
        else:
            stack.append(char)
    return "".join(stack)
```


### 中缀表达式转后缀表达式


**形象理解**：数字直接进入输出，运算符在栈中按优先级排队；括号像临时隔离墙，右括号到来时把墙内运算符全部放行。

#### 执行步骤

```text
// 1. 操作数直接写入输出序列。
// 2. 左括号压入运算符栈。
// 3. 右括号弹出运算符直到左括号，并丢弃括号。
// 4. 普通运算符先弹出栈中优先级不低于自己的运算符，再入栈。
// 5. 扫描结束后把剩余运算符全部输出。
```

#### C++ 实现

```cpp
#include <iostream>
#include <stack>
#include <string>
#include <cctype>

using namespace std;

// 判断操作符优先级
int precedence(char op) {
    if(op == '+' || op == '-')
        return 1;
    if(op == '*' || op == '/')
        return 2;
    if(op == '^')
        return 3;
    return 0;
}

// 判断是否为操作符
bool isOperator(char c) {
    return (c == '+' || c == '-' || c == '*' || c == '/' || c == '^');
}

// 中缀表达式转后缀表达式
string infixToPostfix(string infix) {
    stack<char> s;
    string postfix = "";

    for(char& c : infix) {
        if(isdigit(c) || isalpha(c)) {
            postfix += c;  // 如果是操作数，直接添加到后缀表达式中
        } else if(c == '(') {
            s.push(c);  // 左括号直接压栈
        } else if(c == ')') {
            // 右括号，弹出直到遇到左括号
            while(!s.empty() && s.top() != '(') {
                postfix += s.top();
                s.pop();
            }
            s.pop(); // 弹出左括号
        } else if(isOperator(c)) {
            // 操作符，考虑优先级
            while(!s.empty() && precedence(s.top()) >= precedence(c)) {
                postfix += s.top();
                s.pop();
            }
            s.push(c);
        }
    }

    // 将剩余的操作符全部弹出
    while(!s.empty()) {
        postfix += s.top();
        s.pop();
    }

    return postfix;
}
```

#### Python 实现

```python
def infix_to_postfix(tokens):
    precedence = {"+": 1, "-": 1, "*": 2, "/": 2}
    output, operators = [], []
    for token in tokens:
        if token.isalnum():
            output.append(token)
        elif token == "(":
            operators.append(token)
        elif token == ")":
            while operators[-1] != "(": output.append(operators.pop())
            operators.pop()
        else:
            while operators and operators[-1] != "(" and precedence[operators[-1]] >= precedence[token]:
                output.append(operators.pop())
            operators.append(token)
    return output + operators[::-1]
```


### 逆波兰表达式求值


**形象理解**：数字先放到工作台；遇到运算符就取出最近的两个数字计算，再把结果放回工作台。

#### 执行步骤

```text
// 1. 遇到数字就转换为整数并压栈。
// 2. 遇到运算符时先弹出右操作数，再弹出左操作数。
// 3. 计算 left op right，并把结果压回栈。
// 4. 扫描结束后栈顶就是最终答案。
```

#### C++ 实现

```cpp
int evalRPN(vector<string>& tokens) {
    stack<long long> st;
    for (int i = 0; i < tokens.size(); i++) {
        if (tokens[i] == "+" || tokens[i] == "-" || tokens[i] == "*" || tokens[i] == "/") {
            long long num1 = st.top();
            st.pop();
            long long num2 = st.top();
            st.pop();
            if (tokens[i] == "+") st.push(num2 + num1);
            if (tokens[i] == "-") st.push(num2 - num1);
            if (tokens[i] == "*") st.push(num2 * num1);
            if (tokens[i] == "/") st.push(num2 / num1);
        } else {
            st.push(stoll(tokens[i]));
        }
    }

    int result = st.top();
    st.pop(); // 把栈里最后一个元素弹出（其实不弹出也没事）
    return result;
}
```

#### Python 实现

```python
def eval_rpn(tokens):
    stack = []
    for token in tokens:
        if token not in {"+", "-", "*", "/"}:
            stack.append(int(token))
            continue
        right, left = stack.pop(), stack.pop()
        if token == "+": stack.append(left + right)
        elif token == "-": stack.append(left - right)
        elif token == "*": stack.append(left * right)
        else: stack.append(int(left / right))
    return stack[-1]
```


### 前 K 个高频元素


**形象理解**：先给每个数字计票，再用一个只保留 k 名候选人的小顶堆。堆顶是当前入围者中票数最低的人，新候选人更强时就替换他。

#### 执行步骤

```text
// 1. 哈希表统计每个数字的频率。
// 2. 将 (频率, 数字) 放入最小堆。
// 3. 堆大小超过 k 时弹出最低频元素。
// 4. 最后堆中剩下的就是前 k 高频元素。
```

#### C++ 实现

```cpp
vector<int> topKFrequent(vector<int>& nums, int k) {
    unordered_map<int,int> mp;
    priority_queue<pair<int,int>> pq;
    for(auto num : nums)
        mp[num]++;
    for(auto it = mp.begin(); it !=mp.end(); ++it){
        pq.emplace(it->second, it->first);
    }
    vector<int> res;
    for(int i = 0; i < k; ++i){
        res.push_back(pq.top().second);
        pq.pop();
    }
    return res;
}
```

#### Python 实现

```python
from collections import Counter

def top_k_frequent(nums, k):
    return [value for value, _ in Counter(nums).most_common(k)]
```


### 滑动窗口最大值


**形象理解**：单调队列只保留仍可能成为冠军的人。新选手比队尾强时，队尾以后既更早退场又更弱，可以永久淘汰。

#### 执行步骤

```text
// 1. 队列保存下标，且对应值从队首到队尾递减。
// 2. 新元素进入前，弹出所有不大于它的队尾下标。
// 3. 弹出已经离开窗口的队首下标。
// 4. 当前队首就是窗口最大值的下标。
// 5. 每个下标最多入队出队一次，整体 O(n)。
```

#### C++ 实现

```cpp
vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    int n = nums.size();
    deque<int> q;
    for (int i = 0; i < k; ++i) {
        while (!q.empty() && nums[i] >= nums[q.back()]) {
            q.pop_back();
        }
        q.push_back(i);
    }

    vector<int> ans = {nums[q.front()]};
    for (int i = k; i < n; ++i) {
        while (!q.empty() && nums[i] >= nums[q.back()]) {
            q.pop_back();
        }
        q.push_back(i);
        while (q.front() <= i - k) {
            q.pop_front();
        }
        ans.push_back(nums[q.front()]);
    }
    return ans;
}
```

#### Python 实现

```python
from collections import deque

def max_sliding_window(nums, k):
    queue, answer = deque(), []
    for i, value in enumerate(nums):
        while queue and queue[0] <= i - k: queue.popleft()
        while queue and nums[queue[-1]] <= value: queue.pop()
        queue.append(i)
        if i >= k - 1: answer.append(nums[queue[0]])
    return answer
```


### 和至少为 K 的最短子数组


**形象理解**：前缀和队列保存最值得作为左端点的候选人。更晚且前缀和更小的候选人同时拥有“起点更靠后、减数更小”两项优势，可以淘汰旧候选人。

#### 执行步骤

```text
// 1. 构造 long long 前缀和，避免累加溢出。
// 2. 当前前缀和减队首 >= k 时得到可行区间，更新长度并弹队首。
// 3. 当前前缀和 <= 队尾前缀和时，队尾已被当前下标支配，弹出。
// 4. 将当前下标加入队尾，保持前缀和单调递增。
```

#### C++ 实现

```cpp
int shortestSubarray(vector<int>& nums, int k) {
    int n = nums.size();
    vector<long> preSumArr(n + 1);
    for (int i = 0; i < n; i++) {
        preSumArr[i + 1] = preSumArr[i] + nums[i];
    }
    int res = n + 1;
    deque<int> qu;
    for (int i = 0; i <= n; i++) {
        long curSum = preSumArr[i];
        while (!qu.empty() && curSum - preSumArr[qu.front()] >= k) {
            res = min(res, i - qu.front());
            qu.pop_front();
        }
        while (!qu.empty() && preSumArr[qu.back()] >= curSum) {
            qu.pop_back();
        }
        qu.push_back(i);
    }
    return res < n + 1 ? res : -1;
}
```

#### Python 实现

```python
from collections import deque

def shortest_subarray(nums, k):
    prefix = [0]
    for value in nums: prefix.append(prefix[-1] + value)
    queue, answer = deque(), len(nums) + 1
    for i, total in enumerate(prefix):
        while queue and total - prefix[queue[0]] >= k:
            answer = min(answer, i - queue.popleft())
        while queue and prefix[queue[-1]] >= total: queue.pop()
        queue.append(i)
    return -1 if answer > len(nums) else answer
```


## 排序与缓存设计

### 选择排序


**形象理解**：像每轮从未站好的队伍中选出最矮的人，把他换到当前第一个空位；左侧已排序区每轮增长一人。

#### 执行步骤

```text
// 1. i 指向当前待确定的位置。
// 2. 在 [i,n) 中扫描并记录最小元素下标 minIndex。
// 3. 交换 nums[i] 与 nums[minIndex]。
// 4. i 右移后，左侧元素已处于最终位置。
```

#### C++ 实现

```cpp
void selectionSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        // 将当前位置设置为最小值的索引
        int minIndex = i;
        for (int j = i + 1; j < n; j++) {
            // 在未排序的元素中找到最小值的索引
            if (arr[j] < arr[minIndex]) {
                minIndex = j;
            }
        }
        // 如果找到一个索引不等于当前的最小值索引，交换它们
        if (minIndex != i) {
            swap(arr[i], arr[minIndex]);
        }
    }
}
```

#### Python 实现

```python
def selection_sort(nums):
    for i in range(len(nums)):
        smallest = min(range(i, len(nums)), key=nums.__getitem__)
        nums[i], nums[smallest] = nums[smallest], nums[i]
    return nums
```


### 冒泡排序


**形象理解**：相邻两人逆序就交换，较大的数像气泡一样一步步浮到右端；每轮结束都会固定一个当前最大值。

#### 执行步骤

```text
// 1. 从左到右比较相邻元素 nums[j] 与 nums[j+1]。
// 2. 前者更大时交换，并标记本轮发生过变化。
// 3. 每轮右端已有一个最大值，下轮无需再检查它。
// 4. 某轮没有交换说明数组已经有序，可提前结束。
```

#### C++ 实现

```cpp
void bubbleSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        // flag用于标记这次循环是否发生了交换
        bool flag = false;
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                // 如果当前元素比后一个元素大，则交换它们
                swap(arr[j], arr[j + 1]);
                flag = true; // 发生了交换，将flag设置为true
            }
        }
        // 如果这次遍历没有发生交换，说明数组已经有序，直接跳出循环
        if (!flag) {
            break;
        }
    }
}
```

#### Python 实现

```python
def bubble_sort(nums):
    for end in range(len(nums) - 1, 0, -1):
        changed = False
        for i in range(end):
            if nums[i] > nums[i + 1]:
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
                changed = True
        if not changed:
            break
    return nums
```


### 插入排序


**形象理解**：像整理手里的扑克牌。拿起一张新牌，先把所有比它大的牌向右挪一格，再把它插进腾出的空位。

#### 执行步骤

```text
// 1. 保存 key = nums[i]，左侧 [0,i) 已有序。
// 2. j 从 i-1 向左寻找插入位置。
// 3. nums[j] > key 时将 nums[j] 右移到 j+1。
// 4. 循环结束后把 key 写入 j+1。
```

#### C++ 实现

```cpp
void insertionSort(vector<int>& arr) {
    int n = arr.size();
    for (int i = 1; i < n; i++) {
        int key = arr[i]; // 当前要插入的元素
        int j = i - 1; // 已经排序好的序列的最后一个元素的索引
        // 将比key大的元素往后移动
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j]; // 移动元素
            j--; // 向前移动索引
        }
        // 将key插入到正确的位置
        arr[j + 1] = key;
    }
}
```

#### Python 实现

```python
def insertion_sort(nums):
    for i in range(1, len(nums)):
        value, j = nums[i], i - 1
        while j >= 0 and nums[j] > value:
            nums[j + 1] = nums[j]
            j -= 1
        nums[j + 1] = value
    return nums
```


### 希尔排序


**形象理解**：先让相距很远的元素做插入排序，快速消除大的逆序；随后逐步缩短间隔，最后用 gap=1 的插入排序收尾。

#### 执行步骤

```text
// 1. 选择初始 gap，随后不断缩小直到 1。
// 2. 对每个 gap，把相同余数位置看成一组。
// 3. 在每组内执行“间隔为 gap”的插入排序。
// 4. gap 变小时数组已接近有序，最后一轮移动量很少。
```

#### C++ 实现

```cpp
void shellSort(vector<int>& arr) {
    // 初始化增量为数组长度的一半
    for (int gap = arr.size() / 2; gap > 0; gap /= 2) {
        // 对每个子数组进行直接插入排序
        for (int i = gap; i < arr.size(); i++) {
            int temp = arr[i];
            int j;
            // 比较相距gap的元素，并交换位置
            for (j = i; j >= gap && arr[j - gap] > temp; j -= gap) {
                arr[j] = arr[j - gap];
            }
            arr[j] = temp;
        }
    }
}
```

#### Python 实现

```python
def shell_sort(nums):
    gap = len(nums) // 2
    while gap:
        for i in range(gap, len(nums)):
            value, j = nums[i], i
            while j >= gap and nums[j - gap] > value:
                nums[j] = nums[j - gap]
                j -= gap
            nums[j] = value
        gap //= 2
    return nums
```


### 归并排序


**形象理解**：不断把队伍对半拆开，单人队伍天然有序；再让左右两个有序队伍比较队首，小者先进入新队伍。

#### 执行步骤

```text
// 1. 区间长度不超过 1 时递归返回。
// 2. 按 mid 把区间递归排序成左右两半。
// 3. 用两个指针分别指向左右有序段开头。
// 4. 每次把较小元素写入临时数组。
// 5. 复制任一侧剩余元素，再写回原区间。
```

#### C++ 实现

```cpp
void merge(vector<int>& arr, int left, int mid, int right) {
    int n1 = mid - left + 1; // 左侧子数组的大小
    int n2 = right - mid; // 右侧子数组的大小
    // 创建临时数组
	vector<int> L(n1), R(n2);
    // 复制数据到临时数组中
    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];
    // 合并临时数组
    // 初始化索引
    int i = 0;
    int j = 0;
    int k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            // 稳定排序
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }
    // 复制剩下的元素
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }
    // 复制剩下的元素
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}
// 归并排序的主函数
void mergeSort(vector<int>& arr, int left, int right) {
    // 如果 left == right，表示数组只有一个元素，则不用递归排序
    if (left < right) {
        int mid = left + (right - left) / 2;
        // 对左侧子数组进行归并排序
        mergeSort(arr, left, mid);
        // 对右侧子数组进行归并排序
        mergeSort(arr, mid + 1, right);
        // 合并两个已排序的子数组
        merge(arr, left, mid, right);
    }
}
```

#### Python 实现

```python
def merge_sort(nums):
    if len(nums) <= 1:
        return nums
    middle = len(nums) // 2
    left, right = merge_sort(nums[:middle]), merge_sort(nums[middle:])
    answer = []
    while left and right:
        answer.append((left if left[0] <= right[0] else right).pop(0))
    return answer + left + right
```


### 快速排序


**形象理解**：选一名基准，让比它小的人站左边、比它大的人站右边；基准就到达最终位置，再分别整理两边。

#### 执行步骤

```text
// 1. 选择 pivot；随机或三数取中可降低退化风险。
// 2. partition 移动指针，把小于基准和大于基准的元素分开。
// 3. 将基准放到分界位置，或得到左右分区边界。
// 4. 递归排序左区间和右区间。
// 5. 区间为空或只有一个元素时停止。
```

#### C++ 实现

```cpp
int partition(vector<int>& arr, int low, int high) {
    int pivot = arr[high]; // 选择最右侧的元素作为基准
    int i = (low - 1); // 小于基准的元素的索引
    for (int j = low; j <= high - 1; j++) {
        // 如果当前元素小于或等于基准
        if (arr[j] <= pivot) {
            i++; // 增加小于基准元素的索引
            swap(arr[i], arr[j]); // 交换元素
        }
    }
    swap(arr[i + 1], arr[high]); // 交换基准元素到正确的位置
    return (i + 1);
}
// 快速排序的主函数
void quickSort(vector<int>& arr, int low, int high) {
    if (low < high) {
        // pi是分区索引，arr[pi]现在位于正确的位置
        int pi = partition(arr, low, high);
        // 独立地对基准左侧和右侧的元素进行快速排序
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}
```

#### Python 实现

```python
def quick_sort(nums, left=0, right=None):
    right = len(nums) - 1 if right is None else right
    if left >= right:
        return nums
    pivot, i, j = nums[(left + right) // 2], left, right
    while i <= j:
        while nums[i] < pivot: i += 1
        while nums[j] > pivot: j -= 1
        if i <= j:
            nums[i], nums[j] = nums[j], nums[i]
            i, j = i + 1, j - 1
    quick_sort(nums, left, j)
    quick_sort(nums, i, right)
    return nums
```


### 堆排序与 `priority_queue`


**形象理解**：最大堆像一座冠军擂台，堆顶永远是最大值。把冠军换到数组末尾后缩小赛场，再让剩余元素重新决出冠军。

#### 执行步骤

```text
// 1. 从最后一个非叶子节点向前下沉，原地建立最大堆。
// 2. 交换堆顶与当前未排序区末尾，把最大值固定下来。
// 3. 缩小 heapSize，并对新堆顶执行 siftDown。
// 4. priority_queue<int> 默认最大堆，top() 读取冠军。
// 5. priority_queue<int, vector<int>, greater<int>> 是最小堆。
// 6. 求最大的 k 个元素可维护大小为 k 的最小堆，超出时 pop 最小者。
```

#### C++ 实现

```cpp
void heapify(vector<int>& arr, int n, int i) {
    int largest = i; // 初始化最大值为根节点
    int left = 2 * i + 1; // 左子节点
    int right = 2 * i + 2; // 右子节点
    // 如果左子节点大于根节点
    if (left < n && arr[left] > arr[largest])
        largest = left;
    // 如果右子节点大于目前的最大值
    if (right < n && arr[right] > arr[largest])
        largest = right;
    // 如果最大值不是根节点
    if (largest != i) {
        swap(arr[i], arr[largest]); // 交换根节点和最大值节点
        // 递归地对受影响的子树进行堆化
        heapify(arr, n, largest);
    }
}
// 堆排序的主函数
void heapSort(vector<int>& arr) {
    int n = arr.size();
    // 构建堆（重新排列数组）
    // n/2 --- n-1 都是叶子节点
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);
    // 逐个提取元素
    for (int i = n - 1; i > 0; i--) {
        // 将当前根节点移到末尾
        swap(arr[0], arr[i]);
        // 调用heapify在减少的堆上
        heapify(arr, i, 0);
    }
}
```

#### Python 实现

```python
import heapq

def heap_sort(nums):
    heapq.heapify(nums)
    return [heapq.heappop(nums) for _ in range(len(nums))]
```


### 计数排序


**形象理解**：不比较元素，而是给每个数值准备一个计数格；清点完后按数值从小到大重复输出对应次数。

#### 执行步骤

```text
// 1. 找到最小值与最大值，确定计数数组范围。
// 2. 扫描输入，count[value-minValue]++。
// 3. 按计数下标从小到大遍历。
// 4. 某值出现 count 次，就向原数组写回 count 次。
// 5. 适合数值范围不大的整数数据。
```

#### C++ 实现

```cpp
void countingSort(vector<int>& arr) {
    // 如果数组为空，则直接返回
    if (arr.empty())
        return;
    // 找到数组中的最大值以确定计数数组的大小
    int max_val = *std::max_element(arr.begin(), arr.end());
    // 初始化计数数组，并将所有元素设置为0
    vector<int> count(max_val + 1, 0);
    // 遍历数组，计算每个元素的出现次数
    for (int value : arr) {
        count[value]++;
    }
    // 修改计数数组，使其每个元素都包含小于或等于其索引值的元素数量
    for (size_t i = 1; i < count.size(); i++) {
        count[i] += count[i - 1];
    }
    // 创建输出数组
    vector<int> output(arr.size());
    // 构建输出数组
    for (int i = arr.size() - 1; i >= 0; i--) {
        output[--count[arr[i]]] = arr[i];
    }
    // 将输出数组复制回原数组
    arr = output;
}
```

#### Python 实现

```python
def counting_sort(nums):
    if not nums: return []
    low, high = min(nums), max(nums)
    count = [0] * (high - low + 1)
    for value in nums: count[value - low] += 1
    return [value for i, total in enumerate(count) for value in [i + low] * total]
```


### 桶排序


**形象理解**：先把数据按区间扔进不同桶中，桶之间天然有先后顺序；只需把每个小桶内部排好，再从左到右倒出来。

#### 执行步骤

```text
// 1. 根据数值范围和桶宽创建 buckets。
// 2. 计算每个元素的 bucketIndex 并放入对应桶。
// 3. 分别对每个桶内部排序。
// 4. 按桶下标递增顺序连接所有元素。
// 5. 数据分布较均匀时每桶很小，效率较高。
```

#### C++ 实现

```cpp
void bucketSort(vector<int>& arr) {
    // 如果数组为空，则直接返回
    if (arr.empty())
        return;
    // 找到数组中的最大值和最小值
    int max_val = *max_element(arr.begin(), arr.end());
    int min_val = *min_element(arr.begin(), arr.end());
    // 计算桶的数量
    int bucket_num = (max_val - min_val) / arr.size() + 1;
    // 创建桶
    vector<std::vector<int>> buckets(bucket_num);
    // 将元素分布到各个桶中
    for (int value : arr) {
        int bucket_index = (value - min_val) / arr.size();
        buckets[bucket_index].push_back(value);
    }
    // 对每个桶进行排序，并将结果收集到原数组中
    int index = 0;
    for (auto& bucket : buckets) {
        sort(bucket.begin(), bucket.end()); // 可以使用其他排序算法或递归调用桶排序
        for (int value : bucket) {
            arr[index++] = value;
        }
    }
}
```

#### Python 实现

```python
def bucket_sort(nums, bucket_size=10):
    if not nums: return []
    low = min(nums)
    buckets = [[] for _ in range((max(nums) - low) // bucket_size + 1)]
    for value in nums: buckets[(value - low) // bucket_size].append(value)
    return [value for bucket in buckets for value in sorted(bucket)]
```


### 基数排序


**形象理解**：先按个位稳定排队，再按十位、百位排队；因为每轮排序稳定，较低位的顺序会被保留，最终得到完整数值顺序。

#### 执行步骤

```text
// 1. exp 从 1 开始，依次代表个位、十位、百位。
// 2. 按 (value/exp)%10 对当前位执行稳定计数排序。
// 3. 将本轮结果写回原数组。
// 4. exp *= 10，直到超过最大数位。
// 5. 负数需要单独映射或拆分处理。
```

#### C++ 实现

```cpp

int getMaxDigit(const std::vector<int>& arr) {
    int max_val = *std::max_element(arr.begin(), arr.end());
    int digit = 0;
    while (max_val > 0) {
        max_val /= 10;
        digit++;
    }
    return digit;
}
// 基数排序函数
void radixSort(std::vector<int>& arr) {
    int max_digit = getMaxDigit(arr);
    int mod = 10;
    int dev = 1;
    vector<vector<int>> counter(10); // 因为我们有十个数字（0-9）
    for (int i = 0; i < max_digit; i++, dev *= 10, mod *= 10) {
        // 清空计数器
        for (auto& bucket : counter) {
            bucket.clear();
        }
        // 根据当前位数将元素分配到计数器中
        for (int value : arr) {
            int bucket_index = (value % mod) / dev;
            counter[bucket_index].push_back(value);
        }
        // 将计数器中的元素收集回原数组
        int index = 0;
        for (auto& bucket : counter) {
            for (int value : bucket) {
                arr[index++] = value;
            }
        }
    }
}
```

#### Python 实现

```python
def radix_sort_nonnegative(nums):
    exponent = 1
    while nums and max(nums) // exponent:
        buckets = [[] for _ in range(10)]
        for value in nums: buckets[value // exponent % 10].append(value)
        nums[:] = [value for bucket in buckets for value in bucket]
        exponent *= 10
    return nums
```


### Introsort


**形象理解**：它先用快速排序获得优秀的平均性能；递归过深说明分区不理想，就切换到堆排序兜底；小区间最后交给插入排序收尾。

#### 执行步骤

```text
// 1. 以快速排序开始，并设置最大递归深度约 2*log2(n)。
// 2. 分区规模较大且深度未耗尽时继续 quicksort partition。
// 3. 深度达到上限时改用 heapsort，保证 O(n log n) 最坏复杂度。
// 4. 小区间暂不深递归，最终统一用 insertion sort 整理。
// 5. 这也是常见 std::sort 实现采用的混合思想。
```

#### C++ 实现

```cpp
void introsort(vector<int>& nums) {
    if (nums.empty()) return;
    int depthLimit = 2 * static_cast<int>(log2(nums.size()));
    introsortLoop(nums.begin(), nums.end(), depthLimit); // 快排；达到深度限制时切换堆排
    insertionSort(nums.begin(), nums.end());             // 小区间最后统一插入排序
}
```

#### Python 实现

```python
import heapq
import math

def introsort(nums):
    def insertion(left, right):
        for i in range(left + 1, right + 1):
            value, j = nums[i], i - 1
            while j >= left and nums[j] > value:
                nums[j + 1] = nums[j]
                j -= 1
            nums[j + 1] = value

    def heap_sort(left, right):
        heap = nums[left:right + 1]
        heapq.heapify(heap)
        for i in range(left, right + 1):
            nums[i] = heapq.heappop(heap)

    def sort(left, right, depth):
        if right - left + 1 <= 16:
            insertion(left, right)
            return
        if depth == 0:
            heap_sort(left, right)
            return
        pivot, i, j = nums[(left + right) // 2], left, right
        while i <= j:
            while nums[i] < pivot: i += 1
            while nums[j] > pivot: j -= 1
            if i <= j:
                nums[i], nums[j] = nums[j], nums[i]
                i, j = i + 1, j - 1
        if left < j: sort(left, j, depth - 1)
        if i < right: sort(i, right, depth - 1)

    if nums:
        sort(0, len(nums) - 1, 2 * math.floor(math.log2(len(nums))))
    return nums
```


### 滑动窗口中位数


**形象理解**：`lower` 保存较小一半，`upper` 保存较大一半，像天平两侧；每次插入或删除后重新平衡，中间位置自然落在两个集合边界。

#### 执行步骤

```text
// 1. deque 按时间保存记录，便于从队首淘汰过期价格。
// 2. 新价格不大于 lower 最大值就放 lower，否则放 upper。
// 3. 过期时用 multiset::find 删除那个价格本身。
// 4. 调整两侧大小，使 lower 与 upper 等大或多一个。
// 5. 奇数取 lower 最大值，偶数取 lower 最大与 upper 最小的平均。
```

#### C++ 实现

```cpp
#include <queue>
#include <deque>
#include <vector>
#include <utility>
#include <iostream>

class StockPriceMedian {
private:
    std::deque<std::pair<int, int>> window; // 滑动窗口存储(time, value)
    std::priority_queue<int> maxHeap; // 最大堆，存储左半部分
    std::priority_queue<int, std::vector<int>, std::greater<int>> minHeap; // 最小堆，存储右半部分

    // 平衡两个堆，保证最大堆的大小始终 >= 最小堆的大小
    void balanceHeaps() {
        if (maxHeap.size() > minHeap.size() + 1) {
            minHeap.push(maxHeap.top());
            maxHeap.pop();
        } else if (minHeap.size() > maxHeap.size()) {
            maxHeap.push(minHeap.top());
            minHeap.pop();
        }
    }

    // 插入新值到两个堆中
    void insertPrice(int price) {
        if (maxHeap.empty() || price <= maxHeap.top()) {
            maxHeap.push(price);
        } else {
            minHeap.push(price);
        }
        balanceHeaps();
    }

    // 从堆中删除指定值
    void removePrice(int price) {
        // 标记为懒删除，等到堆顶出现过期元素时移除
        if (!maxHeap.empty() && price <= maxHeap.top()) {
            // 将最大堆中的元素移除
            maxHeap.pop();
        } else {
            // 将最小堆中的元素移除
            minHeap.pop();
        }
        balanceHeaps();
    }

public:
    double insert(int time, int price) {
        // 插入新数据
        window.push_back({time, price});
        insertPrice(price);

        // 移除超过10分钟的数据
        while (!window.empty() && window.front().first <= time - 600) {
            removePrice(window.front().second);
            window.pop_front();
        }

        // 计算中位数
        if (maxHeap.size() == minHeap.size()) {
            return (maxHeap.top() + minHeap.top()) / 2.0;
        } else {
            return maxHeap.top();
        }
    }
};
```

#### Python 实现

```python
from bisect import bisect_left, insort

def median_sliding_window(nums, k):
    window, answer = sorted(nums[:k]), []
    for right in range(k, len(nums) + 1):
        answer.append(window[k // 2] if k % 2 else (window[k // 2 - 1] + window[k // 2]) / 2)
        if right == len(nums): break
        window.pop(bisect_left(window, nums[right - k]))
        insort(window, nums[right])
    return answer
```


### 并行归并排序与快速排序


**形象理解**：左右子区间像两组互不干扰的工人，可以同时工作；只有归并或 partition 形成的数据依赖点需要等待。

#### 执行步骤

```text
// 并行归并：创建左右排序 task -> taskwait -> 合并两个有序区间。
// 并行快排：先完成三路 partition -> 并行排序小于区和大于区。
// 小区间低于 cutoff 时改用串行排序，避免任务调度成本超过计算。
// OpenMP 线程池控制实际线程数，递归任务数不等于线程数。
```

#### C++ 实现

```cpp
#include <algorithm>
#include <vector>

void mergeRange(std::vector<int>& nums,
                std::vector<int>& buffer,
                int left,
                int mid,
                int right) {
    int i = left;
    int j = mid + 1;
    int k = left;

    while (i <= mid && j <= right) {
        if (nums[i] <= nums[j]) {
            buffer[k++] = nums[i++];
        } else {
            buffer[k++] = nums[j++];
        }
    }
    while (i <= mid) {
        buffer[k++] = nums[i++];
    }
    while (j <= right) {
        buffer[k++] = nums[j++];
    }
    std::copy(buffer.begin() + left,
              buffer.begin() + right + 1,
              nums.begin() + left);
}

void parallelMergeSortImpl(std::vector<int>& nums,
                           std::vector<int>& buffer,
                           int left,
                           int right,
                           int cutoff) {
    if (left >= right) {
        return;
    }

    if (right - left + 1 <= cutoff) {
        std::stable_sort(nums.begin() + left, nums.begin() + right + 1);
        return;
    }

    int mid = left + (right - left) / 2;

    #pragma omp task shared(nums, buffer) firstprivate(left, mid, cutoff)
    parallelMergeSortImpl(nums, buffer, left, mid, cutoff);

    #pragma omp task shared(nums, buffer) firstprivate(mid, right, cutoff)
    parallelMergeSortImpl(nums, buffer, mid + 1, right, cutoff);

    #pragma omp taskwait
    mergeRange(nums, buffer, left, mid, right);
}

void parallelMergeSort(std::vector<int>& nums, int cutoff = 1 << 14) {
    if (nums.size() < 2) {
        return;
    }

    std::vector<int> buffer(nums.size());

    #pragma omp parallel
    {
        #pragma omp single nowait
        parallelMergeSortImpl(nums, buffer, 0,
                              static_cast<int>(nums.size()) - 1,
                              cutoff);
    }
}
```

#### Python 实现

```python
from concurrent.futures import ProcessPoolExecutor

def parallel_sort(chunks):
    with ProcessPoolExecutor() as pool:
        sorted_chunks = list(pool.map(sorted, chunks))
    import heapq
    return list(heapq.merge(*sorted_chunks))
```


### LRU 缓存


**形象理解**：像把最近借过的书放回书架最前面，最久没碰的书逐渐沉到末尾。哈希表负责瞬间找到书，双向链表负责瞬间调整书的位置。

#### 执行步骤

```text
// 1. get(key)：哈希表不存在就返回 -1。
// 2. 存在时读取节点值，并用 touch/splice 把节点移到链表头部。
// 3. put 已存在 key：更新值，再 touch 为最近使用。
// 4. put 新 key：在表头插入节点，并把迭代器写入哈希表。
// 5. 超容量时删除链表尾部 LRU 节点，同时从哈希表删除其 key。
```

#### C++ 实现

```cpp
class LRUCache {
public:
    explicit LRUCache(int capacity) : capacity_(capacity) {}

    int get(int key) {
        auto it = index_.find(key);
        if (it == index_.end()) {
            return -1;
        }
        touch(it->second);
        return it->second->second;
    }

    void put(int key, int value) {
        auto it = index_.find(key);
        if (it != index_.end()) {
            it->second->second = value;
            touch(it->second);
            return;
        }

        if (static_cast<int>(cache_.size()) == capacity_) {
            index_.erase(cache_.back().first);
            cache_.pop_back();
        }

        cache_.emplace_front(key, value);
        index_[key] = cache_.begin();
    }

private:
    using ListIterator = std::list<std::pair<int, int>>::iterator;

    void touch(ListIterator it) {
        cache_.splice(cache_.begin(), cache_, it);
    }

    int capacity_;
    std::list<std::pair<int, int>> cache_;  // MRU -> LRU
    std::unordered_map<int, ListIterator> index_;
};
```

#### Python 实现

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity, self.cache = capacity, OrderedDict()

    def get(self, key):
        if key not in self.cache: return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache: self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity: self.cache.popitem(last=False)
```


### LFU 缓存


**形象理解**：先按使用次数把书分到不同楼层，同一楼层再按最近使用排序。淘汰时去最低频楼层，拿走其中最久没用的一本。

#### 执行步骤

```text
// 1. key 表定位 value、frequency 和其在频率链表中的位置。
// 2. get 命中后将节点从频率 f 的链表移到 f+1 链表头。
// 3. 原最低频链表变空时，minFrequency++。
// 4. put 超容量时删除 minFrequency 链表尾部节点。
// 5. 新节点频率为 1，因此插入后把 minFrequency 重置为 1。
```

#### C++ 实现

```cpp
class LFUCache {
public:
    explicit LFUCache(int capacity) : capacity_(capacity) {}

    int get(int key) {
        auto it = entries_.find(key);
        if (it == entries_.end()) {
            return -1;
        }
        increaseFrequency(key, it->second);
        return it->second.value;
    }

    void put(int key, int value) {
        if (capacity_ == 0) {
            return;
        }

        auto it = entries_.find(key);
        if (it != entries_.end()) {
            it->second.value = value;
            increaseFrequency(key, it->second);
            return;
        }

        if (static_cast<int>(entries_.size()) == capacity_) {
            auto &bucket = frequencyKeys_[minFrequency_];
            int evictedKey = bucket.back();
            bucket.pop_back();
            if (bucket.empty()) {
                frequencyKeys_.erase(minFrequency_);
            }
            entries_.erase(evictedKey);
        }

        minFrequency_ = 1;
        auto &bucket = frequencyKeys_[1];
        bucket.push_front(key);
        entries_.emplace(key, Entry{value, 1, bucket.begin()});
    }

private:
    struct Entry {
        int value;
        int frequency;
        std::list<int>::iterator position;
    };

    void increaseFrequency(int key, Entry &entry) {
        int oldFrequency = entry.frequency;
        auto bucketIt = frequencyKeys_.find(oldFrequency);
        bucketIt->second.erase(entry.position);

        if (bucketIt->second.empty()) {
            frequencyKeys_.erase(bucketIt);
            if (minFrequency_ == oldFrequency) {
                ++minFrequency_;
            }
        }

        ++entry.frequency;
        auto &newBucket = frequencyKeys_[entry.frequency];
        newBucket.push_front(key);
        entry.position = newBucket.begin();
    }

    int capacity_;
    int minFrequency_ = 0;
    std::unordered_map<int, Entry> entries_;
    std::unordered_map<int, std::list<int>> frequencyKeys_;
};
```

#### Python 实现

```python
from collections import defaultdict, OrderedDict

class LFUCache:
    def __init__(self, capacity):
        self.capacity, self.minimum = capacity, 0
        self.values, self.frequencies = {}, {}
        self.groups = defaultdict(OrderedDict)

    def _touch(self, key):
        frequency = self.frequencies[key]
        del self.groups[frequency][key]
        if not self.groups[frequency] and self.minimum == frequency:
            self.minimum += 1
        self.frequencies[key] = frequency + 1
        self.groups[frequency + 1][key] = None

    def get(self, key):
        if key not in self.values: return -1
        self._touch(key)
        return self.values[key]

    def put(self, key, value):
        if self.capacity == 0: return
        if key in self.values:
            self.values[key] = value
            self._touch(key)
            return
        if len(self.values) == self.capacity:
            old, _ = self.groups[self.minimum].popitem(last=False)
            del self.values[old], self.frequencies[old]
        self.values[key], self.frequencies[key], self.minimum = value, 1, 1
        self.groups[1][key] = None
```


### 基于时间的键值存储 TimeMap


**形象理解**：每个 key 都有一本按时间追加的档案。查询时间 t 时要找“不晚于 t 的最后一条记录”，即第一个大于 t 的位置再向前一步。

#### 执行步骤

```text
// 1. set 把 (timestamp,value) 追加到该 key 的有序 vector。
// 2. get 先检查 key 是否存在，不存在返回空串。
// 3. 用 upper_bound 找到第一个 timestamp > target 的记录。
// 4. 迭代器位于 begin 说明没有不晚于 target 的值。
// 5. 否则 --it，返回最后一条 timestamp <= target 的 value。
```

#### C++ 实现

```cpp
unordered_map<string, vector<pair<int, string>>> history_;
```

#### Python 实现

```python
from bisect import bisect_right
from collections import defaultdict

class TimeMap:
    def __init__(self): self.data = defaultdict(list)
    def set(self, key, value, timestamp): self.data[key].append((timestamp, value))
    def get(self, key, timestamp):
        records = self.data[key]
        index = bisect_right(records, (timestamp, chr(0x10ffff))) - 1
        return records[index][1] if index >= 0 else ""
```


### 带 TTL 的令牌验证系统


**形象理解**：哈希表是“每张门票当前有效期”，队列是“按过期时间排列的清理提醒”。续期会留下旧提醒，清理时必须核对它是否仍代表最新有效期。

#### 执行步骤

```text
// 1. generate：计算 expire=currentTime+ttl，写哈希表并追加一条过期记录。
// 2. renew 前先清理，令牌不存在说明已过期，不能续期。
// 3. 续期时覆盖哈希表中的最新 expire，并再追加新记录。
// 4. 清理队首时，只有“已到期且等于哈希表当前 expire”的记录才能删除令牌。
// 5. 若过期记录时间不等于当前 expire，它只是续期前的陈旧提醒，跳过即可。
```

#### C++ 实现

```cpp
class AuthenticationManager {
public:
    explicit AuthenticationManager(int timeToLive)
        : timeToLive_(timeToLive) {}

    void generate(std::string tokenId, int currentTime) {
        int expirationTime = currentTime + timeToLive_;
        expiration_[tokenId] = expirationTime;
        timeline_.emplace_back(expirationTime, std::move(tokenId));
    }

    void renew(std::string tokenId, int currentTime) {
        removeExpired(currentTime);
        auto it = expiration_.find(tokenId);
        if (it == expiration_.end()) {
            return;
        }

        int expirationTime = currentTime + timeToLive_;
        it->second = expirationTime;
        timeline_.emplace_back(expirationTime, std::move(tokenId));
    }

    int countUnexpiredTokens(int currentTime) {
        removeExpired(currentTime);
        return static_cast<int>(expiration_.size());
    }

private:
    void removeExpired(int currentTime) {
        while (!timeline_.empty() && timeline_.front().first <= currentTime) {
            auto [expirationTime, tokenId] = std::move(timeline_.front());
            timeline_.pop_front();

            auto it = expiration_.find(tokenId);
            if (it != expiration_.end() && it->second == expirationTime) {
                expiration_.erase(it);
            }
        }
    }

    int timeToLive_;
    std::unordered_map<std::string, int> expiration_;
    std::deque<std::pair<int, std::string>> timeline_;
};
```

#### Python 实现

```python
class AuthenticationManager:
    def __init__(self, timeToLive):
        self.ttl, self.expiration = timeToLive, {}
    def generate(self, tokenId, currentTime):
        self.expiration[tokenId] = currentTime + self.ttl
    def renew(self, tokenId, currentTime):
        if self.expiration.get(tokenId, 0) > currentTime:
            self.expiration[tokenId] = currentTime + self.ttl
    def countUnexpiredTokens(self, currentTime):
        return sum(end > currentTime for end in self.expiration.values())
```


### 全 O(1) 数据结构 AllOne


**形象理解**：每个计数值是一节车厢，同计数 key 坐在同一车厢；加一或减一只会走到相邻车厢，空车厢立即摘掉，因此首尾始终是最小和最大计数。

#### 执行步骤

```text
// 1. 双向链表按 count 递增保存 bucket，每个 bucket 内是 key 集合。
// 2. 哈希表让 key 直接定位所在 bucket。
// 3. inc/dec 只检查相邻 count±1 的桶；不存在就原地插入新桶。
// 4. key 移到新桶后，从旧桶删除；旧桶为空就从链表删除。
// 5. 链表头桶任意 key 是最小值，尾桶任意 key 是最大值。
```

#### C++ 实现

```cpp
class AllOne {
public:
    void inc(std::string key) {
        auto mapIt = positions_.find(key);
        if (mapIt == positions_.end()) {
            auto bucket = buckets_.begin();
            if (bucket == buckets_.end() || bucket->count != 1) {
                bucket = buckets_.insert(bucket, Bucket{1, {}});
            }
            bucket->keys.insert(key);
            positions_[std::move(key)] = bucket;
            return;
        }

        auto current = mapIt->second;
        auto next = std::next(current);
        if (next == buckets_.end() || next->count != current->count + 1) {
            next = buckets_.insert(next, Bucket{current->count + 1, {}});
        }

        next->keys.insert(key);
        mapIt->second = next;
        eraseFromBucket(current, key);
    }

    void dec(std::string key) {
        auto mapIt = positions_.find(key);
        auto current = mapIt->second;

        if (current->count == 1) {
            positions_.erase(mapIt);
        } else {
            auto previous = current == buckets_.begin()
                ? buckets_.end()
                : std::prev(current);

            if (previous == buckets_.end() ||
                previous->count != current->count - 1) {
                previous = buckets_.insert(
                    current, Bucket{current->count - 1, {}});
            }

            previous->keys.insert(key);
            mapIt->second = previous;
        }

        eraseFromBucket(current, key);
    }

    std::string getMaxKey() {
        return buckets_.empty() ? "" : *buckets_.back().keys.begin();
    }

    std::string getMinKey() {
        return buckets_.empty() ? "" : *buckets_.front().keys.begin();
    }

private:
    struct Bucket {
        int count;
        std::unordered_set<std::string> keys;
    };

    using BucketIterator = std::list<Bucket>::iterator;

    void eraseFromBucket(BucketIterator bucket, const std::string &key) {
        bucket->keys.erase(key);
        if (bucket->keys.empty()) {
            buckets_.erase(bucket);
        }
    }

    std::list<Bucket> buckets_;
    std::unordered_map<std::string, BucketIterator> positions_;
};
```

#### Python 实现

```python
class Bucket:
    def __init__(self, count=0):
        self.count, self.keys = count, set()
        self.previous = self.next = None

class AllOne:
    def __init__(self):
        self.root = Bucket()
        self.root.previous = self.root.next = self.root
        self.location = {}

    def _insert_after(self, node, count):
        bucket = Bucket(count)
        bucket.previous, bucket.next = node, node.next
        node.next.previous = bucket
        node.next = bucket
        return bucket

    def _remove_if_empty(self, bucket):
        if bucket is not self.root and not bucket.keys:
            bucket.previous.next = bucket.next
            bucket.next.previous = bucket.previous

    def inc(self, key):
        current = self.location.get(key, self.root)
        target_count = current.count + 1
        target = current.next
        if target is self.root or target.count != target_count:
            target = self._insert_after(current, target_count)
        target.keys.add(key)
        self.location[key] = target
        if current is not self.root:
            current.keys.remove(key)
            self._remove_if_empty(current)

    def dec(self, key):
        current = self.location[key]
        current.keys.remove(key)
        if current.count == 1:
            del self.location[key]
        else:
            target = current.previous
            if target is self.root or target.count != current.count - 1:
                target = self._insert_after(current.previous, current.count - 1)
            target.keys.add(key)
            self.location[key] = target
        self._remove_if_empty(current)

    def getMaxKey(self):
        return next(iter(self.root.previous.keys), "")

    def getMinKey(self):
        return next(iter(self.root.next.keys), "")
```


## 二叉树的遍历、属性与构造

### 递归前序、中序和后序遍历


**形象理解**：递归像让每个节点都执行同一张工作单。前序是“先登记自己，再访问孩子”，中序是“左边回来后登记自己”，后序是“两个孩子都回来后再登记自己”。

#### 执行步骤

```text
// 1. 当前节点为空时直接返回，这是递归出口。
// 2. 前序：记录 root -> 递归左树 -> 递归右树。
// 3. 中序：递归左树 -> 记录 root -> 递归右树。
// 4. 后序：递归左树 -> 递归右树 -> 记录 root。
// 5. 三种写法只改变“处理当前节点”所在的位置。
```

#### C++ 实现

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

```cpp
void traversal(TreeNode* cur, vector<int>& vec) {
    if (cur == NULL) return;
    traversal(cur->left, vec);  // 左
    vec.push_back(cur->val);    // 中
    traversal(cur->right, vec); // 右
}
```

```cpp
void traversal(TreeNode* cur, vector<int>& vec) {
    if (cur == NULL) return;
    traversal(cur->left, vec);  // 左
    traversal(cur->right, vec); // 右
    vec.push_back(cur->val);    // 中
}
```

#### Python 实现

```python
def traversals(root):
    preorder, inorder, postorder = [], [], []
    def visit(node):
        if not node: return
        preorder.append(node.val)
        visit(node.left)
        inorder.append(node.val)
        visit(node.right)
        postorder.append(node.val)
    visit(root)
    return preorder, inorder, postorder
```


### 迭代前序、中序和后序遍历


**形象理解**：递归原本由系统调用栈替你记住“下一步回到哪里”，迭代只是把这个栈拿到自己手里。中序尤其像沿左侧楼梯走到底，再逐层返回并转向右侧。

#### 执行步骤

```text
// 前序：根先入栈；弹出就记录，再先压右孩子、后压左孩子，保证左边先处理。
// 中序：当前节点一路压栈并向左；走空后弹栈记录，再转到它的右孩子。
// 后序：按“根-右-左”收集，最后整体反转为“左-右-根”。
// 每个节点只入栈、出栈一次，因此时间 O(n)，额外空间 O(h)。
```

#### C++ 实现

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

#### Python 实现

```python
def inorder_traversal(root):
    answer, stack, node = [], [], root
    while stack or node:
        while node: stack.append(node); node = node.left
        node = stack.pop(); answer.append(node.val); node = node.right
    return answer
```


### 统一格式迭代遍历


**形象理解**：栈里同时放“待访问节点”和“待执行任务”。空指针标记相当于一张便签：再次看到它时，不再展开节点，而是把紧邻的节点值写入结果。

#### 执行步骤

```text
// 1. 弹出普通节点时，按目标遍历顺序的逆序把孩子、自己和标记压栈。
// 2. “节点后跟 nullptr”表示这个节点下次出现时应被处理。
// 3. 弹出 nullptr 后，再弹出相邻节点并记录其值。
// 4. 只需调整压栈顺序，就能统一实现前序、中序和后序。
```

#### C++ 实现

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

#### Python 实现

```python
def preorder_unified(root):
    answer, stack = [], [(root, False)]
    while stack:
        node, visited = stack.pop()
        if not node: continue
        if visited: answer.append(node.val)
        else:
            stack.extend([(node.right, False), (node.left, False), (node, True)])
    return answer
```


### 二叉树层序遍历


**形象理解**：像水波从树根一圈圈向外扩散。队列中当前已有的元素正好构成一层，先记住这一层人数，处理时加入的孩子留给下一轮。

#### 执行步骤

```text
// 1. 根节点入队；队列为空表示所有层都已访问。
// 2. 每轮先保存 size = queue.size()，锁定当前层节点数。
// 3. 连续弹出 size 个节点，记录值并把非空孩子入队。
// 4. 当前层结果单独加入答案，随后开始下一层。
```

#### C++ 实现

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

#### Python 实现

```python
from collections import deque

def level_order(root):
    if not root: return []
    queue, answer = deque([root]), []
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft(); level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        answer.append(level)
    return answer
```


### 二叉树的右视图


**形象理解**：站在树的右边，每一层只能看见最右侧节点。层序遍历时，每层最后一个出队的人就是答案。

#### 执行步骤

```text
// 1. 按层遍历并保存当前层节点数 size。
// 2. 依次弹出这一层节点，同时照常加入左右孩子。
// 3. 当 i == size - 1 时，记录该节点值。
// 4. 每层恰好记录一次，得到右视图。
```

#### C++ 实现

```cpp
vector<int> rightSideView(TreeNode* root) {
    unordered_map<int, int> rightmostValueAtDepth;
    int max_depth = -1;

    stack<TreeNode*> nodeStack;
    stack<int> depthStack;
    nodeStack.push(root);
    depthStack.push(0);

    while (!nodeStack.empty()) {
        TreeNode* node = nodeStack.top();nodeStack.pop();
        int depth = depthStack.top();depthStack.pop();

        if (node != NULL) {
            // 维护二叉树的最大深度
            max_depth = max(max_depth, depth);

            // 如果不存在对应深度的节点我们才插入
            if (rightmostValueAtDepth.find(depth) == rightmostValueAtDepth.end()) {
                rightmostValueAtDepth[depth] =  node -> val;
            }

            nodeStack.push(node -> left);
            nodeStack.push(node -> right);
            depthStack.push(depth + 1);
            depthStack.push(depth + 1);
        }
    }

    vector<int> rightView;
    for (int depth = 0; depth <= max_depth; ++depth) {
        rightView.push_back(rightmostValueAtDepth[depth]);
    }

    return rightView;
}
```

#### Python 实现

```python
def right_side_view(root):
    return [level[-1] for level in level_order(root)]
```


### 二叉树每层的平均值


**形象理解**：把每一层当成一个班级，先统计总分，再除以该层人数。总和使用更宽类型，避免节点较多时溢出。

#### 执行步骤

```text
// 1. 层序遍历开始时保存本层 size。
// 2. 用 double 或 long long 累加这一层全部节点值。
// 3. 节点的孩子继续入队，供下一层使用。
// 4. 本层结束后将 sum / size 加入答案。
```

#### C++ 实现

```cpp
vector<double> averageOfLevels(TreeNode* root) {
        auto counts = vector<int>();
        auto sums = vector<double>();
        dfs(root, 0, counts, sums);
        auto averages = vector<double>();
        int size = sums.size();
        for (int i = 0; i < size; i++) {
            averages.push_back(sums[i] / counts[i]);
        }
        return averages;
    }

    void dfs(TreeNode* root, int level, vector<int> &counts, vector<double> &sums) {
        if (root == nullptr) {
            return;
        }
        if (level < sums.size()) {
            sums[level] += root->val;
            counts[level] += 1;
        } else {
            sums.push_back(1.0 * root->val);
            counts.push_back(1);
        }
        dfs(root->left, level + 1, counts, sums);
        dfs(root->right, level + 1, counts, sums);
    }
```

#### Python 实现

```python
def average_of_levels(root):
    return [sum(level) / len(level) for level in level_order(root)]
```


### 填充每个节点的下一个右侧节点指针


**形象理解**：每层节点像排成一队，处理到第二个人时，就能让前一个人的 `next` 指向当前人；队尾自然指向空。

#### 执行步骤

```text
// 1. 用队列逐层取出节点。
// 2. prev 保存这一层刚处理过的前一个节点。
// 3. 当前节点到来时，若 prev 非空就令 prev->next = current。
// 4. 更新 prev，并把当前节点孩子入队。
// 5. 每层最后一个节点的 next 保持 nullptr。
```

#### C++ 实现

```cpp
Node* connect(Node* root) {
    if (root == nullptr) {
        return root;
    }
    // 从根节点开始
    Node* leftmost = root;
    while (leftmost->left != nullptr) {
        // 遍历这一层节点组织成的链表，为下一层的节点更新 next 指针
        Node* head = leftmost;
        while (head != nullptr) {
            // CONNECTION 1
            head->left->next = head->right;
            // CONNECTION 2
            if (head->next != nullptr) {
                head->right->next = head->next->left;
            }
            // 指针向后移动
            head = head->next;
        }
        // 去下一层的最左的节点
        leftmost = leftmost->left;
    }
    return root;
}
```

```cpp
void handle(Node* &last, Node* &p, Node* &nextStart) {
    if (last) {
        last->next = p;
    }
    if (!nextStart) {
        nextStart = p;
    }
    last = p;
}

Node* connect(Node* root) {
    if (!root) {
        return nullptr;
    }
    Node *start = root;
    while (start) {
        Node *last = nullptr, *nextStart = nullptr;
        for (Node *p = start; p != nullptr; p = p->next) {
            if (p->left) {
                handle(last, p->left, nextStart);
            }
            if (p->right) {
                handle(last, p->right, nextStart);
            }
        }
        start = nextStart;
    }
    return root;
}
```

#### Python 实现

```python
def connect(root):
    level = root
    while level and level.left:
        node = level
        while node:
            node.left.next = node.right
            if node.next: node.right.next = node.next.left
            node = node.next
        level = level.left
    return root
```


### 对称二叉树


**形象理解**：不是比较两棵子树相同位置，而是把它们放到镜子两侧：左边的外侧要对上右边的外侧，左边的内侧要对上右边的内侧。

#### 执行步骤

```text
// 1. 同时传入左子树和右子树的根。
// 2. 两者都空返回 true；只有一个空或值不同返回 false。
// 3. 比较 left->left 与 right->right 这对外侧节点。
// 4. 比较 left->right 与 right->left 这对内侧节点。
// 5. 两组都对称，整棵树才对称。
```

#### C++ 实现

```cpp
 bool compare(TreeNode* left, TreeNode* right) {
    // 首先排除空节点的情况
    if (left == NULL && right != NULL) return false;
    else if (left != NULL && right == NULL) return false;
    else if (left == NULL && right == NULL) return true;
    // 排除了空节点，再排除数值不相同的情况
    else if (left->val != right->val) return false;

    // 此时就是：左右节点都不为空，且数值相同的情况
    // 此时才做递归，做下一层的判断
    bool outside = compare(left->left, right->right);   // 左子树：左、 右子树：右
    bool inside = compare(left->right, right->left);    // 左子树：右、 右子树：左
    bool isSame = outside && inside;                    // 左子树：中、 右子树：中 （逻辑处理）
    return isSame;

}
bool isSymmetric(TreeNode* root) {
    if (root == NULL) return true;
    return compare(root->left, root->right);
}
```

#### Python 实现

```python
def is_symmetric(root):
    def mirror(left, right):
        if not left or not right: return left is right
        return left.val == right.val and mirror(left.left, right.right) and mirror(left.right, right.left)
    return mirror(root.left, root.right) if root else True
```


### 二叉树和 N 叉树的最大深度


**形象理解**：树高等于“最高孩子的身高再加自己这一层”。N 叉树只是孩子从两个变成一组，核心仍是从所有孩子中挑最大值。

#### 执行步骤

```text
// 1. 空节点深度为 0。
// 2. 二叉树分别递归计算 leftDepth 和 rightDepth。
// 3. N 叉树遍历 children，维护最大的 childDepth。
// 4. 返回 maxChildDepth + 1，把当前节点这一层算进去。
```

#### C++ 实现

```cpp
int maxDepth(TreeNode* root) {
    if (root == null) return 0;
    return 1 + max(maxDepth(root->left), maxDepth(root->right));
}
```

```cpp
int result;
void getdepth(TreeNode* node, int depth) {
    result = depth > result ? depth : result; // 中

    if (node->left == NULL && node->right == NULL) return ;

    if (node->left) { // 左
        depth++;    // 深度+1
        getdepth(node->left, depth);
        depth--;    // 回溯，深度-1
    }
    if (node->right) { // 右
        depth++;    // 深度+1
        getdepth(node->right, depth);
        depth--;    // 回溯，深度-1
    }
    return ;
}
int maxDepth(TreeNode* root) {
    result = 0;
    if (root == NULL) return result;
    getdepth(root, 1);
    return result;
}
```

```cpp
int maxDepth(Node* root) {
    if(root ==nullptr)
        return 0;
    int depth = 0;
    for(auto &node : root->children){
        depth = max(depth, maxDepth(node));
    }
    return depth+1;
}
```

#### Python 实现

```python
def max_depth(root):
    if not root: return 0
    children = getattr(root, "children", None)
    if children is not None:
        return 1 + max((max_depth(child) for child in children), default=0)
    return 1 + max(max_depth(root.left), max_depth(root.right))
```


### 二叉树的最小深度


**形象理解**：最小深度必须走到真正的叶子，不能在“只有一边为空”时提前下车。单孩子节点只能沿存在的那一侧继续走。

#### 执行步骤

```text
// 1. 空节点返回 0。
// 2. 左孩子为空时，只能返回右子树深度 + 1。
// 3. 右孩子为空时，只能返回左子树深度 + 1。
// 4. 两个孩子都存在时，取二者较小值 + 1。
```

#### C++ 实现

```cpp
int minDepth(TreeNode* root) {
    if (root == NULL) return 0;
    if (root->left == NULL && root->right != NULL) {
        return 1 + minDepth(root->right);
    }
    if (root->left != NULL && root->right == NULL) {
        return 1 + minDepth(root->left);
    }
    return 1 + min(minDepth(root->left), minDepth(root->right));
}
```

#### Python 实现

```python
def min_depth(root):
    if not root: return 0
    if not root.left: return 1 + min_depth(root.right)
    if not root.right: return 1 + min_depth(root.left)
    return 1 + min(min_depth(root.left), min_depth(root.right))
```


### 平衡二叉树


**形象理解**：每个节点都是一架天平。孩子先报告自己的高度；任何孩子已经失衡，或两侧高度差超过 1，就用 `-1` 一路上报故障。

#### 执行步骤

```text
// 1. 后序递归先计算左右子树高度。
// 2. 任一侧返回 -1，当前节点无需继续计算，也返回 -1。
// 3. abs(leftHeight - rightHeight) > 1 时返回 -1。
// 4. 否则返回 max(leftHeight, rightHeight) + 1。
// 5. 根节点结果不是 -1 就说明整棵树平衡。
```

#### C++ 实现

```cpp
int getHeight(TreeNode* node) {
    if (node == NULL) {
        return 0;
    }
    int leftHeight = getHeight(node->left);
    if (leftHeight == -1) return -1;
    int rightHeight = getHeight(node->right);
    if (rightHeight == -1) return -1;
    return abs(leftHeight - rightHeight) > 1 ? -1 : 1 + max(leftHeight, rightHeight);
}
bool isBalanced(TreeNode* root) {
    return getHeight(root) == -1 ? false : true;
}
```

#### Python 实现

```python
def is_balanced(root):
    def height(node):
        if not node: return 0
        left, right = height(node.left), height(node.right)
        if left < 0 or right < 0 or abs(left - right) > 1: return -1
        return 1 + max(left, right)
    return height(root) >= 0
```


### 二叉树的所有路径


**形象理解**：路径像旅行清单。进入节点时写下名字，走到叶子时拍照保存；回到岔路口前擦掉刚才那一站，才能复用清单探索另一条路。

#### 执行步骤

```text
// 1. 将当前节点加入 path。
// 2. 当前节点是叶子时，把 path 格式化后加入答案。
// 3. 否则分别递归存在的左、右孩子。
// 4. 每次子调用返回后 pop_back，撤销本次选择。
```

#### C++ 实现

```cpp
void traversal(TreeNode* cur, string path, vector<string>& result) {
    path += to_string(cur->val); // 中
    if (cur->left == NULL && cur->right == NULL) {
        result.push_back(path);
        return;
    }
    if (cur->left) traversal(cur->left, path + "->", result); // 左
    if (cur->right) traversal(cur->right, path + "->", result); // 右
}
vector<string> binaryTreePaths(TreeNode* root) {
    vector<string> result;
    string path;
    if (root == NULL) return result;
    traversal(root, path, result);
    return result;

}
```

#### Python 实现

```python
def binary_tree_paths(root):
    if not root: return []
    if not root.left and not root.right: return [str(root.val)]
    return [f"{root.val}->{path}" for child in (root.left, root.right) if child for path in binary_tree_paths(child)]
```


### 路径总和


**形象理解**：把目标和当成旅行预算，每经过一个节点就扣除节点值；只有走到叶子并且预算恰好清零，才把这条完整路线拍照保存。随后仍要返回岔路口，继续寻找其他路线。

#### 执行步骤

```text
// 1. path 先放入根节点，并从 target 中扣除根值。
// 2. 选择一个非空孩子：把孩子值加入 path，并从 remaining 中扣除。
// 3. 到达叶子且 remaining == 0 时，把 path 复制进 result。
// 4. 递归返回后把孩子值加回 remaining，并从 path 弹出。
// 5. 左右两侧都要遍历，不能找到第一条后就提前结束。
```

#### C++ 实现

```cpp
vector<vector<int>> result;
vector<int> path;
// 递归函数不需要返回值，因为我们要遍历整个树
void traversal(TreeNode* cur, int count) {
    if (!cur->left && !cur->right && count == 0) { // 遇到了叶子节点且找到了和为sum的路径
        result.push_back(path);
        return;
    }

    if (!cur->left && !cur->right) return ; // 遇到叶子节点而没有找到合适的边，直接返回

    if (cur->left) { // 左 （空节点不遍历）
        path.push_back(cur->left->val);
        count -= cur->left->val;
        traversal(cur->left, count);    // 递归
        count += cur->left->val;        // 回溯
        path.pop_back();                // 回溯
    }
    if (cur->right) { // 右 （空节点不遍历）
        path.push_back(cur->right->val);
        count -= cur->right->val;
        traversal(cur->right, count);   // 递归
        count += cur->right->val;       // 回溯
        path.pop_back();                // 回溯
    }
    return ;
}
vector<vector<int>> pathSum(TreeNode* root, int sum) {
    result.clear();
    path.clear();
    if (root == NULL) return result;
    path.push_back(root->val); // 把根节点放进路径
    traversal(root, sum - root->val);
    return result;
}
```

#### Python 实现

```python
def path_sum(root, target):
    answer = []
    def dfs(node, remaining, path):
        if not node: return
        path.append(node.val); remaining -= node.val
        if not node.left and not node.right and remaining == 0: answer.append(path[:])
        dfs(node.left, remaining, path); dfs(node.right, remaining, path)
        path.pop()
    dfs(root, target, [])
    return answer
```


### 左叶子之和


**形象理解**：重点不是“位于左边的节点”，而是“父节点的左孩子恰好是叶子”。因此判断动作应发生在父节点处。

#### 执行步骤

```text
// 1. 当前节点为空就返回 0。
// 2. 若左孩子存在且没有任何孩子，把左孩子值加入结果。
// 3. 无论是否命中，都递归统计左右子树中的其他左叶子。
// 4. 返回当前贡献、左子树贡献和右子树贡献之和。
```

#### C++ 实现

```cpp
int sumOfLeftLeaves(TreeNode* root) {
    if (!root) return 0;
    int leftValue = 0;
    if (root->left && !root->left->left && !root->left->right) {
        leftValue = root->left->val;
    } else {
        leftValue = sumOfLeftLeaves(root->left);
    }
    return leftValue + sumOfLeftLeaves(root->right);
}
```

#### Python 实现

```python
def sum_of_left_leaves(root):
    if not root: return 0
    left = root.left.val if root.left and not root.left.left and not root.left.right else sum_of_left_leaves(root.left)
    return left + sum_of_left_leaves(root.right)
```


### 找树左下角的值


**形象理解**：层序遍历中每到新一层，第一个出队的节点就是该层最左节点；不断覆盖答案，最后留下的就是最深层最左值。

#### 执行步骤

```text
// 1. 根节点入队，逐层遍历。
// 2. 每层开始处理 i == 0 的节点时更新 answer。
// 3. 左右孩子按左后右的顺序入队。
// 4. 最后一层结束后，answer 即树的左下角值。
```

#### C++ 实现

```cpp
int maxDepth = INT_MIN;
int result;
void traversal(TreeNode* root, int depth) {
    if (root->left == NULL && root->right == NULL) {
        if (depth > maxDepth) {
            maxDepth = depth;
            result = root->val;
        }
        return;
    }
    if (root->left) {
        traversal(root->left, depth + 1); // 隐藏着回溯
    }
    if (root->right) {
        traversal(root->right, depth + 1); // 隐藏着回溯
    }
    return;
}
int findBottomLeftValue(TreeNode* root) {
    traversal(root, 0);
    return result;
}
```

#### Python 实现

```python
from collections import deque

def find_bottom_left_value(root):
    queue = deque([root])
    while queue:
        node = queue.popleft()
        if node.right: queue.append(node.right)
        if node.left: queue.append(node.left)
    return node.val
```


### 翻转二叉树


**形象理解**：给树上的每个节点都做同一个动作——交换左右孩子。交换发生在前序还是后序都可以，只要每个节点恰好处理一次。

#### 执行步骤

```text
// 1. 当前节点为空直接返回。
// 2. 交换 root->left 和 root->right。
// 3. 递归翻转交换后的左子树。
// 4. 递归翻转交换后的右子树。
// 5. 返回 root。
```

#### C++ 实现

```cpp
TreeNode* invertTree(TreeNode* root) {
    if (root == NULL) return root;
    swap(root->left, root->right);  // 中
    invertTree(root->left);         // 左
    invertTree(root->right);        // 右
    return root;
}
```

#### Python 实现

```python
def invert_tree(root):
    if root:
        root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root
```


### 根据前序和中序遍历构造二叉树


**形象理解**：前序序列的第一个人一定是组长；在中序序列中找到组长后，左边全属于左组，右边全属于右组，再对两个小组重复这个过程。

#### 执行步骤

```text
// 1. 前序区间首元素确定根节点值。
// 2. 用哈希表在中序序列中 O(1) 找到根位置。
// 3. 中序左段长度决定前序中左子树的边界。
// 4. 按对应区间递归构造左、右子树。
// 5. 区间为空时返回 nullptr。
```

#### C++ 实现

```cpp
TreeNode *helper(vector<int>& preorder, vector<int>& inorder){
    int num = preorder.size();
    if(num == 0) return nullptr;
    TreeNode * root = new TreeNode(preorder.front());
    if(num == 1) return root;
    int part;
    for(int i = 0; i < num; ++i){
        if(inorder[i] == preorder.front())
            part = i;
    }
    vector<int> leftIn(inorder.begin(), inorder.begin()+part);
    vector<int> rightIn(inorder.begin()+part+1, inorder.end());

    vector<int> leftPre(preorder.begin()+1, preorder.begin()+1+part);
    vector<int> rightPre(preorder.begin()+1+part, preorder.end());
    root->left = helper(leftPre,leftIn);
    root->right = helper(rightPre,rightIn);
    return root;
}
TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
    return helper(preorder,inorder);
}
```

#### Python 实现

```python
def build_tree(preorder, inorder):
    positions, pre_index = {value: i for i, value in enumerate(inorder)}, 0
    def build(left, right):
        nonlocal pre_index
        if left > right: return None
        value = preorder[pre_index]; pre_index += 1
        root = TreeNode(value); middle = positions[value]
        root.left, root.right = build(left, middle - 1), build(middle + 1, right)
        return root
    return build(0, len(inorder) - 1)
```


## 二叉搜索树

### 搜索和插入 BST


**形象理解**：BST 像按编号分岔的走廊：小于当前值永远向左，大于当前值永远向右。搜索沿唯一方向走；插入则在第一次遇到空房间时落座。

#### 执行步骤

```text
// 搜索：值相等返回节点；目标更小进入左树，否则进入右树。
// 插入：走到 nullptr 时创建新节点并返回给父节点连接。
// 递归返回时重新接回 root->left 或 root->right。
// BST 的有序性让每层只访问一个分支。
```

#### C++ 实现

```cpp
// 递归
TreeNode *searchBST(TreeNode *root, int val) {
    if (root == nullptr) {
        return nullptr;
    }
    if (val == root->val) {
        return root;
    }
    return searchBST(val < root->val ? root->left : root->right, val);
}
// 迭代
TreeNode* searchBST(TreeNode* root, int val) {
    while (root != NULL) {
        if (root->val > val) root = root->left;
        else if (root->val < val) root = root->right;
        else return root;
    }
    return NULL;
}
```

```cpp
// 递归
TreeNode* insertIntoBST(TreeNode* root, int val) {
    if (root == NULL) {
        TreeNode* node = new TreeNode(val);
        return node;
    }
    if (root->val > val) root->left = insertIntoBST(root->left, val);
    if (root->val < val) root->right = insertIntoBST(root->right, val);
    return root;
}
```

#### Python 实现

```python
def search_bst(root, value):
    while root and root.val != value:
        root = root.left if value < root.val else root.right
    return root

def insert_into_bst(root, value):
    if not root: return TreeNode(value)
    if value < root.val: root.left = insert_into_bst(root.left, value)
    else: root.right = insert_into_bst(root.right, value)
    return root
```


### 删除 BST 节点


**形象理解**：删除没有孩子或只有一个孩子的人，可以让孩子直接顶替；有两个孩子时，请右子树里最小的人接班，才能保持整棵树的排序规则。

#### 执行步骤

```text
// 1. 按目标值大小递归寻找待删节点。
// 2. 无左孩子就返回右孩子，无右孩子就返回左孩子。
// 3. 两个孩子都存在时，找到右子树最左节点（后继）。
// 4. 用后继值覆盖当前节点，再从右子树删除那个后继。
// 5. 返回当前根，让父节点接回修改后的子树。
```

#### C++ 实现

```cpp
 TreeNode* deleteNode(TreeNode* root, int key) {
    if (root == nullptr) return root; // 第一种情况：没找到删除的节点，遍历到空节点直接返回了
    if (root->val == key) {
        // 第二种情况：左右孩子都为空（叶子节点），直接删除节点， 返回NULL为根节点
        if (root->left == nullptr && root->right == nullptr) {
            ///! 内存释放
            delete root;
            return nullptr;
        }
        // 第三种情况：其左孩子为空，右孩子不为空，删除节点，右孩子补位 ，返回右孩子为根节点
        else if (root->left == nullptr) {
            auto retNode = root->right;
            ///! 内存释放
            delete root;
            return retNode;
        }
        // 第四种情况：其右孩子为空，左孩子不为空，删除节点，左孩子补位，返回左孩子为根节点
        else if (root->right == nullptr) {
            auto retNode = root->left;
            ///! 内存释放
            delete root;
            return retNode;
        }
        // 第五种情况：左右孩子节点都不为空，则将删除节点的左子树放到删除节点的右子树的最左面节点的左孩子的位置
        // 并返回删除节点右孩子为新的根节点。
        else {
            TreeNode* cur = root->right; // 找右子树最左面的节点
            while(cur->left != nullptr) {
                cur = cur->left;
            }
            cur->left = root->left; // 把要删除的节点（root）左子树放在cur的左孩子的位置
            TreeNode* tmp = root;   // 把root节点保存一下，下面来删除
            root = root->right;     // 返回旧root的右孩子作为新root
            delete tmp;             // 释放节点内存（这里不写也可以，但C++最好手动释放一下吧）
            return root;
        }
    }
    if (root->val > key) root->left = deleteNode(root->left, key);
    if (root->val < key) root->right = deleteNode(root->right, key);
    return root;
}
```

#### Python 实现

```python
def delete_node(root, key):
    if not root: return None
    if key < root.val: root.left = delete_node(root.left, key)
    elif key > root.val: root.right = delete_node(root.right, key)
    else:
        if not root.left: return root.right
        if not root.right: return root.left
        successor = root.right
        while successor.left: successor = successor.left
        root.val = successor.val
        root.right = delete_node(root.right, successor.val)
    return root
```


### 验证 BST


**形象理解**：只比较节点和直接孩子不够，因为右子树深处也可能混入过小值。中序遍历 BST 应当得到严格递增序列，像检查一列已经排好序的号码。

#### 执行步骤

```text
// 1. 中序遍历先访问左子树。
// 2. 当前值必须严格大于前一个访问值。
// 3. 使用 long long 边界或可空 prev，避免 INT_MIN 特例。
// 4. 更新 prev，再验证右子树。
// 5. 任一处不递增就立即返回 false。
```

#### C++ 实现

```cpp
bool isValidBST(TreeNode* root) {
    stack<TreeNode*> stack;
    long long inorder = (long long)INT_MIN - 1;

    while (!stack.empty() || root != nullptr) {
        while (root != nullptr) {
            stack.push(root);
            root = root -> left;
        }
        root = stack.top();
        stack.pop();
        // 如果中序遍历得到的节点的值小于等于前一个 inorder，说明不是二叉搜索树
        if (root -> val <= inorder) {
            return false;
        }
        inorder = root -> val;
        root = root -> right;
    }
    return true;
}
```

#### Python 实现

```python
def is_valid_bst(root):
    def validate(node, low, high):
        return not node or low < node.val < high and validate(node.left, low, node.val) and validate(node.right, node.val, high)
    return validate(root, float("-inf"), float("inf"))
```


### BST 转双向链表


**形象理解**：中序遍历本来就按从小到大依次“点名”。记住上一个被点名的节点，每来一个新节点就把二者双向牵手。

#### 执行步骤

```text
// 1. 中序递归访问左子树。
// 2. prev 非空时令 prev->right = current、current->left = prev。
// 3. prev 为空说明 current 是链表头，保存 head。
// 4. 更新 prev = current，再访问右子树。
// 5. 若题目要求循环链表，最后连接 head 与 tail。
```

#### C++ 实现

```cpp
Node* treeToDoublyList(Node* root) {
    if (!root) return nullptr;
    Node *first = nullptr, *previous = nullptr;
    function<void(Node*)> inorder = [&](Node* node) {
        if (!node) return;
        inorder(node->left);
        if (previous) previous->right = node, node->left = previous;
        else first = node;
        previous = node;
        inorder(node->right);
    };
    inorder(root);
    first->left = previous;
    previous->right = first;
    return first;
}
```

#### Python 实现

```python
def tree_to_doubly_list(root):
    if not root: return None
    first = previous = None
    def visit(node):
        nonlocal first, previous
        if not node: return
        visit(node.left)
        if previous: previous.right, node.left = node, previous
        else: first = node
        previous = node
        visit(node.right)
    visit(root)
    first.left, previous.right = previous, first
    return first
```


### BST 中的最小绝对差和众数


**形象理解**：中序序列已经排好序，最小差只可能出现在相邻数字之间；相同数字也会连成一段，因此众数可以像数连续车厢一样统计。

#### 执行步骤

```text
// 最小差：中序遍历时用 current - prev 更新答案，然后更新 prev。
// 众数：当前值等于 prev 就 count++，否则把 count 重置为 1。
// count 超过 maxCount 时清空旧答案并加入当前值。
// count 等于 maxCount 时追加当前值；无需额外哈希表。
```

#### C++ 实现

```cpp
void dfs(TreeNode* root, int& pre, int& ans) {
    if (root == nullptr) {
        return;
    }
    dfs(root->left, pre, ans);
    if (pre == -1) {
        pre = root->val;
    } else {
        ans = min(ans, root->val - pre);
        pre = root->val;
    }
    dfs(root->right, pre, ans);
}
int getMinimumDifference(TreeNode* root) {
    int ans = INT_MAX, pre = -1;
    dfs(root, pre, ans);
    return ans;
}
```

```cpp
void searchBST(TreeNode* cur, unordered_map<int, int>& map) { // 前序遍历
    if (cur == NULL) return ;
    map[cur->val]++; // 统计元素频率
    searchBST(cur->left, map);
    searchBST(cur->right, map);
    return ;
}
bool static cmp (const pair<int, int>& a, const pair<int, int>& b) {
    return a.second > b.second;
}
vector<int> findMode(TreeNode* root) {
    unordered_map<int, int> map; // key:元素，value:出现频率
    vector<int> result;
    if (root == NULL) return result;
    searchBST(root, map);
    vector<pair<int, int>> vec(map.begin(), map.end());
    sort(vec.begin(), vec.end(), cmp); // 给频率排个序
    result.push_back(vec[0].first);
    for (int i = 1; i < vec.size(); i++) {
        // 取最高的放到result数组中
        if (vec[i].second == vec[0].second) result.push_back(vec[i].first);
        else break;
    }
    return result;
}
```

```cpp
vector<int> findMode(TreeNode* root) {
    stack<TreeNode*> st;
    TreeNode* cur = root;
    TreeNode* pre = NULL;
    int maxCount = 0; // 最大频率
    int count = 0; // 统计频率
    vector<int> result;
    while (cur != NULL || !st.empty()) {
        if (cur != NULL) { // 指针来访问节点，访问到最底层
            st.push(cur); // 将访问的节点放进栈
            cur = cur->left;                // 左
        } else {
            cur = st.top();
            st.pop();                       // 中
            if (pre == NULL) { // 第一个节点
                count = 1;
            } else if (pre->val == cur->val) { // 与前一个节点数值相同
                count++;
            } else { // 与前一个节点数值不同
                count = 1;
            }
            if (count == maxCount) { // 如果和最大值相同，放进result中
                result.push_back(cur->val);
            }

            if (count > maxCount) { // 如果计数大于最大值频率
                maxCount = count;   // 更新最大频率
                result.clear();     // 很关键的一步，不要忘记清空result，之前result里的元素都失效了
                result.push_back(cur->val);
            }
            pre = cur;
            cur = cur->right;               // 右
        }
    }
    return result;
}
```

#### Python 实现

```python
from collections import Counter

def bst_values(root):
    return bst_values(root.left) + [root.val] + bst_values(root.right) if root else []

def get_minimum_difference(root):
    values = bst_values(root)
    return min(b - a for a, b in zip(values, values[1:]))

def find_mode(root):
    counts = Counter(bst_values(root)); best = max(counts.values())
    return [value for value, count in counts.items() if count == best]
```


### 二叉树与 BST 的最近公共祖先


**形象理解**：普通树中，一个节点若能从左右两边分别收到 p、q 的“找到”信号，它就是汇合点；BST 中则可利用大小关系，p、q 分居当前值两侧时当前节点就是分岔口。

#### 执行步骤

```text
// 普通树：root 为空或等于 p/q 就返回 root。
// 分别递归左右树；两边都非空时返回 root，否则返回非空的一边。
// BST：p、q 都小于 root 就向左，都大于 root 就向右。
// 不再同侧时，root 就是最近公共祖先。
```

#### C++ 实现

```cpp
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if(root == nullptr) return nullptr;
    if(root->val == p->val || root->val == q->val){
        return root;
    }
    TreeNode *left = lowestCommonAncestor(root->left,p,q);
    TreeNode *right = lowestCommonAncestor(root->right,p,q);
    if(left && right)
        return root;
    if(left==nullptr)
        return right;
    if(right ==nullptr)
        return left;
    return nullptr;
}
```

```cpp
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    TreeNode* ancestor = root;
    while (true) {
        if (p->val < ancestor->val && q->val < ancestor->val) {
            ancestor = ancestor->left;
        }
        else if (p->val > ancestor->val && q->val > ancestor->val) {
            ancestor = ancestor->right;
        }
        else {
            break;
        }
    }
    return ancestor;
}
```

#### Python 实现

```python
def lowest_common_ancestor(root, p, q):
    if not root or root is p or root is q: return root
    left, right = lowest_common_ancestor(root.left, p, q), lowest_common_ancestor(root.right, p, q)
    return root if left and right else left or right

def lowest_common_ancestor_bst(root, p, q):
    while root:
        if p.val < root.val and q.val < root.val: root = root.left
        elif p.val > root.val and q.val > root.val: root = root.right
        else: return root
```


### 修剪 BST


**形象理解**：如果当前值小于下界，它的整个左子树只会更小，可以整棵丢弃；值大于上界时同理丢弃整个右子树。

#### 执行步骤

```text
// 1. root 为空直接返回 nullptr。
// 2. root->val < low 时，只需递归修剪并返回右子树。
// 3. root->val > high 时，只需递归修剪并返回左子树。
// 4. 当前值合法时，分别修剪左右孩子并重新接回。
// 5. 返回当前 root。
```

#### C++ 实现

```cpp
TreeNode* trimBST(TreeNode* root, int low, int high) {
    if (root == nullptr) return nullptr;
    if (root->val < low) return trimBST(root->right, low, high);
    if (root->val > high) return trimBST(root->left, low, high);
    root->left = trimBST(root->left, low, high);
    root->right = trimBST(root->right, low, high);
    return root;
}
```

#### Python 实现

```python
def trim_bst(root, low, high):
    if not root: return None
    if root.val < low: return trim_bst(root.right, low, high)
    if root.val > high: return trim_bst(root.left, low, high)
    root.left, root.right = trim_bst(root.left, low, high), trim_bst(root.right, low, high)
    return root
```


### BST 转累加树


**形象理解**：普通中序从小到大；反向中序从大到小。一路维护已经见过的所有更大值之和，当前节点加上它，就得到累加后的值。

#### 执行步骤

```text
// 1. 按“右树 -> 当前节点 -> 左树”反向中序遍历。
// 2. 先处理所有比当前值大的节点。
// 3. sum += root->val，再令 root->val = sum。
// 4. 带着更新后的 sum 继续处理左子树。
```

#### C++ 实现

```cpp
int sum = 0;
TreeNode* convertBST(TreeNode* root) {
    if (root != nullptr) {
        convertBST(root->right);
        sum += root->val;
        root->val = sum;
        convertBST(root->left);
    }
    return root;
}
```

#### Python 实现

```python
def convert_bst(root):
    total = 0
    def visit(node):
        nonlocal total
        if not node: return
        visit(node.right); total += node.val; node.val = total; visit(node.left)
    visit(root)
    return root
```


### 有序数组转平衡 BST


**形象理解**：每次选择数组中点当组长，左右人数最接近；对子区间继续选中点，树就不会偏向一侧。

#### 执行步骤

```text
// 1. 区间为空时返回 nullptr。
// 2. 取 mid = left + (right - left) / 2 创建根节点。
// 3. 用 [left, mid-1] 构造左子树。
// 4. 用 [mid+1, right] 构造右子树。
// 5. 每层规模近似减半，生成高度平衡的 BST。
```

#### C++ 实现

```cpp
TreeNode* traversal(vector<int>& nums, int left, int right) {
    if (left > right) return nullptr;
    int mid = left + ((right - left) / 2);
    TreeNode* root = new TreeNode(nums[mid]);
    root->left = traversal(nums, left, mid - 1);
    root->right = traversal(nums, mid + 1, right);
    return root;
}
TreeNode* sortedArrayToBST(vector<int>& nums) {
    TreeNode* root = traversal(nums, 0, nums.size() - 1);
    return root;
}
```

#### Python 实现

```python
def sorted_array_to_bst(nums):
    if not nums: return None
    middle = len(nums) // 2
    return TreeNode(nums[middle], sorted_array_to_bst(nums[:middle]), sorted_array_to_bst(nums[middle + 1:]))
```


## 单调栈

### 每日温度


**形象理解**：栈里是还在等待更暖一天的人。新温度更高时，它会连续通知栈顶那些更冷的人，并用下标差算出他们等了几天。

#### 执行步骤

```text
// 1. 栈保存尚未找到更高温度的下标，温度从栈底到栈顶递减。
// 2. 当前温度大于栈顶温度时，弹出旧下标。
// 3. answer[old] = currentIndex - old，记录等待天数。
// 4. 重复通知所有更冷下标，再把当前下标入栈。
```

#### C++ 实现

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

#### Python 实现

```python
def daily_temperatures(temperatures):
    answer, stack = [0] * len(temperatures), []
    for i, value in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < value:
            old = stack.pop(); answer[old] = i - old
        stack.append(i)
    return answer
```


### 最大二叉树


**形象理解**：更大的数字会吃掉左边连续比它小的节点并让它们成为左孩子；它又可能成为左侧最近更大节点的右孩子。单调栈正好维护这条父子边界。

#### 执行步骤

```text
// 1. 栈中节点值保持递减。
// 2. 当前值更大时，连续弹栈；最后弹出的节点成为当前节点左孩子。
// 3. 栈仍非空时，当前节点成为栈顶节点的右孩子。
// 4. 当前节点入栈；最终栈底是整棵树的根。
```

#### C++ 实现

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

#### Python 实现

```python
def construct_maximum_binary_tree(nums):
    if not nums: return None
    i = max(range(len(nums)), key=nums.__getitem__)
    return TreeNode(nums[i], construct_maximum_binary_tree(nums[:i]), construct_maximum_binary_tree(nums[i + 1:]))
```


### 接雨水


**形象理解**：栈里保存还没找到右挡板的低洼地。新柱子更高时，弹出的柱子是池底，新的栈顶是左挡板，当前柱子是右挡板。

#### 执行步骤

```text
// 1. 栈保存下标，对应高度单调递减。
// 2. 当前柱更高时弹出 bottom；栈空说明没有左挡板。
// 3. width = current - left - 1。
// 4. boundedHeight = min(height[left], height[current]) - height[bottom]。
// 5. 把 width * boundedHeight 加入总水量，再继续结算更深池子。
```

#### C++ 实现

```cpp
int trap(vector<int>& height) {
    if(height.size() <= 1)
        return 0;
    int res = 0;
    stack<int> stk;
    stk.push(0);
    for(int i = 1; i < height.size(); ++i){
        if (height[i] < height[stk.top()]){
            stk.push(i);
        } else if(height[i] == height[stk.top()]){
            stk.pop();
            stk.push(i);
        }
        else{
            while(!stk.empty() && height[i] > height[stk.top()]){
                int mid = stk.top();
                stk.pop();
                if(!stk.empty()){
                    int left = stk.top();
                    int h = min(height[left], height[i]) - height[mid];
                    res += h * (i-left-1);
                }
            }
            stk.push(i);
        }
    }
    return res;
}
```

#### Python 实现

```python
def trap(height):
    left, right, left_max, right_max, water = 0, len(height) - 1, 0, 0, 0
    while left < right:
        if height[left] < height[right]:
            left_max = max(left_max, height[left]); water += left_max - height[left]; left += 1
        else:
            right_max = max(right_max, height[right]); water += right_max - height[right]; right -= 1
    return water
```


### 柱状图中最大的矩形


**形象理解**：每根柱子都想知道自己能向左右延伸多远。遇到更矮柱子时，栈顶高柱的右边界已经确定，而弹栈后的新栈顶就是它左边第一个更矮位置。

#### 执行步骤

```text
// 1. 首尾加入高度 0 的哨兵，统一清算边界柱子。
// 2. 栈保存高度单调递增的下标。
// 3. 当前高度更小时弹出 mid，将 height[mid] 作为矩形高度。
// 4. 右边界是 current，左边界是弹栈后的 stack.top()。
// 5. width = current - left - 1，用 height[mid] * width 更新最大面积。
```

#### C++ 实现

```cpp
int largestRectangleArea(vector<int>& heights) {
    int res = 0;
    heights.insert(heights.begin(),0);
    heights.push_back(0);
    stack<int> stk;
    stk.push(0);
    for(int i = 1; i < heights.size(); ++i){
        if(heights[i] >= heights[stk.top()]){
            stk.push(i);
        } else{
            while(!stk.empty() && heights[i] < heights[stk.top()]){
                int mid = stk.top();
                stk.pop();
                if(!stk.empty()){
                    int left = stk.top();
                    res = max(res, heights[mid] * (i-left-1));
                }
            }
            stk.push(i);
        }
    }
    return res;
}
```

#### Python 实现

```python
def largest_rectangle_area(heights):
    stack, answer = [-1], 0
    for i, height in enumerate(heights + [0]):
        while stack[-1] != -1 and heights[stack[-1]] >= height:
            answer = max(answer, heights[stack.pop()] * (i - stack[-1] - 1))
        stack.append(i)
    return answer
```


## 回溯

### 组合


**形象理解**：从一排号码中挑 k 个，路径是手里已经拿到的号码，`startIndex` 是下一次只能从哪里往后挑。走满 k 层就拍下一张组合照片。

#### 执行步骤

```text
// 1. path.size() == k 时，把当前组合加入答案并返回。
// 2. i 从 startIndex 开始，保证同一组数字不会换序后重复出现。
// 3. 选择 i：path.push_back(i)。
// 4. 递归下一层：backtrack(i + 1)。
// 5. 撤销选择：path.pop_back()，再尝试下一个 i。
// 6. 上界可剪枝为 n - (k - path.size()) + 1，保证剩余数字够选。
```

#### C++ 实现

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

#### Python 实现

```python
def combine(n, k):
    answer = []
    def dfs(start, path):
        if len(path) == k: answer.append(path[:]); return
        for value in range(start, n - (k - len(path)) + 2):
            path.append(value); dfs(value + 1, path); path.pop()
    dfs(1, [])
    return answer
```


### 组合总和 I


**形象理解**：像用不同面额硬币凑金额，每种面额可反复使用。选择某个数后下一层仍从当前下标开始，而不是从下一个下标开始。

#### 执行步骤

```text
// 1. remaining == 0 时说明正好凑齐，保存 path。
// 2. remaining < 0 时说明超支，立即回退。
// 3. 从 startIndex 枚举候选数并加入 path。
// 4. 递归时仍传 i，允许再次选择 candidates[i]。
// 5. 返回后 pop_back，恢复现场并换下一个数。
```

#### C++ 实现

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

#### Python 实现

```python
def combination_sum(candidates, target):
    answer = []
    def dfs(start, remaining, path):
        if remaining == 0: answer.append(path[:]); return
        for i in range(start, len(candidates)):
            value = candidates[i]
            if value > remaining: break
            path.append(value); dfs(i, remaining - value, path); path.pop()
    candidates.sort(); dfs(0, target, [])
    return answer
```


### 组合总和 II（原记录“组合求和 III”，Leetcode 40）


**形象理解**：每张数字卡只能使用一次，而且相同数字卡很多。先排序，让同一层中相同的卡站在一起；同层跳过重复卡，但不同层仍可选择另一个相同值。

#### 执行步骤

```text
// 1. 先排序 candidates，便于剪枝和去重。
// 2. remaining == 0 时保存答案。
// 3. 同一层若 i > startIndex 且 candidates[i] == candidates[i-1]，跳过。
// 4. 选择 candidates[i] 后递归 i + 1，表示这张卡不能再用。
// 5. 当前数已大于 remaining 时可直接 break。
// 6. 回溯时弹出当前数。
```

#### C++ 实现

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

#### Python 实现

```python
def combination_sum2(candidates, target):
    answer = []
    def dfs(start, remaining, path):
        if remaining == 0: answer.append(path[:]); return
        for i in range(start, len(candidates)):
            if i > start and candidates[i] == candidates[i - 1]: continue
            if candidates[i] > remaining: break
            dfs(i + 1, remaining - candidates[i], path + [candidates[i]])
    candidates.sort(); dfs(0, target, [])
    return answer
```


### 组合总和 III（原记录“组合求和 II”，Leetcode 216）


**形象理解**：只能从 1 到 9 中挑 k 个不同数字凑成 n。既要控制手里卡片数量，也要控制剩余金额；任一条件不可能满足都可提前结束。

#### 执行步骤

```text
// 1. path.size() == k 时，只在 remaining == 0 时保存答案。
// 2. 从 startIndex 到 9 枚举，确保数字不重复且组合有序。
// 3. i > remaining 时，后面数字更大，直接停止本层。
// 4. 选择 i 后递归 i + 1，并令 remaining - i。
// 5. 返回后撤销 i，继续尝试下一数字。
```

#### C++ 实现

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

#### Python 实现

```python
def combination_sum3(k, n):
    answer = []
    def dfs(start, remaining, path):
        if len(path) == k:
            if remaining == 0: answer.append(path)
            return
        for value in range(start, 10):
            if value > remaining: break
            dfs(value + 1, remaining - value, path + [value])
    dfs(1, n, [])
    return answer
```


### 分割回文串


**形象理解**：在字符串的字符缝隙中放剪刀。每次只剪下一段已经确认是回文的片段；走到字符串末尾时，桌上的所有片段就是一种合法切法。

#### 执行步骤

```text
// 1. start == s.size() 时保存当前切分方案。
// 2. end 从 start 向右枚举当前片段终点。
// 3. 若 s[start..end] 不是回文，跳过这把剪刀位置。
// 4. 是回文就加入 path，递归处理 end + 1 之后的字符串。
// 5. 返回后弹出片段，尝试更长的当前片段。
```

#### C++ 实现

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

#### Python 实现

```python
def partition(s):
    answer = []
    def dfs(start, path):
        if start == len(s): answer.append(path); return
        for end in range(start + 1, len(s) + 1):
            part = s[start:end]
            if part == part[::-1]: dfs(end, path + [part])
    dfs(0, [])
    return answer
```


### 复原 IP 地址


**形象理解**：要在数字串中准确插入三个点，切成四段。每段必须是 0 到 255，且除了单独的 0 之外不能有前导零。

#### 执行步骤

```text
// 1. 已得到 4 段时，只有恰好用完整个字符串才保存答案。
// 2. 每段最多向后尝试 3 个字符。
// 3. 遇到前导零、非数字或数值大于 255 时停止扩展。
// 4. 合法片段加入 path，递归处理下一位置。
// 5. 返回后删除该片段，尝试另一个终点。
```

#### C++ 实现

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

#### Python 实现

```python
def restore_ip_addresses(s):
    answer = []
    def dfs(start, parts):
        if len(parts) == 4:
            if start == len(s): answer.append(".".join(parts))
            return
        for end in range(start + 1, min(start + 3, len(s)) + 1):
            part = s[start:end]
            if (part[0] == "0" and len(part) > 1) or int(part) > 255: continue
            dfs(end, parts + [part])
    dfs(0, [])
    return answer
```


### 子集 I


**形象理解**：组合题只在长度达标时拍照，子集题在到达每个节点时都拍照，因为空集、一个元素、两个元素等每种长度都是答案。

#### 执行步骤

```text
// 1. 进入每层递归时先把当前 path 加入答案。
// 2. 从 startIndex 向后枚举下一元素。
// 3. 选择 nums[i] 并递归 i + 1。
// 4. 回来后弹出 nums[i]，尝试不选它而选后面的元素。
// 5. startIndex 到末尾时自然结束，无需额外终止条件。
```

#### C++ 实现

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

#### Python 实现

```python
def subsets(nums):
    answer = [[]]
    for value in nums: answer += [part + [value] for part in answer]
    return answer
```


### 子集 II


**形象理解**：与子集 I 相同，但输入中有相同卡片。排序后，同一父节点下只允许第一个相同值开分支，避免生成两棵完全相同的子树。

#### 执行步骤

```text
// 1. 先排序 nums，让重复值相邻。
// 2. 每次进入递归都保存 path。
// 3. i > startIndex 且 nums[i] == nums[i-1] 时跳过同层重复。
// 4. 选择 nums[i]，递归 i + 1，随后撤销。
// 5. 不同递归层仍可选择相同值，因此能产生 [2,2]。
```

#### C++ 实现

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

#### Python 实现

```python
def subsets_with_dup(nums):
    nums.sort(); answer = [[]]; previous_size = 0
    for i, value in enumerate(nums):
        start = previous_size if i and nums[i] == nums[i - 1] else 0
        previous_size = len(answer)
        answer += [answer[j] + [value] for j in range(start, previous_size)]
    return answer
```


### 递增子序列


**形象理解**：原数组不能排序，因为顺序本身就是题目的一部分。每一层用一张临时名单记录本层已经选过的数，避免相同值从不同下标长出重复分支。

#### 执行步骤

```text
// 1. path.size() >= 2 时保存当前递增子序列。
// 2. i 从 startIndex 向后枚举，保持原下标顺序。
// 3. nums[i] 小于 path.back() 时跳过，保证非递减。
// 4. nums[i] 已在本层 used 集合中时跳过，防止同层重复。
// 5. 标记并选择 nums[i]，递归 i + 1，返回后弹出。
```

#### C++ 实现

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

#### Python 实现

```python
def find_subsequences(nums):
    answer = set()
    def dfs(index, path):
        if len(path) >= 2: answer.add(tuple(path))
        for i in range(index, len(nums)):
            if not path or nums[i] >= path[-1]: dfs(i + 1, path + [nums[i]])
    dfs(0, [])
    return list(map(list, answer))
```


### 全排列 I


**形象理解**：有 n 个座位，每层决定一个座位坐谁。`used[i]` 表示第 i 个人已经坐下，直到所有座位填满时记录一种排列。

#### 执行步骤

```text
// 1. path.size() == nums.size() 时保存排列。
// 2. 每层都从下标 0 开始枚举所有人。
// 3. used[i] 为 true 说明这个人已坐下，跳过。
// 4. 选择 nums[i] 并令 used[i] = true，递归下一个座位。
// 5. 返回后弹出并恢复 used[i] = false。
```

#### C++ 实现

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

#### Python 实现

```python
def permute(nums):
    if not nums: return [[]]
    return [[nums[i]] + tail for i in range(len(nums)) for tail in permute(nums[:i] + nums[i + 1:])]
```


### 全排列 II


**形象理解**：相同数字像长相一样的人。排序后规定在同一个座位上，只有前一个相同数字已经被使用时，后一个才有资格出场，避免交换双胞胎产生重复排列。

#### 执行步骤

```text
// 1. 先排序 nums，使相同值相邻。
// 2. used[i] 为 true 时跳过已经放入 path 的元素。
// 3. i > 0、nums[i] == nums[i-1] 且 used[i-1] == false 时跳过。
// 4. 选择当前元素、标记 used、递归，再撤销两项状态。
// 5. path 长度等于 n 时保存答案。
```

#### C++ 实现

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

#### Python 实现

```python
def permute_unique(nums):
    answer = []
    def dfs(path, remaining):
        if not remaining: answer.append(path); return
        for value in sorted(set(remaining)):
            copy = remaining[:]; copy.remove(value); dfs(path + [value], copy)
    dfs([], nums)
    return answer
```


### N 皇后


**形象理解**：逐行摆皇后，每行只放一个。新皇后只需向上检查同列和两条斜线，因为下面的行还没有摆任何皇后。

#### 执行步骤

```text
// 1. row == n 时说明 n 行都合法放置，保存棋盘。
// 2. 枚举当前行的每一列 col。
// 3. 检查同列、左上斜线、右上斜线是否已有皇后。
// 4. 合法就把 board[row][col] 改为 'Q'，递归下一行。
// 5. 返回后恢复为 '.'，尝试当前行的下一列。
```

#### C++ 实现

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

#### Python 实现

```python
def solve_n_queens(n):
    answer, columns, diagonals1, diagonals2 = [], set(), set(), set()
    def dfs(row, board):
        if row == n: answer.append(["".join(line) for line in board]); return
        for col in range(n):
            if col in columns or row - col in diagonals1 or row + col in diagonals2: continue
            columns.add(col); diagonals1.add(row - col); diagonals2.add(row + col)
            line = ["."] * n; line[col] = "Q"; dfs(row + 1, board + [line])
            columns.remove(col); diagonals1.remove(row - col); diagonals2.remove(row + col)
    dfs(0, [])
    return answer
```


### 数独


**形象理解**：找到第一个空格，尝试放入 1 到 9；一旦后续无解就擦掉重试。找到一条完整解后立即向上返回 true，不再枚举其他分支。

#### 执行步骤

```text
// 1. 从左到右、从上到下寻找第一个 '.'。
// 2. 对字符 '1' 到 '9' 检查所在行、列和 3x3 宫格。
// 3. 合法数字写入格子，然后递归求解剩余空格。
// 4. 递归成功立即返回 true。
// 5. 失败就把格子恢复为 '.'；所有数字失败则返回 false。
```

#### C++ 实现

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

#### Python 实现

```python
def solve_sudoku(board):
    empty = [(r, c) for r in range(9) for c in range(9) if board[r][c] == "."]
    def dfs(index):
        if index == len(empty): return True
        r, c = empty[index]
        used = set(board[r]) | {board[i][c] for i in range(9)} | {board[i][j] for i in range(r // 3 * 3, r // 3 * 3 + 3) for j in range(c // 3 * 3, c // 3 * 3 + 3)}
        for digit in "123456789":
            if digit not in used:
                board[r][c] = digit
                if dfs(index + 1): return True
                board[r][c] = "."
        return False
    dfs(0)
```


### 重新安排行程


**形象理解**：机票是只能使用一次的边，机场是图节点。行程必须用完所有票，且同一机场有多个目的地时优先尝试字典序更小者。

#### 执行步骤

```text
// 1. 邻接表按字典序保存每个出发地可用的目的地及票数。
// 2. path 从 "JFK" 开始。
// 3. 从当前机场依字典序尝试一张仍有余量的机票。
// 4. 票数减一、目的地加入 path，递归下一机场。
// 5. path.size() == tickets.size() + 1 时成功；失败就恢复票数和路径。
```

#### C++ 实现

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

#### Python 实现

```python
from collections import defaultdict

def find_itinerary(tickets):
    graph = defaultdict(list)
    for source, target in sorted(tickets, reverse=True): graph[source].append(target)
    route = []
    def visit(airport):
        while graph[airport]: visit(graph[airport].pop())
        route.append(airport)
    visit("JFK")
    return route[::-1]
```


## 贪心算法

### 分发饼干


**形象理解**：小饼干留给容易满足的孩子，大饼干优先尝试满足胃口最大的孩子，避免大饼干被不必要地浪费。

#### 执行步骤

```text
// 1. 将孩子胃口和饼干尺寸分别排序。
// 2. 从最大胃口孩子和最大饼干开始比较。
// 3. 饼干够大时匹配成功，两个指针都左移。
// 4. 饼干不够时只移动孩子指针，尝试胃口更小的孩子。
// 5. 匹配次数就是最多满足人数。
```

#### C++ 实现

```cpp
int findContentChildren(vector<int>& g, vector<int>& s) {
    sort(g.begin(), g.end());
    sort(s.begin(), s.end());
    int index = s.size() - 1; // 饼干数组的下标
    int result = 0;
    for (int i = g.size() - 1; i >= 0; i--) { // 遍历胃口
        if (index >= 0 && s[index] >= g[i]) { // 遍历饼干
            result++;
            index--;
        }
    }
    return result;

}
```

#### Python 实现

```python
def find_content_children(children, cookies):
    children.sort(); cookies.sort(); child = 0
    for cookie in cookies:
        if child < len(children) and cookie >= children[child]: child += 1
    return child
```


### 摆动序列


**形象理解**：只保留山峰和山谷，中间同坡度的点都可以删去。当前差值与上一段有效差值异号时，才真正形成一次摆动。

#### 执行步骤

```text
// 1. prevDiff 表示上一个被计入的有效坡度。
// 2. 遍历相邻元素得到 curDiff。
// 3. curDiff > 0 且 prevDiff <= 0，或 curDiff < 0 且 prevDiff >= 0 时出现峰谷。
// 4. 答案加一，并令 prevDiff = curDiff。
// 5. 平坡或同向坡不更新 prevDiff，保留更有利的端点。
```

#### C++ 实现

```cpp
int wiggleMaxLength(vector<int>& nums) {
    if (nums.size() <= 1) return nums.size();
    int curDiff = 0; // 当前一对差值
    int preDiff = 0; // 前一对差值
    int result = 1;  // 记录峰值个数，序列默认序列最右边有一个峰值
    for (int i = 0; i < nums.size() - 1; i++) {
        curDiff = nums[i + 1] - nums[i];
        // 出现峰值
        if ((preDiff <= 0 && curDiff > 0) || (preDiff >= 0 && curDiff < 0)) {
            result++;
            preDiff = curDiff; // 注意这里，只在摆动变化的时候更新prediff
        }
    }
    return result;
}
```

#### Python 实现

```python
def wiggle_max_length(nums):
    up = down = 1
    for a, b in zip(nums, nums[1:]):
        if b > a: up = down + 1
        elif b < a: down = up + 1
    return max(up, down)
```


### 最大子序和


**形象理解**：前面的累计和如果已经是负债，就不值得带到今天；从当前数字重新开一段一定更好。

#### 执行步骤

```text
// 1. current 表示必须以当前位置结尾的最大子数组和。
// 2. current = max(nums[i], current + nums[i])。
// 3. 前一段为负时等价于从 nums[i] 重新开始。
// 4. 每步用 current 更新全局 best。
```

#### C++ 实现

```cpp
int maxSubArray(vector<int>& nums) {
    int result = INT32_MIN;
    int count = 0;
    for (int i = 0; i < nums.size(); i++) {
        count += nums[i];
        if (count > result) { // 取区间累计的最大值（相当于不断确定最大子序终止位置）
            result = count;
        }
        if (count <= 0) count = 0; // 相当于重置最大子序起始位置，因为遇到负数一定是拉低总和
    }
    return result;
}
```

#### Python 实现

```python
def max_sub_array(nums):
    current = answer = nums[0]
    for value in nums[1:]:
        current = max(value, current + value); answer = max(answer, current)
    return answer
```


### 买卖股票的最佳时机 II


**形象理解**：只要明天比今天贵，就把这一小段上涨收入囊中。连续上涨的每个小台阶之和，正好等于最低点买、最高点卖的总利润。

#### 执行步骤

```text
// 1. 从第二天开始计算 prices[i] - prices[i-1]。
// 2. 差值为正时加入 profit。
// 3. 差值为负或零时跳过，不做亏损交易。
// 4. 所有正收益之和就是不限交易次数的最大利润。
```

#### C++ 实现

```cpp
int maxProfit(vector<int>& prices) {
    int result = 0;
    for (int i = 1; i < prices.size(); i++) {
        result += max(prices[i] - prices[i - 1], 0);
    }
    return result;
}
```

#### Python 实现

```python
def max_profit_unlimited(prices):
    return sum(max(0, right - left) for left, right in zip(prices, prices[1:]))
```


### 跳跃游戏 I


**形象理解**：维护目前最远能铺到哪里，像不断延长一块安全地毯。只要当前下标仍在地毯内，就能从这里继续把地毯向前铺。

#### 执行步骤

```text
// 1. cover 表示当前可到达的最远下标。
// 2. 仅遍历 i <= cover 的位置，超出部分尚不可达。
// 3. cover = max(cover, i + nums[i])。
// 4. cover >= n - 1 时立即返回 true。
// 5. 可访问位置用尽仍未覆盖终点时返回 false。
```

#### C++ 实现

```cpp
bool canJump(vector<int>& nums) {
    int cover = 0;
    if (nums.size() == 1) return true; // 只有一个元素，就是能达到
    for (int i = 0; i <= cover; i++) { // 注意这里是小于等于cover
        cover = max(i + nums[i], cover);
        if (cover >= nums.size() - 1) return true; // 说明可以覆盖到终点了
    }
    return false;
}
```

#### Python 实现

```python
def can_jump(nums):
    farthest = 0
    for i, jump in enumerate(nums):
        if i > farthest: return False
        farthest = max(farthest, i + jump)
    return True
```


### 跳跃游戏 II


**形象理解**：一次跳跃能覆盖一段区间。扫描当前区间内的所有起跳点，计算下一跳最远能覆盖哪里；走到本层边界时才把跳数加一，类似按层 BFS。

#### 执行步骤

```text
// 1. currentEnd 表示当前跳数能到达的右边界。
// 2. nextEnd 记录扫描当前层时发现的最远位置。
// 3. 每到一个 i，更新 nextEnd = max(nextEnd, i + nums[i])。
// 4. i == currentEnd 时必须再跳一次，令 currentEnd = nextEnd。
// 5. 到达终点前停止，避免在终点多计一次。
```

#### C++ 实现

```cpp
int jump(vector<int>& nums) {
    if (nums.size() == 1) return 0;
    int curDistance = 0;    // 当前覆盖最远距离下标
    int ans = 0;            // 记录走的最大步数
    int nextDistance = 0;   // 下一步覆盖最远距离下标
    for (int i = 0; i < nums.size(); i++) {
        nextDistance = max(nums[i] + i, nextDistance);  // 更新下一步覆盖最远距离下标
        if (i == curDistance) {                         // 遇到当前覆盖最远距离下标
            ans++;                                  // 需要走下一步
            curDistance = nextDistance;             // 更新当前覆盖最远距离下标（相当于加油了）
            if (nextDistance >= nums.size() - 1) break;  // 当前覆盖最远距到达集合终点，不用做ans++操作了，直接结束
        }
    }
    return ans;
}
```

#### Python 实现

```python
def jump(nums):
    steps = end = farthest = 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == end: steps, end = steps + 1, farthest
    return steps
```


### K 次取反后最大化数组和


**形象理解**：绝对值大的负数翻正收益最大，因此按绝对值从大到小处理。负数都翻完后若还剩奇数次，只能翻绝对值最小的数字，损失最少。

#### 执行步骤

```text
// 1. 按绝对值从大到小排序。
// 2. 从前向后把负数翻正，每次消耗一次 k。
// 3. 若 k 仍为奇数，翻转数组中绝对值最小的最后一个元素。
// 4. 累加最终数组得到最大和。
```

#### C++ 实现

```cpp
static bool cmp(int a, int b) {
    return abs(a) > abs(b);
}
int largestSumAfterKNegations(vector<int>& A, int K) {
    sort(A.begin(), A.end(), cmp);       // 第一步
    for (int i = 0; i < A.size(); i++) { // 第二步
        if (A[i] < 0 && K > 0) {
            A[i] *= -1;
            K--;
        }
    }
    if (K % 2 == 1) A[A.size() - 1] *= -1; // 第三步
    int result = 0;
    for (int a : A) result += a;        // 第四步
    return result;
}
```

#### Python 实现

```python
def largest_sum_after_k_negations(nums, k):
    nums.sort(key=abs, reverse=True)
    for i in range(len(nums)):
        if nums[i] < 0 and k: nums[i], k = -nums[i], k - 1
    if k % 2: nums[-1] = -nums[-1]
    return sum(nums)
```


### 加油站


**形象理解**：从某站出发后若在 j 站前油量变负，那么这段区间里的任何站作为起点都无法越过 j；可以整段跳过，从 j+1 重新开始。

#### 执行步骤

```text
// 1. total 累加所有 gas[i] - cost[i]，判断全程总体是否可行。
// 2. current 累加当前候选起点以来的剩余油量。
// 3. current < 0 时，令 start = i + 1 并把 current 清零。
// 4. 扫描结束后 total < 0 返回 -1，否则返回 start。
```

#### C++ 实现

```cpp
int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
    int curSum = 0;
    int totalSum = 0;
    int start = 0;
    for (int i = 0; i < gas.size(); i++) {
        curSum += gas[i] - cost[i];
        totalSum += gas[i] - cost[i];
        if (curSum < 0) {   // 当前累加rest[i]和 curSum一旦小于0
            start = i + 1;  // 起始位置更新为i+1
            curSum = 0;     // curSum从0开始
        }
    }
    if (totalSum < 0) return -1; // 说明怎么走都不可能跑一圈了
    return start;
}
```

#### Python 实现

```python
def can_complete_circuit(gas, cost):
    total = tank = start = 0
    for i, (supply, expense) in enumerate(zip(gas, cost)):
        delta = supply - expense; total += delta; tank += delta
        if tank < 0: start, tank = i + 1, 0
    return start if total >= 0 else -1
```


### 分发糖果


**形象理解**：先只听左邻居的要求，从左向右保证高分者比左边多；再只听右邻居的要求，从右向左取两种要求的较大值。

#### 执行步骤

```text
// 1. 每个孩子先分 1 颗糖。
// 2. 从左向右，ratings[i] > ratings[i-1] 时令 candy[i] = candy[i-1] + 1。
// 3. 从右向左，ratings[i] > ratings[i+1] 时更新为 max(当前值, candy[i+1]+1)。
// 4. 两遍分别满足左右约束，最后求和。
```

#### C++ 实现

```cpp
int candy(vector<int>& ratings) {
    vector<int> candyVec(ratings.size(), 1);
    // 从前向后
    for (int i = 1; i < ratings.size(); i++) {
        if (ratings[i] > ratings[i - 1]) candyVec[i] = candyVec[i - 1] + 1;
    }
    // 从后向前
    for (int i = ratings.size() - 2; i >= 0; i--) {
        if (ratings[i] > ratings[i + 1] ) {
            candyVec[i] = max(candyVec[i], candyVec[i + 1] + 1);
        }
    }
    // 统计结果
    int result = 0;
    for (int i = 0; i < candyVec.size(); i++) result += candyVec[i];
    return result;
}
```

#### Python 实现

```python
def candy(ratings):
    sweets = [1] * len(ratings)
    for i in range(1, len(ratings)):
        if ratings[i] > ratings[i - 1]: sweets[i] = sweets[i - 1] + 1
    for i in range(len(ratings) - 2, -1, -1):
        if ratings[i] > ratings[i + 1]: sweets[i] = max(sweets[i], sweets[i + 1] + 1)
    return sum(sweets)
```


### 柠檬水找零


**形象理解**：5 元是最灵活的零钱。收到 20 元时优先用一张 10 元加一张 5 元，保留更多 5 元应对只能用 5 元组合的情况。

#### 执行步骤

```text
// 1. 收到 5 元时增加 five。
// 2. 收到 10 元时必须消耗一张 five，并增加 ten。
// 3. 收到 20 元时优先消耗 ten + five。
// 4. 没有 10 元时再消耗三张 five。
// 5. 任一步零钱不足立即返回 false。
```

#### C++ 实现

```cpp
bool lemonadeChange(vector<int>& bills) {
    int five = 0, ten = 0, twenty = 0;
    for (int bill : bills) {
        // 情况一
        if (bill == 5) five++;
        // 情况二
        if (bill == 10) {
            if (five <= 0) return false;
            ten++;
            five--;
        }
        // 情况三
        if (bill == 20) {
            // 优先消耗10美元，因为5美元的找零用处更大，能多留着就多留着
            if (five > 0 && ten > 0) {
                five--;
                ten--;
                twenty++; // 其实这行代码可以删了，因为记录20已经没有意义了，不会用20来找零
            } else if (five >= 3) {
                five -= 3;
                twenty++; // 同理，这行代码也可以删了
            } else return false;
        }
    }
    return true;
}
```

#### Python 实现

```python
def lemonade_change(bills):
    five = ten = 0
    for bill in bills:
        if bill == 5: five += 1
        elif bill == 10: five, ten = five - 1, ten + 1
        elif ten: five, ten = five - 1, ten - 1
        else: five -= 3
        if five < 0: return False
    return True
```


### 根据身高重建队列


**形象理解**：高个子不受矮个子影响，所以先安排高个子。按身高从高到低处理时，把人插入下标 k，前面恰好已有 k 个不矮于他的人。

#### 执行步骤

```text
// 1. 按身高降序排序；身高相同按 k 升序排序。
// 2. 依次取出 person = [height, k]。
// 3. 将 person 插入结果队列的第 k 个位置。
// 4. 后插入的更矮者不会改变已安排高个子的 k 条件。
```

#### C++ 实现

```cpp
// 身高从大到小排（身高相同k小的站前面）
static bool cmp(const vector<int>& a, const vector<int>& b) {
    if (a[0] == b[0]) return a[1] < b[1];
    return a[0] > b[0];
}
vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
    sort (people.begin(), people.end(), cmp);
    list<vector<int>> que; // list底层是链表实现，插入效率比vector高的多
    for (int i = 0; i < people.size(); i++) {
        int position = people[i][1]; // 插入到下标为position的位置
        std::list<vector<int>>::iterator it = que.begin();
        while (position--) { // 寻找在插入位置
            it++;
        }
        que.insert(it, people[i]);
    }
    return vector<vector<int>>(que.begin(), que.end());
}
```

#### Python 实现

```python
def reconstruct_queue(people):
    answer = []
    for person in sorted(people, key=lambda item: (-item[0], item[1])):
        answer.insert(person[1], person)
    return answer
```


### 用最少数量的箭引爆气球


**形象理解**：把每个气球看成横轴上的区间。一支箭要尽量同时穿过更多区间，因此把箭放在当前重叠区域最靠右的位置，为后面的气球留下最大余地。

#### 执行步骤

```text
// 1. 按区间右端点升序排序。
// 2. 第一支箭放在第一个气球的右端点。
// 3. 下一个气球左端点 <= arrowPos 时可被同一箭射中。
// 4. 否则必须新增一支箭，并更新 arrowPos 为该气球右端点。
```

#### C++ 实现

```cpp
int findMinArrowShots(vector<vector<int>>& points) {
        if (points.size() == 0) return 0;
        sort(points.begin(), points.end());

        int result = 1; // points 不为空至少需要一支箭
        for (int i = 1; i < points.size(); i++) {
            if (points[i][0] > points[i - 1][1]) {  // 气球i和气球i-1不挨着，注意这里不是>=
                result++; // 需要一支箭
            }
            else {  // 气球i和气球i-1挨着
                points[i][1] = min(points[i - 1][1], points[i][1]); // 更新重叠气球最小右边界
            }
        }
        return result;
    }
```

#### Python 实现

```python
def find_min_arrow_shots(points):
    arrows, end = 0, float("-inf")
    for start, finish in sorted(points, key=lambda point: point[1]):
        if start > end: arrows, end = arrows + 1, finish
    return arrows
```


### 无重叠区间


**形象理解**：要保留尽可能多的会议，就总选最早结束的那场，它给后续会议留下的时间最多；删除数等于总区间数减保留数。

#### 执行步骤

```text
// 1. 按右端点升序排序区间。
// 2. 保留第一个区间并记录 end。
// 3. 当前左端点 >= end 时可以保留，并更新 end。
// 4. 否则它与已保留区间重叠，计为删除。
```

#### C++ 实现

```cpp
int eraseOverlapIntervals(vector<vector<int>>& intervals) {
    sort(intervals.begin(), intervals.end());
    int res = 1;
    int end = intervals[0][1];
    for(int i = 1; i < intervals.size(); ++i){
        if(intervals[i][0] < end){
            end = min(end, intervals[i][1]);
        }else{
            res++;
            end = intervals[i][1];
        }
    }
    return intervals.size()-res;
}
```

#### Python 实现

```python
def erase_overlap_intervals(intervals):
    kept, end = 0, float("-inf")
    for start, finish in sorted(intervals, key=lambda interval: interval[1]):
        if start >= end: kept, end = kept + 1, finish
    return len(intervals) - kept
```


### 划分字母区间


**形象理解**：每个字符都拉着一根线直到它最后一次出现的位置。扫描一个片段时，遇到的新字符可能把片段右边界继续拉长；走到最远边界才能切断。

#### 执行步骤

```text
// 1. 先记录每个字符最后出现的下标。
// 2. 扫描时令 end = max(end, last[s[i]])。
// 3. 当 i == end 时，当前片段中所有字符都不会在后面出现。
// 4. 记录 end - start + 1，并令 start = i + 1。
```

#### C++ 实现

```cpp
 vector<int> partitionLabels(string S) {
    int hash[27] = {0}; // i为字符，hash[i]为字符出现的最后位置
    for (int i = 0; i < S.size(); i++) { // 统计每一个字符最后出现的位置
        hash[S[i] - 'a'] = i;
    }
    vector<int> result;
    int left = 0;
    int right = 0;
    for (int i = 0; i < S.size(); i++) {
        right = max(right, hash[S[i] - 'a']); // 找到字符出现的最远边界
        if (i == right) {
            result.push_back(right - left + 1);
            left = i + 1;
        }
    }
    return result;
}
```

#### Python 实现

```python
def partition_labels(s):
    last, start, end, answer = {char: i for i, char in enumerate(s)}, 0, 0, []
    for i, char in enumerate(s):
        end = max(end, last[char])
        if i == end: answer.append(end - start + 1); start = i + 1
    return answer
```


### 合并区间


**形象理解**：区间按起点排队后，只需观察新来的区间是否碰到结果中最后一段；碰到就拉长终点，没碰到就另开一段。

#### 执行步骤

```text
// 1. 按左端点升序排序。
// 2. 结果为空或 current.left > result.back().right 时追加新区间。
// 3. 否则两段重叠，更新 result.back().right 为更大的右端点。
// 4. 扫描完成后，结果中的区间互不重叠。
```

#### C++ 实现

```cpp
vector<vector<int>> merge(vector<vector<int>>& intervals) {
    vector<vector<int>> result;
    if (intervals.size() == 0) return result; // 区间集合为空直接返回
    // 排序的参数使用了lambda表达式
    sort(intervals.begin(), intervals.end(), [](const vector<int>& a, const vector<int>& b){return a[0] < b[0];});

    // 第一个区间就可以放进结果集里，后面如果重叠，在result上直接合并
    result.push_back(intervals[0]);

    for (int i = 1; i < intervals.size(); i++) {
        if (result.back()[1] >= intervals[i][0]) { // 发现重叠区间
            // 合并区间，只更新右边界就好，因为result.back()的左边界一定是最小值，因为我们按照左边界排序的
            result.back()[1] = max(result.back()[1], intervals[i][1]);
        } else {
            result.push_back(intervals[i]); // 区间不重叠
        }
    }
    return result;
}
```

#### Python 实现

```python
def merge(intervals):
    answer = []
    for interval in sorted(intervals):
        if not answer or interval[0] > answer[-1][1]: answer.append(interval)
        else: answer[-1][1] = max(answer[-1][1], interval[1])
    return answer
```


### 单调递增的数字


**形象理解**：从右向左找数字下降的位置。左边数字减一后，把右侧全部改成 9，既能恢复单调，又能让结果尽可能大。

#### 执行步骤

```text
// 1. 将 n 转成字符串，从右向左扫描。
// 2. 若 digits[i-1] > digits[i]，令 digits[i-1]--。
// 3. 记录 marker = i，表示从这里向右都应变成 9。
// 4. 继续向左检查，因为减一可能制造新的下降。
// 5. 扫描后把 marker 及其右侧全部置为 '9'。
```

#### C++ 实现

```cpp
int monotoneIncreasingDigits(int N) {
    string strNum = to_string(N);
    // flag用来标记赋值9从哪里开始
    // 设置为这个默认值，为了防止第二个for循环在flag没有被赋值的情况下执行
    int flag = strNum.size();
    for (int i = strNum.size() - 1; i > 0; i--) {
        if (strNum[i - 1] > strNum[i] ) {
            flag = i;
            strNum[i - 1]--;
        }
    }
    for (int i = flag; i < strNum.size(); i++) {
        strNum[i] = '9';
    }
    return stoi(strNum);
}
```

#### Python 实现

```python
def monotone_increasing_digits(n):
    digits = list(str(n)); marker = len(digits)
    for i in range(len(digits) - 1, 0, -1):
        if digits[i - 1] > digits[i]:
            digits[i - 1] = str(int(digits[i - 1]) - 1); marker = i
    digits[marker:] = "9" * (len(digits) - marker)
    return int("".join(digits))
```


### 监控二叉树


**形象理解**：摄像头放在叶子上很浪费；让叶子先保持“未覆盖”，它的父节点就会安装摄像头，同时覆盖父、孩子和祖父。状态从下向上传递最自然。

#### 执行步骤

```text
// 1. 后序遍历，让孩子先报告状态：未覆盖、装摄像头、已覆盖。
// 2. 空节点视为已覆盖，避免在叶子上装摄像头。
// 3. 任一孩子未覆盖时，当前节点安装摄像头并增加计数。
// 4. 任一孩子有摄像头时，当前节点已覆盖。
// 5. 两个孩子都已覆盖时，当前节点暂时未覆盖，交给父节点处理。
// 6. 根最终未覆盖时还需补装一个摄像头。
```

#### C++ 实现

```cpp
int result;
    int traversal(TreeNode* cur) {

        // 空节点，该节点有覆盖
        if (cur == NULL) return 2;

        int left = traversal(cur->left);    // 左
        int right = traversal(cur->right);  // 右

        // 情况1
        // 左右节点都有覆盖
        if (left == 2 && right == 2) return 0;

        // 情况2
        if (left == 0 || right == 0) {
            result++;
            return 1;
        }
        if (left == 1 || right == 1) return 2;

        // 以上代码我没有使用else，主要是为了把各个分支条件展现出来，这样代码有助于读者理解
        // 这个 return -1 逻辑不会走到这里。
        return -1;
    }
public:
    int minCameraCover(TreeNode* root) {
        result = 0;
        // 情况4
        if (traversal(root) == 0) { // root 无覆盖
            result++;
        }
        return result;
    }
```

#### Python 实现

```python
def min_camera_cover(root):
    cameras = 0
    def state(node):
        nonlocal cameras
        if not node: return 1
        left, right = state(node.left), state(node.right)
        if left == 2 or right == 2: cameras += 1; return 0
        if left == 0 or right == 0: return 1
        return 2
    if state(root) == 2: cameras += 1
    return cameras
```


## 动态规划基础

### 爬楼梯


**形象理解**：到第 i 级的最后一步只有两种来源：从 i-1 走一步，或从 i-2 跨两步，因此总方法数就是两条来路之和。

#### 执行步骤

```text
// 1. 定义 dp[i] 为到达第 i 级的方法数。
// 2. 初始化 dp[1] = 1、dp[2] = 2。
// 3. 对 i >= 3，计算 dp[i] = dp[i-1] + dp[i-2]。
// 4. 只依赖前两个状态时可用两个变量滚动，空间降为 O(1)。
```

#### C++ 实现

```cpp
 int climbStairs(int n) {
    if (n <= 1) return n; // 因为下面直接对dp[2]操作了，防止空指针
    vector<int> dp(n + 1);
    dp[1] = 1;
    dp[2] = 2;
    for (int i = 3; i <= n; i++) { // 注意i是从3开始的
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}
```

#### Python 实现

```python
def climb_stairs(n):
    first, second = 1, 1
    for _ in range(n): first, second = second, first + second
    return first
```


### 使用最小花费爬楼梯


**形象理解**：站上第 i 级可以从前一级或前两级跨来，比较两条路线截至出发台阶所付的总费用，选择更便宜的一条。

#### 执行步骤

```text
// 1. dp[i] 表示到达第 i 个位置的最小费用，楼顶位置为 n。
// 2. dp[0] = dp[1] = 0，因为可以从 0 或 1 开始。
// 3. dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])。
// 4. 计算到 dp[n] 即到达楼顶的最小费用。
```

#### C++ 实现

```cpp
int minCostClimbingStairs(vector<int>& cost) {
    vector<int> dp(cost.size() + 1);
    dp[0] = 0; // 默认第一步都是不花费体力的
    dp[1] = 0;
    for (int i = 2; i <= cost.size(); i++) {
        dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2]);
    }
    return dp[cost.size()];
}
```

#### Python 实现

```python
def min_cost_climbing_stairs(cost):
    previous, current = 0, 0
    for value in cost: previous, current = current, min(previous, current) + value
    return min(previous, current)
```


### 不同路径


**形象理解**：机器人进入一个格子只能来自上方或左方，所以当前格子的路线数等于“从上方来的路线数 + 从左方来的路线数”。

#### 执行步骤

```text
// 1. dp[i][j] 表示到达格子 (i,j) 的路径数量。
// 2. 第一行和第一列都只有一条直线路径，初始化为 1。
// 3. 其余格子计算 dp[i][j] = dp[i-1][j] + dp[i][j-1]。
// 4. 右下角 dp[m-1][n-1] 就是答案。
```

#### C++ 实现

```cpp
int uniquePaths(int m, int n) {
    vector<vector<int>> dp(m, vector<int>(n, 0));
    for (int i = 0; i < m; i++) dp[i][0] = 1;
    for (int j = 0; j < n; j++) dp[0][j] = 1;
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
        }
    }
    return dp[m - 1][n - 1];
}
```

#### Python 实现

```python
def unique_paths(m, n):
    dp = [1] * n
    for _ in range(1, m):
        for col in range(1, n): dp[col] += dp[col - 1]
    return dp[-1]
```


### 不同路径 II


**形象理解**：障碍格像封死的路口，到达方法数直接清零；其他格仍把上方和左方的路线汇总过来。

#### 执行步骤

```text
// 1. 起点或终点是障碍时直接返回 0。
// 2. dp[0][0] = 1，表示从起点出发的一种方式。
// 3. 扫描每格，障碍位置令 dp[i][j] = 0。
// 4. 非障碍位置累加上方与左方存在的路径数。
// 5. 返回终点状态。
```

#### C++ 实现

```cpp
int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
    int m = obstacleGrid.size();
    int n = obstacleGrid[0].size();
    if (obstacleGrid[m - 1][n - 1] == 1 || obstacleGrid[0][0] == 1) //如果在起点或终点出现了障碍，直接返回0
        return 0;
    vector<vector<int>> dp(m, vector<int>(n, 0));
    for (int i = 0; i < m && obstacleGrid[i][0] == 0; i++) dp[i][0] = 1;
    for (int j = 0; j < n && obstacleGrid[0][j] == 0; j++) dp[0][j] = 1;
    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            if (obstacleGrid[i][j] == 1) continue;
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
        }
    }
    return dp[m - 1][n - 1];
}
```

#### Python 实现

```python
def unique_paths_with_obstacles(grid):
    dp = [0] * len(grid[0]); dp[0] = 1
    for row in grid:
        for col, blocked in enumerate(row):
            dp[col] = 0 if blocked else dp[col] + (dp[col - 1] if col else 0)
    return dp[-1]
```


### 整数拆分


**形象理解**：第一次把 i 切出一段 j，剩下的 i-j 可以选择不再拆，也可以继续按最优方式拆；两者取更大的乘积。

#### 执行步骤

```text
// 1. dp[i] 表示整数 i 拆分后可得到的最大乘积。
// 2. 初始化 dp[2] = 1。
// 3. 枚举第一段 j，比较 j*(i-j) 与 j*dp[i-j]。
// 4. 用所有 j 的最大值更新 dp[i]。
// 5. 只需枚举到 i/2 也能覆盖对称切分。
```

#### C++ 实现

```cpp
int integerBreak(int n) {
    vector<int> dp(n + 1);
    dp[2] = 1;
    for (int i = 3; i <= n ; i++) {
        for (int j = 1; j <= i / 2; j++) {
            dp[i] = max(dp[i], max((i - j) * j, dp[i - j] * j));
        }
    }
    return dp[n];
}
```

#### Python 实现

```python
def integer_break(n):
    dp = [0] * (n + 1); dp[1] = 1
    for total in range(2, n + 1):
        dp[total] = max(max(part * (total - part), part * dp[total - part]) for part in range(1, total))
    return dp[n]
```


### 不同的二叉搜索树


**形象理解**：从 1 到 n 中选 i 当根后，左边 i-1 个数可组成若干左树，右边 n-i 个数可组成若干右树；左右方案可以任意配对，所以要相乘。

#### 执行步骤

```text
// 1. dp[n] 表示由 n 个有序节点构成的 BST 数量。
// 2. dp[0] = dp[1] = 1，空树也是一种组合方式。
// 3. 枚举根节点 i = 1..n。
// 4. 累加 dp[i-1] * dp[n-i]，分别代表左右子树方案数。
// 5. 自小到大计算直到 dp[n]。
```

#### C++ 实现

```cpp
int numTrees(int n) {
    vector<int> dp(n + 1);
    dp[0] = 1;
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= i; j++) {
            dp[i] += dp[j - 1] * dp[i - j];
        }
    }
    return dp[n];
}
```

#### Python 实现

```python
def num_trees(n):
    dp = [1] + [0] * n
    for nodes in range(1, n + 1):
        dp[nodes] = sum(dp[root - 1] * dp[nodes - root] for root in range(1, nodes + 1))
    return dp[n]
```


## 打家劫舍系列

### 打家劫舍 I


**形象理解**：走到第 i 家时只有两种合法决定：不偷它，继承前一家最优值；偷它，就必须跳过前一家并加上前两家的最优值。

#### 执行步骤

```text
// 1. dp[i] 表示考虑到第 i 家能偷到的最大金额。
// 2. dp[0] = nums[0]，dp[1] = max(nums[0], nums[1])。
// 3. dp[i] = max(dp[i-1], dp[i-2] + nums[i])。
// 4. 返回最后状态；也可用两个变量滚动保存。
```

#### C++ 实现

```cpp
int rob(vector<int>& nums) {
    if (nums.size() == 0) return 0;
    if (nums.size() == 1) return nums[0];
    vector<int> dp(nums.size());
    dp[0] = nums[0];
    dp[1] = max(nums[0], nums[1]);
    for (int i = 2; i < nums.size(); i++) {
        dp[i] = max(dp[i - 2] + nums[i], dp[i - 1]);
    }
    return dp[nums.size() - 1];
}
```

#### Python 实现

```python
def rob(nums):
    skip = take = 0
    for value in nums: skip, take = max(skip, take), skip + value
    return max(skip, take)
```


### 打家劫舍 II


**形象理解**：首尾相邻意味着不能同时偷。把环拆成两个互斥场景：不考虑最后一家，或不考虑第一家，各做一次线性打家劫舍并取较大值。

#### 执行步骤

```text
// 1. 只有一家时直接返回其金额。
// 2. 对区间 [0, n-2] 运行线性打家劫舍。
// 3. 对区间 [1, n-1] 再运行一次。
// 4. 两种场景覆盖所有合法方案，返回较大值。
```

#### C++ 实现

```cpp
int rob(vector<int>& nums) {
    if (nums.size() == 0) return 0;
    if (nums.size() == 1) return nums[0];
    int result1 = robRange(nums, 0, nums.size() - 2); // 情况二
    int result2 = robRange(nums, 1, nums.size() - 1); // 情况三
    return max(result1, result2);
}
// 198.打家劫舍的逻辑
int robRange(vector<int>& nums, int start, int end) {
    if (end == start) return nums[start];
    vector<int> dp(nums.size());
    dp[start] = nums[start];
    dp[start + 1] = max(nums[start], nums[start + 1]);
    for (int i = start + 2; i <= end; i++) {
        dp[i] = max(dp[i - 2] + nums[i], dp[i - 1]);
    }
    return dp[end];
}
```

#### Python 实现

```python
def rob_circle(nums):
    if len(nums) == 1: return nums[0]
    return max(rob(nums[:-1]), rob(nums[1:]))
```


### 打家劫舍 III


**形象理解**：每个树节点向父亲汇报两个数字：偷我时最多拿多少、不偷我时最多拿多少。父亲根据自己的选择组合孩子的两类状态。

#### 执行步骤

```text
// 1. 后序递归返回 {notRob, rob}。
// 2. rob = node->val + left.notRob + right.notRob。
// 3. notRob = max(left.notRob, left.rob) + max(right.notRob, right.rob)。
// 4. 根节点返回后取两种状态的较大值。
```

#### C++ 实现

```cpp
int rob(TreeNode* root) {
    vector<int> result = robTree(root);
    return max(result[0], result[1]);
}
// 长度为2的数组，0：不偷，1：偷
vector<int> robTree(TreeNode* cur) {
    if (cur == NULL) return vector<int>{0, 0};
    vector<int> left = robTree(cur->left);
    vector<int> right = robTree(cur->right);
    // 偷cur，那么就不能偷左右节点。
    int val1 = cur->val + left[0] + right[0];
    // 不偷cur，那么可以偷也可以不偷左右节点，则取较大的情况
    int val2 = max(left[0], left[1]) + max(right[0], right[1]);
    return {val2, val1};
}
```

#### Python 实现

```python
def rob_tree(root):
    def dfs(node):
        if not node: return 0, 0
        left, right = dfs(node.left), dfs(node.right)
        return node.val + left[1] + right[1], max(left) + max(right)
    return max(dfs(root))
```


## 股票动态规划

### 只能买卖一次


**形象理解**：每天结束只记录两种账户状态：手里有股票或没有股票。只允许一次交易时，“持有”的买入来源必须是初始现金，而不能接在上一笔卖出之后。

#### 执行步骤

```text
// 1. hold 表示当天结束持有股票的最大收益，初始化为 -prices[0]。
// 2. cash 表示当天结束不持股的最大收益，初始化为 0。
// 3. hold = max(旧 hold, -prices[i])，决定继续持有或今天首次买入。
// 4. cash = max(旧 cash, 旧 hold + prices[i])，决定不动或今天卖出。
// 5. 最终返回 cash。
```

#### C++ 实现

```cpp
int maxProfit(vector<int>& prices) {
    int low = INT_MAX;
    int result = 0;
    for (int i = 0; i < prices.size(); i++) {
        low = min(low, prices[i]);  // 取最左最小价格
        result = max(result, prices[i] - low); // 直接取最大区间利润
    }
    return result;
}
```

#### Python 实现

```python
def max_profit_once(prices):
    minimum, answer = float("inf"), 0
    for price in prices: minimum, answer = min(minimum, price), max(answer, price - minimum)
    return answer
```


### 可以买卖任意次


**形象理解**：仍是持股和空仓两个账户，但今天买入可以使用之前已经赚到的现金，因此允许一笔交易结束后再开始下一笔。

#### 执行步骤

```text
// 1. 保存更新前的 oldHold 和 oldCash，避免同一天状态串用。
// 2. hold = max(oldHold, oldCash - prices[i])。
// 3. cash = max(oldCash, oldHold + prices[i])。
// 4. 扫描所有天后，空仓状态 cash 是最大已实现利润。
```

#### C++ 实现

```cpp
int maxProfit(vector<int>& prices) {
    int len = prices.size();
    vector<vector<int>> dp(len, vector<int>(2, 0));
    dp[0][0] -= prices[0];
    dp[0][1] = 0;
    for (int i = 1; i < len; i++) {
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] - prices[i]); // 注意这里是和121. 买卖股票的最佳时机唯一不同的地方。
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i]);
    }
    return dp[len - 1][1];
    }
```

#### Python 实现

```python
def max_profit_many(prices):
    cash, hold = 0, float("-inf")
    for price in prices: cash, hold = max(cash, hold + price), max(hold, cash - price)
    return cash
```


### 最多买卖两次


**形象理解**：一天结束可能处于五个阶段：未操作、第一次持有、第一次卖出、第二次持有、第二次卖出。每个阶段只从自己或前一阶段转移。

#### 执行步骤

```text
// 1. buy1 = max(buy1, -price)。
// 2. sell1 = max(sell1, buy1 + price)。
// 3. buy2 = max(buy2, sell1 - price)。
// 4. sell2 = max(sell2, buy2 + price)。
// 5. 更新时注意使用上一天状态，最终返回 sell2。
```

#### C++ 实现

```cpp
int maxProfit(vector<int>& prices) {
    if (prices.size() == 0) return 0;
    vector<vector<int>> dp(prices.size(), vector<int>(5, 0));
    dp[0][1] = -prices[0];
    dp[0][3] = -prices[0];
    for (int i = 1; i < prices.size(); i++) {
        dp[i][0] = dp[i - 1][0];
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
        dp[i][2] = max(dp[i - 1][2], dp[i - 1][1] + prices[i]);
        dp[i][3] = max(dp[i - 1][3], dp[i - 1][2] - prices[i]);
        dp[i][4] = max(dp[i - 1][4], dp[i - 1][3] + prices[i]);
    }
    return dp[prices.size() - 1][4];
}
```

#### Python 实现

```python
def max_profit_twice(prices):
    buy1 = buy2 = float("-inf"); sell1 = sell2 = 0
    for price in prices:
        buy1 = max(buy1, -price); sell1 = max(sell1, buy1 + price)
        buy2 = max(buy2, sell1 - price); sell2 = max(sell2, buy2 + price)
    return sell2
```


### 最多买卖 K 次


**形象理解**：把两次交易的五个阶段扩展成 `2k+1` 个阶段，奇数编号表示第几次持有，偶数编号表示第几次卖出。

#### 执行步骤

```text
// 1. dp[j] 表示当天处于第 j 个交易阶段的最大收益。
// 2. 所有买入阶段初始化为 -prices[0]，卖出阶段初始化为 0。
// 3. 奇数 j：dp[j] = max(dp[j], dp[j-1] - price)。
// 4. 偶数 j：dp[j] = max(dp[j], dp[j-1] + price)。
// 5. 每天按阶段更新，返回最后一次卖出阶段。
```

#### C++ 实现

```cpp
int maxProfit(int k, vector<int>& prices) {
    if (prices.size() == 0) return 0;
    vector<vector<int>> dp(prices.size(), vector<int>(2 * k + 1, 0));
    for (int j = 1; j < 2 * k; j += 2) {
        dp[0][j] = -prices[0];
    }
    for (int i = 1;i < prices.size(); i++) {
        for (int j = 0; j < 2 * k - 1; j += 2) {
            dp[i][j + 1] = max(dp[i - 1][j + 1], dp[i - 1][j] - prices[i]);
            dp[i][j + 2] = max(dp[i - 1][j + 2], dp[i - 1][j + 1] + prices[i]);
        }
    }
    return dp[prices.size() - 1][2 * k];
}
```

#### Python 实现

```python
def max_profit_k(k, prices):
    if k >= len(prices) // 2: return max_profit_many(prices)
    buy, sell = [float("-inf")] * (k + 1), [0] * (k + 1)
    for price in prices:
        for transaction in range(1, k + 1):
            buy[transaction] = max(buy[transaction], sell[transaction - 1] - price)
            sell[transaction] = max(sell[transaction], buy[transaction] + price)
    return sell[k]
```


### 含冷冻期的股票交易


**形象理解**：卖出后的第二天不能买，所以不能只区分持股和空仓；要单独记住“今天刚卖出”和“处于冷冻/普通空仓”的状态。

#### 执行步骤

```text
// 1. hold：持股；sold：今天刚卖出；rest：不持股且可继续等待。
// 2. newHold = max(oldHold, oldRest - price)，只能从可买状态买入。
// 3. newSold = oldHold + price，卖出必须来自昨天持股。
// 4. newRest = max(oldRest, oldSold)，冷冻一天后进入可等待状态。
// 5. 最终返回 sold 与 rest 的较大值。
```

#### C++ 实现

```cpp
int maxProfit(vector<int>& prices) {
    int n = prices.size();
    if (n == 0) return 0;
    vector<vector<int>> dp(n, vector<int>(4, 0));
    dp[0][0] -= prices[0]; // 持股票
    for (int i = 1; i < n; i++) {
        dp[i][0] = max(dp[i - 1][0], max(dp[i - 1][3] - prices[i], dp[i - 1][1] - prices[i]));
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][3]);
        dp[i][2] = dp[i - 1][0] + prices[i];
        dp[i][3] = dp[i - 1][2];
    }
    return max(dp[n - 1][3], max(dp[n - 1][1], dp[n - 1][2]));
}
```

#### Python 实现

```python
def max_profit_cooldown(prices):
    hold, sold, rest = float("-inf"), 0, 0
    for price in prices: hold, sold, rest = max(hold, rest - price), hold + price, max(rest, sold)
    return max(sold, rest)
```


### 含手续费的股票交易


**形象理解**：仍使用持股/空仓两状态，只需在买入或卖出的一侧扣一次手续费，不能两边都扣。

#### 执行步骤

```text
// 1. hold 初始化为 -prices[0]，cash 初始化为 0。
// 2. newHold = max(oldHold, oldCash - price)。
// 3. newCash = max(oldCash, oldHold + price - fee)。
// 4. 每笔完整交易只在卖出时扣一次 fee。
// 5. 返回最终 cash。
```

#### C++ 实现

```cpp
 int maxProfit(vector<int>& prices, int fee) {
    int n = prices.size();
    vector<vector<int>> dp(n, vector<int>(2, 0));
    dp[0][0] -= prices[0]; // 持股票
    for (int i = 1; i < n; i++) {
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] - prices[i]);
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i] - fee);
    }
    return max(dp[n - 1][0], dp[n - 1][1]);
}
```

#### Python 实现

```python
def max_profit_fee(prices, fee):
    cash, hold = 0, -prices[0]
    for price in prices[1:]: cash, hold = max(cash, hold + price - fee), max(hold, cash - price)
    return cash
```


## 0-1 背包和完全背包

### 二维 0-1 背包


**形象理解**：每件物品像一次性卡牌。走到第 i 件时，可以不拿它，或在背包容量足够时拿它并接上“只考虑前 i-1 件”的最优值。

#### 执行步骤

```text
// 1. dp[i][j] 表示只考虑 0..i 的物品、容量 j 时的最大价值。
// 2. 不选物品 i：继承 dp[i-1][j]。
// 3. 选物品 i：dp[i-1][j-weight[i]] + value[i]。
// 4. 容量足够时取两者最大值，否则只能不选。
// 5. 初始化第一行后逐物品、逐容量计算。
```

#### C++ 实现

```cpp
int main() {
    int n, bagweight;// bagweight代表行李箱空间
    cin >> n >> bagweight;
    vector<int> weight(n, 0); // 存储每件物品所占空间
    vector<int> value(n, 0);  // 存储每件物品价值
    for(int i = 0; i < n; ++i) {
        cin >> weight[i];
    }
    for(int j = 0; j < n; ++j) {
        cin >> value[j];
    }
    // dp数组, dp[i][j]代表行李箱空间为j的情况下,从下标为[0, i]的物品里面任意取,能达到的最大价值
    vector<vector<int>> dp(weight.size(), vector<int>(bagweight + 1, 0));

    // 初始化, 因为需要用到dp[i - 1]的值
    // j < weight[0]已在上方被初始化为0
    // j >= weight[0]的值就初始化为value[0]
    for (int j = weight[0]; j <= bagweight; j++) {
        dp[0][j] = value[0];
    }

    for(int i = 1; i < weight.size(); i++) { // 遍历科研物品
        for(int j = 0; j <= bagweight; j++) { // 遍历行李箱容量
            if (j < weight[i]) dp[i][j] = dp[i - 1][j]; // 如果装不下这个物品,那么就继承dp[i - 1][j]的值
            else {
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight[i]] + value[i]);
            }
        }
    }
    cout << dp[n - 1][bagweight] << endl;
    return 0;
}
```

#### Python 实现

```python
def knapsack_2d(weights, values, capacity):
    dp = [[0] * (capacity + 1) for _ in range(len(weights) + 1)]
    for i, (weight, value) in enumerate(zip(weights, values), 1):
        for cap in range(capacity + 1):
            dp[i][cap] = dp[i - 1][cap]
            if cap >= weight: dp[i][cap] = max(dp[i][cap], dp[i - 1][cap - weight] + value)
    return dp[-1][-1]
```


### 一维 0-1 背包


**形象理解**：把二维表压成一行后，容量必须从大到小更新。这样读取的 `dp[j-weight]` 仍属于上一轮，保证当前物品不会在同一轮被拿多次。

#### 执行步骤

```text
// 1. dp[j] 表示容量 j 的最大价值。
// 2. 外层逐个枚举物品。
// 3. 内层 j 从 capacity 递减到 weight[i]。
// 4. dp[j] = max(dp[j], dp[j-weight[i]] + value[i])。
// 5. 倒序是“一件物品只用一次”的关键。
```

#### C++ 实现

```cpp
    for(int i = 0; i < weight.size(); i++) { // 遍历物品
        for(int j = bagWeight; j >= weight[i]; j--) { // 遍历背包容量
            dp[j] = max(dp[j], dp[j - weight[i]] + value[i]);

        }
    }
```

#### Python 实现

```python
def knapsack(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for weight, value in zip(weights, values):
        for cap in range(capacity, weight - 1, -1): dp[cap] = max(dp[cap], dp[cap - weight] + value)
    return dp[-1]
```


### 分割等和子集


**形象理解**：总和若为奇数一定无法平分；否则问题就是从数组中挑一些数，恰好装满容量为 `sum/2` 的 0-1 背包。

#### 执行步骤

```text
// 1. 求总和，奇数直接返回 false。
// 2. target = sum / 2。
// 3. 对每个 num，容量从 target 向 num 倒序更新。
// 4. dp[j] = max(dp[j], dp[j-num] + num)。
// 5. 最终 dp[target] == target 时可平分。
```

#### C++ 实现

```cpp
bool canPartition(vector<int>& nums) {
    int sum = 0;

    // dp[i]中的i表示背包内总和
    // 题目中说：每个数组中的元素不会超过 100，数组的大小不会超过 200
    // 总和不会大于20000，背包最大只需要其中一半，所以10001大小就可以了
    vector<int> dp(10001, 0);
    for (int i = 0; i < nums.size(); i++) {
        sum += nums[i];
    }
    // 也可以使用库函数一步求和
    // int sum = accumulate(nums.begin(), nums.end(), 0);
    if (sum % 2 == 1) return false;
    int target = sum / 2;

    // 开始 01背包
    for(int i = 0; i < nums.size(); i++) {
        for(int j = target; j >= nums[i]; j--) { // 每一个元素一定是不可重复放入，所以从大到小遍历
            dp[j] = max(dp[j], dp[j - nums[i]] + nums[i]);
        }
    }
    // 集合中的元素正好可以凑成总和target
    if (dp[target] == target) return true;
    return false;
}
```

#### Python 实现

```python
def can_partition(nums):
    total = sum(nums)
    if total % 2: return False
    reachable = 1
    for value in nums: reachable |= reachable << value
    return bool(reachable >> (total // 2) & 1)
```


### 最后一块石头的重量 II


**形象理解**：把石头分成两堆互相碰撞，最终重量是两堆总重之差。让较轻那堆尽量接近总重一半，差值就最小。

#### 执行步骤

```text
// 1. target = total / 2，建立 0-1 背包。
// 2. 每块石头既是重量也是价值。
// 3. 容量倒序更新 dp[j] = max(dp[j], dp[j-stone] + stone)。
// 4. dp[target] 是不超过一半的最大一堆重量。
// 5. 答案为 total - 2 * dp[target]。
```

#### C++ 实现

```cpp
int lastStoneWeightII(vector<int>& stones) {
    vector<int> dp(15001, 0);
    int sum = 0;
    for (int i = 0; i < stones.size(); i++)
        sum += stones[i];
    int target = sum / 2;
    for (int i = 0; i < stones.size(); i++) {
        // 遍历物品
        for (int j = target; j >= stones[i]; j--) { // 遍历背包
            dp[j] = max(dp[j], dp[j - stones[i]] + stones[i]);
        }
    }
    return sum - dp[target] - dp[target];
}
```

#### Python 实现

```python
def last_stone_weight_ii(stones):
    possible = {0}
    for stone in stones: possible |= {total + stone for total in possible}
    half = max(total for total in possible if total <= sum(stones) // 2)
    return sum(stones) - 2 * half
```


### 目标和


**形象理解**：设加正号的数之和为 P、加负号的数之和为 N，则 `P-N=target` 且 `P+N=sum`，所以只需统计和为 `(sum+target)/2` 的子集数量。

#### 执行步骤

```text
// 1. 若 abs(target) > sum 或 sum + target 为奇数，返回 0。
// 2. bag = (sum + target) / 2。
// 3. dp[0] = 1，表示什么都不选恰好组成 0 的一种方法。
// 4. 对每个 num，容量从 bag 向 num 倒序。
// 5. dp[j] += dp[j-num]，累计选择 num 后新增的方案数。
```

#### C++ 实现

```cpp
int findTargetSumWays(vector<int>& nums, int target) {
    int sum = 0;
    for (int i = 0; i < nums.size(); i++) sum += nums[i];
    if (abs(target) > sum) return 0; // 此时没有方案
    if ((target + sum) % 2 == 1) return 0; // 此时没有方案
    int bagSize = (target + sum) / 2;
    vector<int> dp(bagSize + 1, 0);
    dp[0] = 1;
    for (int i = 0; i < nums.size(); i++) {
        for (int j = bagSize; j >= nums[i]; j--) {
            dp[j] += dp[j - nums[i]];
        }
    }
    return dp[bagSize];
}
```

#### Python 实现

```python
from collections import Counter

def find_target_sum_ways(nums, target):
    ways = Counter({0: 1})
    for value in nums:
        next_ways = Counter()
        for total, count in ways.items():
            next_ways[total + value] += count
            next_ways[total - value] += count
        ways = next_ways
    return ways[target]
```


### 一和零


**形象理解**：背包有两个容量维度：最多能用 m 个 0 和 n 个 1。每个字符串是一件同时消耗两种资源、价值为 1 的物品。

#### 执行步骤

```text
// 1. 统计当前字符串的 zeroCount 和 oneCount。
// 2. dp[i][j] 表示最多使用 i 个 0、j 个 1 能选择的字符串数。
// 3. i 从 m 向 zeroCount 倒序，j 从 n 向 oneCount 倒序。
// 4. dp[i][j] = max(dp[i][j], dp[i-zero][j-one] + 1)。
// 5. 双维度都倒序，保证字符串只使用一次。
```

#### C++ 实现

```cpp
int findMaxForm(vector<string>& strs, int m, int n) {
    vector<vector<int>> dp(m + 1, vector<int> (n + 1, 0)); // 默认初始化0
    for (string str : strs) { // 遍历物品
        int oneNum = 0, zeroNum = 0;
        for (char c : str) {
            if (c == '0') zeroNum++;
            else oneNum++;
        }
        for (int i = m; i >= zeroNum; i--) { // 遍历背包容量且从后向前遍历！
            for (int j = n; j >= oneNum; j--) {
                dp[i][j] = max(dp[i][j], dp[i - zeroNum][j - oneNum] + 1);
            }
        }
    }
    return dp[m][n];
}
```

#### Python 实现

```python
def find_max_form(strings, m, n):
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for string in strings:
        zeros, ones = string.count("0"), string.count("1")
        for i in range(m, zeros - 1, -1):
            for j in range(n, ones - 1, -1): dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)
    return dp[m][n]
```


### 完全背包的遍历顺序


**形象理解**：完全背包中的物品可重复拿，所以容量从小到大更新；这样本轮刚更新的 `dp[j-weight]` 可以继续使用当前物品。

#### 执行步骤

```text
// 1. 外层物品、内层容量正序：统计组合，且允许当前物品重复使用。
// 2. 外层容量、内层物品：不同选择顺序会被分别统计，得到排列数。
// 3. 求最大价值时二者通常都可行；求方案数时顺序决定含义。
// 4. 0-1 背包容量倒序，完全背包容量正序，不能混淆。
```

#### C++ 实现

```cpp
int completeKnapsack(const vector<int>& weight, const vector<int>& value, int capacity) {
    vector<int> dp(capacity + 1, 0);
    for (int i = 0; i < weight.size(); ++i) {
        for (int cap = weight[i]; cap <= capacity; ++cap) {
            dp[cap] = max(dp[cap], dp[cap - weight[i]] + value[i]);
        }
    }
    return dp[capacity];
}
```

#### Python 实现

```python
def complete_knapsack(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for weight, value in zip(weights, values):
        for cap in range(weight, capacity + 1): dp[cap] = max(dp[cap], dp[cap - weight] + value)
    return dp[-1]
```


### 零钱兑换 II


**形象理解**：要统计不考虑顺序的硬币组合，所以先固定硬币种类，再逐步扩充金额；同一组合不会因为拿币顺序不同被重复计算。

#### 执行步骤

```text
// 1. dp[0] = 1，组成金额 0 有一种空方案。
// 2. 外层遍历每种 coin。
// 3. 内层 amount 从 coin 正序到目标金额。
// 4. dp[amount] += dp[amount-coin]。
// 5. 返回 dp[target]，相同硬币可在本轮重复使用。
```

#### C++ 实现

```cpp
int change(int amount, vector<int>& coins) {
    vector<int> dp(amount + 1, 0);
    dp[0] = 1;
    for (int i = 0; i < coins.size(); i++) { // 遍历物品
        for (int j = coins[i]; j <= amount; j++) { // 遍历背包
            dp[j] += dp[j - coins[i]];
        }
    }
    return dp[amount];
}
```

#### Python 实现

```python
def change(amount, coins):
    dp = [1] + [0] * amount
    for coin in coins:
        for total in range(coin, amount + 1): dp[total] += dp[total - coin]
    return dp[amount]
```


### 组合总和 IV


**形象理解**：`[1,2]` 和 `[2,1]` 算两种答案，因此先枚举目标总和，再枚举最后放入哪个数字，让不同顺序进入不同转移路径。

#### 执行步骤

```text
// 1. dp[0] = 1。
// 2. 外层 sum 从 1 正序到 target。
// 3. 内层枚举每个 num。
// 4. sum >= num 时，dp[sum] += dp[sum-num]。
// 5. 外容量、内物品使答案按排列计数。
```

#### C++ 实现

```cpp
int combinationSum4(vector<int>& nums, int target) {
    vector<int> dp(target + 1, 0);
    dp[0] = 1;
    for (int i = 0; i <= target; i++) { // 遍历背包
        for (int j = 0; j < nums.size(); j++) { // 遍历物品
            if (i - nums[j] >= 0 && dp[i] < INT_MAX - dp[i - nums[j]]) {
                dp[i] += dp[i - nums[j]];
            }
        }
    }
    return dp[target];
}
```

#### Python 实现

```python
def combination_sum4(nums, target):
    dp = [1] + [0] * target
    for total in range(1, target + 1): dp[total] = sum(dp[total - value] for value in nums if value <= total)
    return dp[target]
```


### 爬楼梯作为完全背包


**形象理解**：若一次可走 1..m 级，每一种走法就是用这些“步长硬币”按顺序凑出楼层 n；顺序不同的步长序列算不同路线。

#### 执行步骤

```text
// 1. dp[0] = 1，站在原地是一种起始方式。
// 2. 外层枚举当前要到达的楼层 i。
// 3. 内层枚举最后一步 step = 1..m。
// 4. i >= step 时累加 dp[i-step]。
// 5. 容量在外、步长在内，因此统计排列。
```

#### C++ 实现

```cpp
int main() {
    int n, m;
    while (cin >> n >> m) {
        vector<int> dp(n + 1, 0);
        dp[0] = 1;
        for (int i = 1; i <= n; i++) { // 遍历背包
            for (int j = 1; j <= m; j++) { // 遍历物品
                if (i - j >= 0) dp[i] += dp[i - j];
            }
        }
        cout << dp[n] << endl;
    }
}
```

#### Python 实现

```python
def climb_stairs_complete(n, steps=(1, 2)):
    dp = [1] + [0] * n
    for total in range(1, n + 1): dp[total] = sum(dp[total - step] for step in steps if step <= total)
    return dp[n]
```


### 零钱兑换


**形象理解**：每个金额都问：“如果最后使用某枚硬币，之前最少需要几枚？”从所有合法硬币给出的候选值中选最小。

#### 执行步骤

```text
// 1. dp[0] = 0，其余初始化为不可达的大值 target + 1。
// 2. 对每枚 coin，金额从 coin 向 target 正序更新。
// 3. dp[j-coin] 可达时，dp[j] = min(dp[j], dp[j-coin] + 1)。
// 4. 最终仍是大值说明无法组成，返回 -1。
```

#### C++ 实现

```cpp
int coinChange(vector<int>& coins, int amount) {
    vector<int> dp(amount + 1, INT_MAX);
    dp[0] = 0;
    for (int i = 0; i < coins.size(); i++) { // 遍历物品
        for (int j = coins[i]; j <= amount; j++) { // 遍历背包
            if (dp[j - coins[i]] != INT_MAX) { // 如果dp[j - coins[i]]是初始值则跳过
                dp[j] = min(dp[j - coins[i]] + 1, dp[j]);
            }
        }
    }
    if (dp[amount] == INT_MAX) return -1;
    return dp[amount];
}
```

#### Python 实现

```python
def coin_change(coins, amount):
    dp = [0] + [amount + 1] * amount
    for total in range(1, amount + 1): dp[total] = min((dp[total - coin] + 1 for coin in coins if coin <= total), default=amount + 1)
    return -1 if dp[amount] > amount else dp[amount]
```


### 完全平方数


**形象理解**：把 `1,4,9...` 这些平方数当成可无限使用的硬币，目标是用最少硬币凑成 n。

#### 执行步骤

```text
// 1. dp[0] = 0，其余初始化为大值。
// 2. 枚举平方数 square = i*i，且 square <= n。
// 3. 容量 j 从 square 正序到 n。
// 4. dp[j] = min(dp[j], dp[j-square] + 1)。
// 5. 返回 dp[n]。
```

#### C++ 实现

```cpp
int numSquares(int n) {
    vector<int> dp(n + 1, INT_MAX);
    dp[0] = 0;
    for (int i = 0; i <= n; i++) { // 遍历背包
        for (int j = 1; j * j <= i; j++) { // 遍历物品
            dp[i] = min(dp[i - j * j] + 1, dp[i]);
        }
    }
    return dp[n];
}
```

#### Python 实现

```python
def num_squares(n):
    dp = [0] + [n] * n
    for total in range(1, n + 1): dp[total] = 1 + min(dp[total - root * root] for root in range(1, int(total ** 0.5) + 1))
    return dp[n]
```


### 单词拆分


**形象理解**：`dp[i]` 表示字符串前 i 个字符已经能被词典切好。枚举最后一个切口 j，只要前半段可达且 `s[j..i)` 在词典中，i 就可达。

#### 执行步骤

```text
// 1. dp[0] = true，空前缀天然可拆分。
// 2. 枚举结尾 i = 1..n。
// 3. 枚举最后切口 j = 0..i-1。
// 4. 若 dp[j] 为真且 s.substr(j, i-j) 在字典中，令 dp[i] = true 并停止。
// 5. 返回 dp[n]。
```

#### C++ 实现

```cpp
bool wordBreak(string s, vector<string>& wordDict) {
    unordered_set<string> wordSet(wordDict.begin(), wordDict.end());
    vector<bool> dp(s.size() + 1, false);
    dp[0] = true;
    for (int i = 1; i <= s.size(); i++) {   // 遍历背包
        for (int j = 0; j < i; j++) {       // 遍历物品
            string word = s.substr(j, i - j); //substr(起始位置，截取的个数)
            if (wordSet.find(word) != wordSet.end() && dp[j]) {
                dp[i] = true;
            }
        }
    }
    return dp[s.size()];
}
```

#### Python 实现

```python
def word_break(s, word_dict):
    words, dp = set(word_dict), [True] + [False] * len(s)
    for end in range(1, len(s) + 1): dp[end] = any(dp[start] and s[start:end] in words for start in range(end))
    return dp[-1]
```


## 子序列与字符串动态规划

### 最长递增子序列


**形象理解**：`dp[i]` 只负责“必须以 nums[i] 收尾”的队伍。向前寻找所有比它小的末尾，把 nums[i] 接在其中最长的一队后面。

#### 执行步骤

```text
// 1. 每个元素单独都能组成长度 1，初始化 dp[i] = 1。
// 2. 对每个 i，枚举所有 j < i。
// 3. nums[j] < nums[i] 时，可把 i 接到 j 后面。
// 4. dp[i] = max(dp[i], dp[j] + 1)。
// 5. 答案是所有 dp[i] 的最大值，不一定在最后位置结束。
```

#### C++ 实现

```cpp
int lengthOfLIS(vector<int>& nums) {
    if (nums.size() <= 1) return nums.size();
    vector<int> dp(nums.size(), 1);
    int result = 0;
    for (int i = 1; i < nums.size(); i++) {
        for (int j = 0; j < i; j++) {
            if (nums[i] > nums[j]) dp[i] = max(dp[i], dp[j] + 1);
        }
        if (dp[i] > result) result = dp[i]; // 取长的子序列
    }
    return result;
}
```

#### Python 实现

```python
from bisect import bisect_left

def length_of_lis(nums):
    tails = []
    for value in nums:
        i = bisect_left(tails, value)
        if i == len(tails): tails.append(value)
        else: tails[i] = value
    return len(tails)
```


### 最长连续递增序列


**形象理解**：连续意味着不能从更早位置跳过来。当前数字比前一个大时，当前增长跑道延长一格；否则从当前位置重新起跑。

#### 执行步骤

```text
// 1. current 和 best 初始化为 1。
// 2. nums[i] > nums[i-1] 时令 current++。
// 3. 否则令 current = 1，从当前元素重新开始。
// 4. 每一步用 current 更新 best。
```

#### C++ 实现

```cpp
int findLengthOfLCIS(vector<int>& nums) {
    if (nums.size() == 0) return 0;
    int result = 1;
    vector<int> dp(nums.size() ,1);
    for (int i = 1; i < nums.size(); i++) {
        if (nums[i] > nums[i - 1]) { // 连续记录
            dp[i] = dp[i - 1] + 1;
        }
        if (dp[i] > result) result = dp[i];
    }
    return result;
}
```

#### Python 实现

```python
def find_length_of_lcis(nums):
    current = answer = int(bool(nums))
    for left, right in zip(nums, nums[1:]):
        current = current + 1 if right > left else 1; answer = max(answer, current)
    return answer
```


### 最长重复子数组


**形象理解**：把两个数组排成棋盘，元素相等时当前格能沿左上角的连续对角线延长一格；不相等则连续段立即归零。

#### 执行步骤

```text
// 1. dp[i][j] 表示以 nums1[i-1]、nums2[j-1] 结尾的相同连续段长度。
// 2. 两元素相等时 dp[i][j] = dp[i-1][j-1] + 1。
// 3. 不相等时 dp[i][j] = 0，因为子数组必须连续。
// 4. 计算过程中维护所有状态的最大值。
```

#### C++ 实现

```cpp
int findLength(vector<int>& nums1, vector<int>& nums2) {
    vector<vector<int>> dp (nums1.size() + 1, vector<int>(nums2.size() + 1, 0));
    int result = 0;
    for (int i = 1; i <= nums1.size(); i++) {
        for (int j = 1; j <= nums2.size(); j++) {
            if (nums1[i - 1] == nums2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            }
            if (dp[i][j] > result) result = dp[i][j];
        }
    }
    return result;
}
```

#### Python 实现

```python
def find_length(nums1, nums2):
    dp, answer = [0] * (len(nums2) + 1), 0
    for a in nums1:
        for j in range(len(nums2) - 1, -1, -1):
            dp[j + 1] = dp[j] + 1 if a == nums2[j] else 0; answer = max(answer, dp[j + 1])
    return answer
```


### 最长公共子序列


**形象理解**：两个字符串的末尾字符相同，就把它接到两个前缀的公共子序列后；末尾不同，就尝试放弃任意一边的末尾字符。

#### 执行步骤

```text
// 1. dp[i][j] 表示 text1 前 i 个字符与 text2 前 j 个字符的 LCS 长度。
// 2. 字符相同：dp[i][j] = dp[i-1][j-1] + 1。
// 3. 字符不同：dp[i][j] = max(dp[i-1][j], dp[i][j-1])。
// 4. 第一行和第一列为空串情况，初始化为 0。
// 5. 返回 dp[m][n]。
```

#### C++ 实现

```cpp
int longestCommonSubsequence(string text1, string text2) {
    vector<vector<int>> dp(text1.size() + 1, vector<int>(text2.size() + 1, 0));
    for (int i = 1; i <= text1.size(); i++) {
        for (int j = 1; j <= text2.size(); j++) {
            if (text1[i - 1] == text2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
            }
        }
    }
    return dp[text1.size()][text2.size()];
}
```

#### Python 实现

```python
def longest_common_subsequence(a, b):
    dp = [0] * (len(b) + 1)
    for char_a in a:
        diagonal = 0
        for j, char_b in enumerate(b, 1):
            old = dp[j]; dp[j] = diagonal + 1 if char_a == char_b else max(dp[j], dp[j - 1]); diagonal = old
    return dp[-1]
```


### 不相交的线


**形象理解**：相同数字之间连线且不能相交，等价于保持两数组原顺序选择相同元素，也就是最长公共子序列。

#### 执行步骤

```text
// 1. dp[i][j] 表示两个数组前缀最多能连多少条不相交线。
// 2. nums1[i-1] == nums2[j-1] 时连接它们，接在 dp[i-1][j-1] 后。
// 3. 不相等时放弃一边当前元素，取上方与左方状态较大值。
// 4. 返回右下角状态。
```

#### C++ 实现

```cpp
int maxUncrossedLines(vector<int>& nums1, vector<int>& nums2) {
    vector<vector<int>> dp(nums1.size() + 1, vector<int>(nums2.size() + 1));
    for (int i = 1; i <= nums1.size(); ++i) {
        for (int j = 1; j <= nums2.size(); ++j) {
            if (nums1[i - 1] == nums2[j - 1]) dp[i][j] = dp[i - 1][j - 1] + 1;
            else dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
        }
    }
    return dp.back().back();
}
```

#### Python 实现

```python
def max_uncrossed_lines(nums1, nums2):
    return longest_common_subsequence(nums1, nums2)
```


### 动态规划版本的最大子序和


**形象理解**：与贪心解释一致，只是显式定义 `dp[i]` 为必须以 i 结尾的最大和，决定接上前一段还是从当前数重新开始。

#### 执行步骤

```text
// 1. dp[0] = nums[0]。
// 2. dp[i] = max(nums[i], dp[i-1] + nums[i])。
// 3. dp[i-1] 为负时，丢弃旧段更有利。
// 4. 返回 dp 数组中的最大值。
```

#### C++ 实现

```cpp
int maxSubArray(vector<int>& nums) {
    if (nums.size() == 0) return 0;
    vector<int> dp(nums.size());
    dp[0] = nums[0];
    int result = dp[0];
    for (int i = 1; i < nums.size(); i++) {
        dp[i] = max(dp[i - 1] + nums[i], nums[i]); // 状态转移公式
        if (dp[i] > result) result = dp[i]; // result 保存dp[i]的最大值
    }
    return result;
}
```

#### Python 实现

```python
def max_sub_array_dp(nums):
    dp = nums[:]
    for i in range(1, len(nums)): dp[i] = max(nums[i], dp[i - 1] + nums[i])
    return max(dp)
```


### 判断子序列


**形象理解**：二维 DP 版本记录 s 的前 i 个字符能否全部嵌入 t 的前 j 个字符；若当前字符不匹配，只能丢弃 t 的当前字符继续寻找。

#### 执行步骤

```text
// 1. dp[i][j] 表示 s[0..i) 是否为 t[0..j) 的子序列。
// 2. 空字符串是任何字符串的子序列，初始化 dp[0][j] = true。
// 3. 字符相等时继承 dp[i-1][j-1]。
// 4. 字符不等时继承 dp[i][j-1]，表示跳过 t[j-1]。
// 5. 返回 dp[m][n]；双指针还可把空间降到 O(1)。
```

#### C++ 实现

```cpp
 bool isSubsequence(string s, string t) {
    vector<vector<int>> dp(s.size() + 1, vector<int>(t.size() + 1, 0));
    for (int i = 1; i <= s.size(); i++) {
        for (int j = 1; j <= t.size(); j++) {
            if (s[i - 1] == t[j - 1]) dp[i][j] = dp[i - 1][j - 1] + 1;
            else dp[i][j] = dp[i][j - 1];
        }
    }
    if (dp[s.size()][t.size()] == s.size()) return true;
    return false;
}
```

#### Python 实现

```python
def is_subsequence(s, t):
    iterator = iter(t)
    return all(char in iterator for char in s)
```


### 不同的子序列


**形象理解**：计算 s 中能选出多少个 t。若当前字符相同，可以用它匹配 t 的末尾，也可以不用它；不同则只能不用它。

#### 执行步骤

```text
// 1. dp[i][j] 表示 s 前 i 个字符组成 t 前 j 个字符的方案数。
// 2. dp[i][0] = 1，因为任何前缀删除全部字符都能得到空串。
// 3. s[i-1] == t[j-1] 时：dp[i][j] = dp[i-1][j-1] + dp[i-1][j]。
// 4. 不相等时：dp[i][j] = dp[i-1][j]。
// 5. 使用足够宽的整数类型保存方案数。
```

#### C++ 实现

```cpp
int numDistinct(string s, string t) {
    vector<vector<uint64_t>> dp(s.size() + 1, vector<uint64_t>(t.size() + 1));
    for (int i = 0; i < s.size(); i++) dp[i][0] = 1;
    // 注意是从1开始
    for (int j = 1; j < t.size(); j++) dp[0][j] = 0;
    for (int i = 1; i <= s.size(); i++) {
        for (int j = 1; j <= t.size(); j++) {
            if (s[i - 1] == t[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
            } else {
                dp[i][j] = dp[i - 1][j];
            }
        }
    }
    return dp[s.size()][t.size()];
}
```

#### Python 实现

```python
def num_distinct(s, t):
    dp = [1] + [0] * len(t)
    for source in s:
        for j in range(len(t) - 1, -1, -1):
            if source == t[j]: dp[j + 1] += dp[j]
    return dp[-1]
```


### 两个字符串的删除操作


**形象理解**：先找两字符串都愿意保留的最长公共子序列，其余字符全部删除；两边删除数就是总长度减去两倍的保留长度。

#### 执行步骤

```text
// 1. 用 LCS 动态规划求 longestCommon。
// 2. word1 需要删除 word1.size() - longestCommon 个字符。
// 3. word2 同理删除 word2.size() - longestCommon 个字符。
// 4. 返回 m + n - 2 * longestCommon。
```

#### C++ 实现

```cpp
int minDistance(string word1, string word2) {
    vector<vector<int>> dp(word1.size() + 1, vector<int>(word2.size() + 1));
    for (int i = 0; i <= word1.size(); i++) dp[i][0] = i;
    for (int j = 0; j <= word2.size(); j++) dp[0][j] = j;
    for (int i = 1; i <= word1.size(); i++) {
        for (int j = 1; j <= word2.size(); j++) {
            if (word1[i - 1] == word2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            } else {
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1);
            }
        }
    }
    return dp[word1.size()][word2.size()];
}
```

#### Python 实现

```python
def min_distance_delete(word1, word2):
    common = longest_common_subsequence(word1, word2)
    return len(word1) + len(word2) - 2 * common
```


### 编辑距离


**形象理解**：让两个前缀对齐。末尾相同时不花钱；不同时考虑删除、插入、替换三种动作，选择从哪个相邻状态走一步最便宜。

#### 执行步骤

```text
// 1. dp[i][j] 表示 word1 前 i 个字符变成 word2 前 j 个字符的最少操作数。
// 2. dp[i][0] = i、dp[0][j] = j。
// 3. 末尾字符相同：dp[i][j] = dp[i-1][j-1]。
// 4. 不同：取删除 dp[i-1][j]、插入 dp[i][j-1]、替换 dp[i-1][j-1] 的最小值再加 1。
// 5. 返回 dp[m][n]。
```

#### C++ 实现

```cpp
 int minDistance(string word1, string word2) {
    vector<vector<int>> dp(word1.size() + 1, vector<int>(word2.size() + 1, 0));
    for (int i = 0; i <= word1.size(); i++) dp[i][0] = i;
    for (int j = 0; j <= word2.size(); j++) dp[0][j] = j;
    for (int i = 1; i <= word1.size(); i++) {
        for (int j = 1; j <= word2.size(); j++) {
            if (word1[i - 1] == word2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
            }
            else {
                dp[i][j] = min({dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]}) + 1;
            }
        }
    }
    return dp[word1.size()][word2.size()];
}
```

#### Python 实现

```python
def min_distance(word1, word2):
    dp = list(range(len(word2) + 1))
    for i, first in enumerate(word1, 1):
        next_row = [i]
        for j, second in enumerate(word2, 1):
            next_row.append(dp[j - 1] if first == second else 1 + min(dp[j - 1], dp[j], next_row[-1]))
        dp = next_row
    return dp[-1]
```


### 回文子串


**形象理解**：区间 `[i,j]` 是回文，需要两端字符相同，并且内部 `[i+1,j-1]` 已是回文；长度 1 或 2 时内部为空，可直接成立。

#### 执行步骤

```text
// 1. dp[i][j] 表示 s[i..j] 是否为回文。
// 2. i 从右向左遍历，保证 dp[i+1][j-1] 已计算。
// 3. s[i] == s[j] 且 (j-i <= 1 或 dp[i+1][j-1]) 时为真。
// 4. 每得到一个 true 就把回文子串数量加一。
```

#### C++ 实现

```cpp
int countSubstrings(string s) {
    vector<vector<bool>> dp(s.size(), vector<bool>(s.size(), false));
    int result = 0;
    for (int i = s.size() - 1; i >= 0; i--) {  // 注意遍历顺序
        for (int j = i; j < s.size(); j++) {
            if (s[i] == s[j]) {
                if (j - i <= 1) { // 情况一 和 情况二
                    result++;
                    dp[i][j] = true;
                } else if (dp[i + 1][j - 1]) { // 情况三
                    result++;
                    dp[i][j] = true;
                }
            }
        }
    }
    return result;
}
```

#### Python 实现

```python
def count_substrings(s):
    answer = 0
    for center in range(2 * len(s) - 1):
        left, right = center // 2, (center + 1) // 2
        while left >= 0 and right < len(s) and s[left] == s[right]: answer += 1; left -= 1; right += 1
    return answer
```


### 最长回文子序列


**形象理解**：区间两端相等时，可以把它们一起包在内部最长回文两侧；不相等时，只能舍弃左端或右端并选择更长结果。

#### 执行步骤

```text
// 1. dp[i][j] 表示 s[i..j] 内最长回文子序列长度。
// 2. 单字符初始化 dp[i][i] = 1。
// 3. s[i] == s[j] 时 dp[i][j] = dp[i+1][j-1] + 2。
// 4. 否则 dp[i][j] = max(dp[i+1][j], dp[i][j-1])。
// 5. i 倒序、j 正序计算，返回 dp[0][n-1]。
```

#### C++ 实现

```cpp
int longestPalindromeSubseq(string s) {
    vector<vector<int>> dp(s.size(), vector<int>(s.size(), 0));
    for (int i = 0; i < s.size(); i++) dp[i][i] = 1;
    for (int i = s.size() - 1; i >= 0; i--) {
        for (int j = i + 1; j < s.size(); j++) {
            if (s[i] == s[j]) {
                dp[i][j] = dp[i + 1][j - 1] + 2;
            } else {
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
            }
        }
    }
    return dp[0][s.size() - 1];
}
```

#### Python 实现

```python
def longest_palindrome_subseq(s):
    return longest_common_subsequence(s, s[::-1])
```


### 正则表达式匹配


**形象理解**：普通字符或 `.` 只消费一次；`*` 像一个可伸缩印章，可以让前一个模式出现零次，也可以在字符匹配时再消费一次文本、但继续保留这个印章。

#### 执行步骤

```text
// 1. dp[i][j] 表示 s 前 i 个字符是否匹配 p 前 j 个字符。
// 2. dp[0][0] = true；形如 a*b* 的模式可通过“出现零次”匹配空串。
// 3. 普通字符或 '.' 匹配时，继承 dp[i-1][j-1]。
// 4. p[j-1] == '*' 时，dp[i][j] 可取 dp[i][j-2]，表示前一字符出现零次。
// 5. 若前一模式字符匹配 s[i-1]，还可取 dp[i-1][j]，表示多匹配一次。
// 6. 返回 dp[m][n]。
```

#### C++ 实现

```cpp
bool isMatch(string s, string p) {
        int m = s.size();
        int n = p.size();

        auto matches = [&](int i, int j){
            if(i == 0)
                return false;
            if(p[j-1] == '.')
                return true;
            return s[i-1] == p[j-1];
        };
        vector<vector<int>> dp(m+1, vector<int>(n+1, false));
        dp[0][0] = true;
        for(int i = 0; i <= m; ++i){
            for(int j = 1; j <= n; ++j){
                if(p[j-1] == '*'){
                    dp[i][j] |= dp[i][j-2];
                    if(matches(i,j-1)) //注意这一步，很巧妙！！
                        dp[i][j] |= dp[i-1][j];
                } else{
                    if(matches(i, j))
                        dp[i][j] |= dp[i-1][j-1];
                }
            }
        }
        return dp[m][n];
    }
```

#### Python 实现

```python
def is_match(s, p):
    dp = [[False] * (len(p) + 1) for _ in range(len(s) + 1)]; dp[0][0] = True
    for j in range(2, len(p) + 1):
        if p[j - 1] == "*": dp[0][j] = dp[0][j - 2]
    for i in range(1, len(s) + 1):
        for j in range(1, len(p) + 1):
            if p[j - 1] in {s[i - 1], "."}: dp[i][j] = dp[i - 1][j - 1]
            elif p[j - 1] == "*": dp[i][j] = dp[i][j - 2] or p[j - 2] in {s[i - 1], "."} and dp[i - 1][j]
    return dp[-1][-1]
```


## 并查集

### 寻找图中是否存在路径


**形象理解**：并查集给每个连通块选一个代表人。每条边让两边代表人合并；最后只需看起点和终点是否拥有同一个代表人。

#### 执行步骤

```text
// 1. parent[i] = i，初始每个节点自成集合。
// 2. find(x) 沿父指针找到根，并用路径压缩让沿途节点直连根。
// 3. 对每条边执行 unite(u, v)，把两个根合并。
// 4. 比较 find(source) 与 find(destination) 是否相等。
```

#### C++ 实现

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

#### Python 实现

```python
def valid_path(n, edges, source, destination):
    parent = list(range(n))
    def find(x):
        while x != parent[x]: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in edges: parent[find(a)] = find(b)
    return find(source) == find(destination)
```


### 冗余连接


**形象理解**：一棵树中新增一条连接同一连通块内部两点的边，就会闭合成环。扫描边时若两端代表人已相同，这条边就是多余边。

#### 执行步骤

```text
// 1. 初始化并查集。
// 2. 按输入顺序扫描边 (u,v)。
// 3. find(u) == find(v) 时，加入这条边会形成环，立即返回。
// 4. 否则 unite(u,v)，让两个连通块合并。
```

#### C++ 实现

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

#### Python 实现

```python
def find_redundant_connection(edges):
    parent = list(range(len(edges) + 1))
    def find(x):
        if parent[x] != x: parent[x] = find(parent[x])
        return parent[x]
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra == rb: return [a, b]
        parent[ra] = rb
```


### 冗余连接 II


**形象理解**：有向树失效只有两类原因：某节点有两个父亲，或图中出现环。先找“双父节点”的两条候选边，再通过跳过候选边做并查集验证。

#### 执行步骤

```text
// 1. 扫描每条有向边，记录每个节点第一次出现的父边。
// 2. 若某节点收到第二条父边，保存前后两条候选边。
// 3. 优先跳过后出现的候选边，检查其余边能否构成合法树。
// 4. 若可以，后候选边就是答案；否则前候选边才是答案。
// 5. 若不存在双父节点，直接用并查集返回形成环的边。
```

#### C++ 实现

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

#### Python 实现

```python
def find_redundant_directed_connection(edges):
    incoming = {}
    first = second = None
    for edge in edges:
        source, target = edge
        if target in incoming: first, second = incoming[target], edge
        else: incoming[target] = edge
    parent = list(range(len(edges) + 1))
    def find(x):
        if parent[x] != x: parent[x] = find(parent[x])
        return parent[x]
    for edge in edges:
        if edge is second: continue
        source, target = edge; rs, rt = find(source), find(target)
        if rs == rt: return first if first else edge
        parent[rt] = rs
    return second
```


## 图的遍历与岛屿问题

### 所有可能的路径


**形象理解**：从 0 号节点出发沿每条有向边试走，路径像一条面包屑轨迹；到达终点就复制轨迹，返回时擦掉最后一步去试另一条边。

#### 执行步骤

```text
// 1. path 初始化为 {0}。
// 2. 当前节点等于 n-1 时保存 path 并返回。
// 3. 枚举 graph[current] 中的每个 next。
// 4. 将 next 加入 path，递归 next。
// 5. 返回后 pop_back，恢复路径。
```

#### C++ 实现

```cpp
vector<vector<int>> result; // 收集符合条件的路径
vector<int> path; // 1节点到终点的路径

void dfs (const vector<vector<int>>& graph, int x, int n) {
    // 当前遍历的节点x 到达节点n
    if (x == n) { // 找到符合条件的一条路径
        result.push_back(path);
        return;
    }
    for (int i = 1; i <= n; i++) { // 遍历节点x链接的所有节点
        if (graph[x][i] == 1) { // 找到 x链接的节点
            path.push_back(i); // 遍历到的节点加入到路径中来
            dfs(graph, i, n); // 进入下一层递归
            path.pop_back(); // 回溯，撤销本节点
        }
    }
}

int main() {
    int n, m, s, t;
    cin >> n >> m;

    // 节点编号从1到n，所以申请 n+1 这么大的数组
    vector<vector<int>> graph(n + 1, vector<int>(n + 1, 0));

    while (m--) {
        cin >> s >> t;
        // 使用邻接矩阵 表示无线图，1 表示 s 与 t 是相连的
        graph[s][t] = 1;
    }

    path.push_back(1); // 无论什么路径已经是从0节点出发
    dfs(graph, 1, n); // 开始遍历

    // 输出结果
    if (result.size() == 0) cout << -1 << endl;
    for (const vector<int> &pa : result) {
        for (int i = 0; i < pa.size() - 1; i++) {
            cout << pa[i] << " ";
        }
        cout << pa[pa.size() - 1]  << endl;
    }
}
```

#### Python 实现

```python
def all_paths_source_target(graph):
    answer = []
    def dfs(node, path):
        if node == len(graph) - 1: answer.append(path); return
        for neighbor in graph[node]: dfs(neighbor, path + [neighbor])
    dfs(0, [0])
    return answer
```


### 岛屿数量


**形象理解**：每发现一块尚未标记的陆地，就发现了一座新岛；从它出发把上下左右连着的陆地全部“涂掉”，同一座岛之后就不会重复计数。

#### 执行步骤

```text
// 1. 双重循环扫描网格。
// 2. 遇到 '1' 时岛屿数加一，并启动 DFS/BFS。
// 3. 搜索把当前陆地标记为已访问。
// 4. 对四个方向中仍为 '1' 的邻居继续搜索。
// 5. 整张网格扫描完后返回计数。
```

#### C++ 实现

```cpp
int dir[4][2] = {0, 1, 1, 0, -1, 0, 0, -1}; // 四个方向
void dfs(const vector<vector<int>>& grid, vector<vector<bool>>& visited, int x, int y) {
    for (int i = 0; i < 4; i++) {
        int nextx = x + dir[i][0];
        int nexty = y + dir[i][1];
        if (nextx < 0 || nextx >= grid.size() || nexty < 0 || nexty >= grid[0].size()) continue;  // 越界了，直接跳过
        if (!visited[nextx][nexty] && grid[nextx][nexty] == 1) { // 没有访问过的 同时 是陆地的

            visited[nextx][nexty] = true;
            dfs(grid, visited, nextx, nexty);
        }
    }
}

int main() {
    int n, m;
    cin >> n >> m;
    vector<vector<int>> grid(n, vector<int>(m, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cin >> grid[i][j];
        }
    }

    vector<vector<bool>> visited(n, vector<bool>(m, false));

    int result = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (!visited[i][j] && grid[i][j] == 1) {
                visited[i][j] = true;
                result++; // 遇到没访问过的陆地，+1
                dfs(grid, visited, i, j); // 将与其链接的陆地都标记上 true
            }
        }
    }

    cout << result << endl;
}
```

#### Python 实现

```python
def num_islands(grid):
    def flood(row, col):
        if row < 0 or row == len(grid) or col < 0 or col == len(grid[0]) or grid[row][col] != "1": return
        grid[row][col] = "0"
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)): flood(row + dr, col + dc)
    answer = 0
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == "1": answer += 1; flood(row, col)
    return answer
```


### 岛屿的最大面积


**形象理解**：每次从一块新陆地开始做“洪水填充”，搜索返回这座岛包含的格子数，再用它更新最大面积。

#### 执行步骤

```text
// 1. DFS 进入合法未访问陆地时先贡献面积 1。
// 2. 标记当前格，防止沿相邻格走回来形成无限递归。
// 3. 累加上下左右四个方向返回的面积。
// 4. 每次发现新岛调用 DFS，并更新全局最大值。
```

#### C++ 实现

```cpp
int count;
int dir[4][2] = {0, 1, 1, 0, -1, 0, 0, -1}; // 四个方向
void dfs(vector<vector<int>>& grid, vector<vector<bool>>& visited, int x, int y) {
    for (int i = 0; i < 4; i++) {
        int nextx = x + dir[i][0];
        int nexty = y + dir[i][1];
        if (nextx < 0 || nextx >= grid.size() || nexty < 0 || nexty >= grid[0].size()) continue;  // 越界了，直接跳过
        if (!visited[nextx][nexty] && grid[nextx][nexty] == 1) { // 没有访问过的 同时 是陆地的
            visited[nextx][nexty] = true;
            count++;
            dfs(grid, visited, nextx, nexty);
        }
    }
}
int main() {
    int n, m;
    cin >> n >> m;
    vector<vector<int>> grid(n, vector<int>(m, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cin >> grid[i][j];
        }
    }
    vector<vector<bool>> visited(n, vector<bool>(m, false));
    int result = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (!visited[i][j] && grid[i][j] == 1) {
                count = 1;  // 因为dfs处理下一个节点，所以这里遇到陆地了就先计数，dfs处理接下来的相邻陆地
                visited[i][j] = true;
                dfs(grid, visited, i, j); // 将与其链接的陆地都标记上 true
                result = max(result, count);
            }
        }
    }
    cout << result << endl;
}
```

#### Python 实现

```python
def max_area_of_island(grid):
    def flood(row, col):
        if row < 0 or row == len(grid) or col < 0 or col == len(grid[0]) or not grid[row][col]: return 0
        grid[row][col] = 0
        return 1 + sum(flood(row + dr, col + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    return max((flood(r, c) for r in range(len(grid)) for c in range(len(grid[0]))), default=0)
```


### 孤岛的最大面积


**形象理解**：从边界能走出去的陆地都不是封闭孤岛。先从四条边界出发淹掉所有相连陆地，剩余陆地才是被水完全包围的部分；原记录最终统计这些剩余陆地的总面积。

#### 执行步骤

```text
// 1. 遍历首尾行和首尾列。
// 2. 从边界陆地启动 DFS/BFS，将相连陆地标记为非孤岛。
// 3. 再扫描内部剩余陆地，通过 DFS 累加其总面积。
// 4. 已从边界访问的格子不再计入。
```

#### C++ 实现

```cpp
int dir[4][2] = {-1, 0, 0, -1, 1, 0, 0, 1}; // 保存四个方向
int count; // 统计符合题目要求的陆地空格数量
void dfs(vector<vector<int>>& grid, int x, int y) {
    grid[x][y] = 0;
    count++;
    for (int i = 0; i < 4; i++) { // 向四个方向遍历
        int nextx = x + dir[i][0];
        int nexty = y + dir[i][1];
        // 超过边界
        if (nextx < 0 || nextx >= grid.size() || nexty < 0 || nexty >= grid[0].size()) continue;
        // 不符合条件，不继续遍历
        if (grid[nextx][nexty] == 0) continue;

        dfs (grid, nextx, nexty);
    }
    return;
}

int main() {
    int n, m;
    cin >> n >> m;
    vector<vector<int>> grid(n, vector<int>(m, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cin >> grid[i][j];
        }
    }

    // 从左侧边，和右侧边 向中间遍历
    for (int i = 0; i < n; i++) {
        if (grid[i][0] == 1) dfs(grid, i, 0);
        if (grid[i][m - 1] == 1) dfs(grid, i, m - 1);
    }
    // 从上边和下边 向中间遍历
    for (int j = 0; j < m; j++) {
        if (grid[0][j] == 1) dfs(grid, 0, j);
        if (grid[n - 1][j] == 1) dfs(grid, n - 1, j);
    }
    count = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (grid[i][j] == 1) dfs(grid, i, j);
        }
    }
    cout << count << endl;
}
```

#### Python 实现

```python
def largest_enclosed_island(grid):
    for row in range(len(grid)):
        for col in (0, len(grid[0]) - 1):
            if grid[row][col]: max_area_of_island_from(grid, row, col)
    for col in range(len(grid[0])):
        for row in (0, len(grid) - 1):
            if grid[row][col]: max_area_of_island_from(grid, row, col)
    return max_area_of_island(grid)

def max_area_of_island_from(grid, row, col):
    if row < 0 or row == len(grid) or col < 0 or col == len(grid[0]) or not grid[row][col]: return 0
    grid[row][col] = 0
    return 1 + sum(max_area_of_island_from(grid, row + dr, col + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
```


### 沉没孤岛（被围绕的区域）


**形象理解**：边界上的陆地 `1` 及其连通区域不会被包围，先给它们贴上临时安全标记 `2`；剩下的 `1` 全部沉成水域 `0`，最后再恢复安全区。

#### 执行步骤

```text
// 1. 从四条边界的陆地 1 出发搜索，临时改成安全标记 2。
// 2. 扫描整个网格，把仍为 1 的格子改成水域 0。
// 3. 这些未被边界搜索触达的 1 就是应被沉没的孤岛。
// 4. 再把所有安全标记 2 恢复为陆地 1。
```

#### C++ 实现

```cpp
int dir[4][2] = {-1, 0, 0, -1, 1, 0, 0, 1}; // 保存四个方向
void dfs(vector<vector<int>>& grid, int x, int y) {
    grid[x][y] = 2;
    for (int i = 0; i < 4; i++) { // 向四个方向遍历
        int nextx = x + dir[i][0];
        int nexty = y + dir[i][1];
        // 超过边界
        if (nextx < 0 || nextx >= grid.size() || nexty < 0 || nexty >= grid[0].size()) continue;
        // 不符合条件，不继续遍历
        if (grid[nextx][nexty] == 0 || grid[nextx][nexty] == 2) continue;
        dfs (grid, nextx, nexty);
    }
    return;
}

int main() {
    int n, m;
    cin >> n >> m;
    vector<vector<int>> grid(n, vector<int>(m, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cin >> grid[i][j];
        }
    }

    // 步骤一：
    // 从左侧边，和右侧边 向中间遍历
    for (int i = 0; i < n; i++) {
        if (grid[i][0] == 1) dfs(grid, i, 0);
        if (grid[i][m - 1] == 1) dfs(grid, i, m - 1);
    }

    // 从上边和下边 向中间遍历
    for (int j = 0; j < m; j++) {
        if (grid[0][j] == 1) dfs(grid, 0, j);
        if (grid[n - 1][j] == 1) dfs(grid, n - 1, j);
    }
    // 步骤二、步骤三
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (grid[i][j] == 1) grid[i][j] = 0;
            if (grid[i][j] == 2) grid[i][j] = 1;
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cout << grid[i][j] << " ";
        }
        cout << endl;
    }
}
```

#### Python 实现

```python
def solve_surrounded_regions(board):
    def mark(row, col):
        if row < 0 or row == len(board) or col < 0 or col == len(board[0]) or board[row][col] != "O": return
        board[row][col] = "S"
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)): mark(row + dr, col + dc)
    for row in range(len(board)): mark(row, 0); mark(row, len(board[0]) - 1)
    for col in range(len(board[0])): mark(0, col); mark(len(board) - 1, col)
    for row in range(len(board)):
        for col in range(len(board[0])): board[row][col] = "O" if board[row][col] == "S" else "X"
```


### 太平洋大西洋水流问题


**形象理解**：正向从每个格子试着往低处流很重复。反过来分别从两片海岸向高处爬，能被两次逆向搜索都到达的格子，就能顺流到两片海。

#### 执行步骤

```text
// 1. 建立 pacificVisited 和 atlanticVisited 两张标记表。
// 2. 从上边界、左边界做逆向 DFS/BFS，邻格高度必须不低于当前格。
// 3. 从下边界、右边界做同样搜索。
// 4. 扫描所有格子，同时被两张表标记的坐标加入答案。
```

#### C++ 实现

```cpp
int n, m;
int dir[4][2] = {-1, 0, 0, -1, 1, 0, 0, 1};
void dfs(vector<vector<int>>& grid, vector<vector<bool>>& visited, int x, int y) {
    if (visited[x][y]) return;

    visited[x][y] = true;

    for (int i = 0; i < 4; i++) {
        int nextx = x + dir[i][0];
        int nexty = y + dir[i][1];
        if (nextx < 0 || nextx >= n || nexty < 0 || nexty >= m) continue;
        if (grid[x][y] > grid[nextx][nexty]) continue; // 注意：这里是从低向高遍历

        dfs (grid, visited, nextx, nexty);
    }
    return;
}

int main() {

    cin >> n >> m;
    vector<vector<int>> grid(n, vector<int>(m, 0));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cin >> grid[i][j];
        }
    }
    // 标记从第一组边界上的节点出发，可以遍历的节点
    vector<vector<bool>> firstBorder(n, vector<bool>(m, false));

    // 标记从第一组边界上的节点出发，可以遍历的节点
    vector<vector<bool>> secondBorder(n, vector<bool>(m, false));

    // 从最上和最下行的节点出发，向高处遍历
    for (int i = 0; i < n; i++) {
        dfs (grid, firstBorder, i, 0); // 遍历最左列，接触第一组边界
        dfs (grid, secondBorder, i, m - 1); // 遍历最右列，接触第二组边界
    }

    // 从最左和最右列的节点出发，向高处遍历
    for (int j = 0; j < m; j++) {
        dfs (grid, firstBorder, 0, j); // 遍历最上行，接触第一组边界
        dfs (grid, secondBorder, n - 1, j); // 遍历最下行，接触第二组边界
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            // 如果这个节点，从第一组边界和第二组边界出发都遍历过，就是结果
            if (firstBorder[i][j] && secondBorder[i][j]) cout << i << " " << j << endl;;
        }
    }
}
```

#### Python 实现

```python
def pacific_atlantic(heights):
    rows, cols = len(heights), len(heights[0])
    def reach(starts):
        seen, stack = set(starts), list(starts)
        while stack:
            row, col = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen and heights[nr][nc] >= heights[row][col]: seen.add((nr, nc)); stack.append((nr, nc))
        return seen
    pacific = reach([(r, 0) for r in range(rows)] + [(0, c) for c in range(cols)])
    atlantic = reach([(r, cols - 1) for r in range(rows)] + [(rows - 1, c) for c in range(cols)])
    return list(map(list, pacific & atlantic))
```


### 建造最大岛屿


**形象理解**：先给每座已有岛屿刷上独立颜色并记录面积。尝试把一个 0 改成 1 时，只需把上下左右不同颜色岛屿的面积相加，再加上新格子本身。

#### 执行步骤

```text
// 1. DFS 给每座岛分配 id >= 2，并在 area[id] 中记录面积。
// 2. 枚举每个水格 0，建立集合保存四邻域出现的不同岛 id。
// 3. candidate = 1 + 所有不同相邻岛面积之和。
// 4. 用 candidate 更新答案，集合负责防止同一岛重复相加。
// 5. 若没有水格，答案就是整张网格面积。
```

#### C++ 实现

```cpp
int n, m;
int count;

int dir[4][2] = {0, 1, 1, 0, -1, 0, 0, -1}; // 四个方向
void dfs(vector<vector<int>>& grid, vector<vector<bool>>& visited, int x, int y, int mark) {
    if (visited[x][y] || grid[x][y] == 0) return; // 终止条件：访问过的节点 或者 遇到海水
    visited[x][y] = true; // 标记访问过
    grid[x][y] = mark; // 给陆地标记新标签
    count++;
    for (int i = 0; i < 4; i++) {
        int nextx = x + dir[i][0];
        int nexty = y + dir[i][1];
        if (nextx < 0 || nextx >= n || nexty < 0 || nexty >= m) continue;  // 越界了，直接跳过
        dfs(grid, visited, nextx, nexty, mark);
    }
}

int main() {
    cin >> n >> m;
    vector<vector<int>> grid(n, vector<int>(m, 0));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cin >> grid[i][j];
        }
    }
    vector<vector<bool>> visited(n, vector<bool>(m, false)); // 标记访问过的点
    unordered_map<int ,int> gridNum;
    int mark = 2; // 记录每个岛屿的编号
    bool isAllGrid = true; // 标记是否整个地图都是陆地
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (grid[i][j] == 0) isAllGrid = false;
            if (!visited[i][j] && grid[i][j] == 1) {
                count = 0;
                dfs(grid, visited, i, j, mark); // 将与其链接的陆地都标记上 true
                gridNum[mark] = count; // 记录每一个岛屿的面积
                mark++; // 记录下一个岛屿编号
            }
        }
    }
    if (isAllGrid) {
        cout << n * m << endl; // 如果都是陆地，返回全面积
        return 0; // 结束程序
    }

    // 以下逻辑是根据添加陆地的位置，计算周边岛屿面积之和
    int result = 0; // 记录最后结果
    unordered_set<int> visitedGrid; // 标记访问过的岛屿
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            count = 1; // 记录连接之后的岛屿数量
            visitedGrid.clear(); // 每次使用时，清空
            if (grid[i][j] == 0) {
                for (int k = 0; k < 4; k++) {
                    int neari = i + dir[k][1]; // 计算相邻坐标
                    int nearj = j + dir[k][0];
                    if (neari < 0 || neari >= n || nearj < 0 || nearj >= m) continue;
                    if (visitedGrid.count(grid[neari][nearj])) continue; // 添加过的岛屿不要重复添加
                    // 把相邻四面的岛屿数量加起来
                    count += gridNum[grid[neari][nearj]];
                    visitedGrid.insert(grid[neari][nearj]); // 标记该岛屿已经添加过
                }
            }
            result = max(result, count);
        }
    }
}
```

#### Python 实现

```python
def largest_island(grid):
    n, areas, label = len(grid), {0: 0}, 2
    def paint(row, col):
        if not (0 <= row < n and 0 <= col < n) or grid[row][col] != 1: return 0
        grid[row][col] = label
        return 1 + sum(paint(row + dr, col + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    for row in range(n):
        for col in range(n):
            if grid[row][col] == 1: areas[label] = paint(row, col); label += 1
    answer = max(areas.values())
    for row in range(n):
        for col in range(n):
            if grid[row][col] == 0:
                neighbors = {grid[row + dr][col + dc] for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= row + dr < n and 0 <= col + dc < n}
                answer = max(answer, 1 + sum(areas[item] for item in neighbors))
    return answer
```


### 岛屿的周长


**形象理解**：每块陆地最初贡献四条边；每与另一块陆地共享一条边，总周长就少两条，因为两边都变成内部边。

#### 执行步骤

```text
// 1. 扫描每个陆地格，先令 perimeter += 4。
// 2. 只检查上方和左方，避免一对相邻陆地重复处理。
// 3. 上方是陆地时 perimeter -= 2。
// 4. 左方是陆地时 perimeter -= 2。
// 5. 返回最终周长。
```

#### C++ 实现

```cpp
int main() {
    int n, m;
    cin >> n >> m;
    vector<vector<int>> grid(n, vector<int>(m, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cin >> grid[i][j];
        }
    }
    int direction[4][2] = {0, 1, 1, 0, -1, 0, 0, -1};
    int result = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            if (grid[i][j] == 1) {
                for (int k = 0; k < 4; k++) {       // 上下左右四个方向
                    int x = i + direction[k][0];
                    int y = j + direction[k][1];    // 计算周边坐标x,y
                    if (x < 0|| x >= grid.siz()
                            || y <0
                            || y >= grid[0].size()
                            || grid[x][y] == 0) {
                        result++;
                    }
                }
            }
        }
    }
    cout << result << endl;

}
```

#### Python 实现

```python
def island_perimeter(grid):
    land = sum(map(sum, grid)); shared = 0
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col]: shared += (row > 0 and grid[row - 1][col]) + (col > 0 and grid[row][col - 1])
    return 4 * land - 2 * shared
```


### 单词接龙


**形象理解**：每个单词是节点，只差一个字符的单词之间有边。BFS 像从起点同时扩散的波纹，第一次碰到终点时走过的层数一定最少。

#### 执行步骤

```text
// 1. 将 wordList 放入哈希集合；endWord 不存在时直接返回 0。
// 2. beginWord 入队，并记录距离 1。
// 3. 弹出当前单词，逐位置尝试替换为 'a'..'z'。
// 4. 新单词在集合中时立即删除并入队，删除同时完成 visited 标记。
// 5. 第一次生成 endWord 时返回当前距离 + 1。
```

#### C++ 实现

```cpp
int main() {
    string beginStr, endStr, str;
    int n;
    cin >> n;
    unordered_set<string> strSet;
    cin >> beginStr >> endStr;
    for (int i = 0; i < n; i++) {
        cin >> str;
        strSet.insert(str);
    }

    // 记录strSet里的字符串是否被访问过，同时记录路径长度
    unordered_map<string, int> visitMap; // <记录的字符串，路径长度>

    // 初始化队列
    queue<string> que;
    que.push(beginStr);

    // 初始化visitMap
    visitMap.insert(pair<string, int>(beginStr, 1));

    while(!que.empty()) {
        string word = que.front();
        que.pop();
        int path = visitMap[word]; // 这个字符串在路径中的长度

        // 开始在这个str中，挨个字符去替换
        for (int i = 0; i < word.size(); i++) {
            string newWord = word; // 用一个新字符串替换str，因为每次要置换一个字符

            // 遍历26的字母
            for (int j = 0 ; j < 26; j++) {
                newWord[i] = j + 'a';
                if (newWord == endStr) { // 发现替换字母后，字符串与终点字符串相同
                    cout <<  path + 1 << endl; // 找到了路径
                    return 0;
                }
                // 字符串集合里出现了newWord，并且newWord没有被访问过
                if (strSet.find(newWord) != strSet.end()
                        && visitMap.find(newWord) == visitMap.end()) {
                    // 添加访问信息，并将新字符串放到队列中
                    visitMap.insert(pair<string, int>(newWord, path + 1));
                    que.push(newWord);
                }
            }
        }
    }
    // 没找到输出0
    cout << 0 << endl;
}
```

#### Python 实现

```python
from collections import deque

def ladder_length(begin, end, words):
    words, queue = set(words), deque([(begin, 1)])
    while queue:
        word, distance = queue.popleft()
        if word == end: return distance
        for i in range(len(word)):
            for char in "abcdefghijklmnopqrstuvwxyz":
                candidate = word[:i] + char + word[i + 1:]
                if candidate in words: words.remove(candidate); queue.append((candidate, distance + 1))
    return 0
```


## 最小生成树与最短路

### Prim 算法


**形象理解**：已经连通的节点形成一座岛，每轮选择从岛内伸向岛外最便宜的一座桥，把一个新节点纳入岛中。

#### 执行步骤

```text
// 1. minDist[v] 记录当前生成树连接到 v 的最小边权。
// 2. 每轮从未加入节点中选择 minDist 最小的 u。
// 3. 将 u 标记为已加入，并把 minDist[u] 加入总权重。
// 4. 用 u 的边更新所有未加入邻居的 minDist。
// 5. 重复 n 次；有节点始终不可达则图不连通。
```

#### C++ 实现

```cpp
int main() {
    int v, e;
    int x, y, k;
    cin >> v >> e;
    // 填一个默认最大值，题目描述val最大为10000
    vector<vector<int>> grid(v + 1, vector<int>(v + 1, 10001));
    while (e--) {
        cin >> x >> y >> k;
        // 因为是双向图，所以两个方向都要填上
        grid[x][y] = k;
        grid[y][x] = k;

    }
    // 所有节点到最小生成树的最小距离
    vector<int> minDist(v + 1, 10001);

    // 这个节点是否在树里
    vector<bool> isInTree(v + 1, false);

     //加上初始化
    vector<int> parent(v + 1, -1);

    // 我们只需要循环 n-1次，建立 n - 1条边，就可以把n个节点的图连在一起
    for (int i = 1; i < v; i++) {

        // 1、prim三部曲，第一步：选距离生成树最近节点
        int cur = -1; // 选中哪个节点 加入最小生成树
        int minVal = INT_MAX;
        for (int j = 1; j <= v; j++) { // 1 - v，顶点编号，这里下标从1开始
            //  选取最小生成树节点的条件：
            //  （1）不在最小生成树里
            //  （2）距离最小生成树最近的节点
            if (!isInTree[j] &&  minDist[j] < minVal) {
                minVal = minDist[j];
                cur = j;
            }
        }
        // 2、prim三部曲，第二步：最近节点（cur）加入生成树
        isInTree[cur] = true;

        // 3、prim三部曲，第三步：更新非生成树节点到生成树的距离（即更新minDist数组）
        // cur节点加入之后， 最小生成树加入了新的节点，那么所有节点到 最小生成树的距离（即minDist数组）需要更新一下
        // 由于cur节点是新加入到最小生成树，那么只需要关心与 cur 相连的 非生成树节点 的距离 是否比 原来 非生成树节点到生成树节点的距离更小了呢
        for (int j = 1; j <= v; j++) {
            // 更新的条件：
            // （1）节点是 非生成树里的节点
            // （2）与cur相连的某节点的权值 比 该某节点距离最小生成树的距离小
            // 很多录友看到自己 就想不明白什么意思，其实就是 cur 是新加入 最小生成树的节点，那么 所有非生成树的节点距离生成树节点的最近距离 由于 cur的新加入，需要更新一下数据了
            if (!isInTree[j] && grid[cur][j] < minDist[j]) {
                minDist[j] = grid[cur][j];
                parent[j] = cur; // 记录边
            }
        }
    }
    // 统计结果，minDist[i]均为最小生成树的一条边的权值
    int result = 0;
    for (int i = 2; i <= v; i++) { // 不计第一个顶点，因为统计的是边的权值，v个节点有 v-1条边
        result += minDist[i];
    }
}
```

#### Python 实现

```python
import heapq

def prim_mst(n, edges):
    graph = [[] for _ in range(n)]
    for a, b, weight in edges: graph[a].append((weight, b)); graph[b].append((weight, a))
    heap, visited, cost = [(0, 0)], set(), 0
    while heap and len(visited) < n:
        weight, node = heapq.heappop(heap)
        if node in visited: continue
        visited.add(node); cost += weight
        for edge in graph[node]:
            if edge[1] not in visited: heapq.heappush(heap, edge)
    return cost if len(visited) == n else -1
```


### Kruskal 算法


**形象理解**：把所有道路按造价从低到高排队，只要一条路连接的是两个不同村落联盟，就修建它；已经在同一联盟的两端再连会成环，必须跳过。

#### 执行步骤

```text
// 1. 将所有边按权重升序排序。
// 2. 初始化并查集，让每个节点自成连通块。
// 3. 扫描边，若 find(u) != find(v)，选择该边并 unite(u,v)。
// 4. 累加边权；选满 n-1 条边时生成树完成。
// 5. 同集合边直接跳过，避免形成环。
```

#### C++ 实现

```cpp
struct Edge {
    int l, r, val;
};

int n = 10001;

vector<int> father(n, -1);

void init() {
    for (int i = 0; i < n; ++i) {
        father[i] = i;
    }
}

int find(int u) {
    return u == father[u] ? u : father[u] = find(father[u]);
}

void join(int u, int v) {
    u = find(u);
    v = find(v);
    if (u == v) return ;
    father[v] = u;
}

int main() {

    int v, e;
    int v1, v2, val;
    vector<Edge> edges;
    int result_val = 0;
    cin >> v >> e;
    while (e--) {
        cin >> v1 >> v2 >> val;
        edges.push_back({v1, v2, val});
    }

    sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
            return a.val < b.val;
    });

    vector<Edge> result; // 存储最小生成树的边

    init();

    for (Edge edge : edges) {

        int x = find(edge.l);
        int y = find(edge.r);


        if (x != y) {
            result.push_back(edge); // 保存最小生成树的边
            result_val += edge.val;
            join(x, y);
        }
    }

    // 打印最小生成树的边
    for (Edge edge : result) {
        cout << edge.l << " - " << edge.r << " : " << edge.val << endl;
    }

    return 0;
}
```

#### Python 实现

```python
def kruskal_mst(n, edges):
    parent = list(range(n))
    def find(x):
        if parent[x] != x: parent[x] = find(parent[x])
        return parent[x]
    cost = used = 0
    for a, b, weight in sorted(edges, key=lambda edge: edge[2]):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb; cost += weight; used += 1
    return cost if used == n - 1 else -1
```


### Dijkstra 朴素算法


**形象理解**：每轮从尚未确定的城市中选当前距离最近者。非负边保证以后绕路不可能让它更近，因此可以盖章确认，再用它帮助邻居缩短距离。

#### 执行步骤

```text
// 1. dist[source] = 0，其余为无穷大。
// 2. 每轮线性寻找未访问且 dist 最小的节点 u。
// 3. 标记 u 已确定。
// 4. 对 u 的每条边执行 dist[v] = min(dist[v], dist[u] + weight)。
// 5. 重复直到所有可达节点确定，复杂度 O(V^2)。
```

#### C++ 实现

```cpp
int main() {
    int n, m, p1, p2, val;
    cin >> n >> m;

    vector<vector<int>> grid(n + 1, vector<int>(n + 1, INT_MAX));
    for(int i = 0; i < m; i++){
        cin >> p1 >> p2 >> val;
        grid[p1][p2] = val;
    }

    int start = 1;
    int end = n;

    std::vector<int> minDist(n + 1, INT_MAX);
    std::vector<bool> visited(n + 1, false);
    minDist[start] = 0;

    //加上初始化
    vector<int> parent(n + 1, -1);

    for (int i = 1; i <= n; i++) {

        int minVal = INT_MAX;
        int cur = 1;

        for (int v = 1; v <= n; ++v) {
            if (!visited[v] && minDist[v] < minVal) {
                minVal = minDist[v];
                cur = v;
            }
        }

        visited[cur] = true;

        for (int v = 1; v <= n; v++) {
            if (!visited[v] && grid[cur][v] != INT_MAX && minDist[cur] + grid[cur][v] < minDist[v]) {
                minDist[v] = minDist[cur] + grid[cur][v];
                parent[v] = cur; // 记录边
            }
        }

    }

    // 输出最短情况
    for (int i = 1; i <= n; i++) {
        cout << parent[i] << "->" << i << endl;
    }
}
```

#### Python 实现

```python
def dijkstra_dense(graph, source):
    n, distance, visited = len(graph), [float("inf")] * len(graph), [False] * len(graph)
    distance[source] = 0
    for _ in range(n):
        node = min((i for i in range(n) if not visited[i]), key=distance.__getitem__, default=-1)
        if node < 0 or distance[node] == float("inf"): break
        visited[node] = True
        for neighbor, weight in enumerate(graph[node]):
            if weight != float("inf"): distance[neighbor] = min(distance[neighbor], distance[node] + weight)
    return distance
```


### Dijkstra 堆优化


**形象理解**：最小堆替代“每轮在所有城市中找最近者”的线性搜索。堆里可能保留旧报价，弹出时若比最新 `dist` 更差就丢弃。

#### 执行步骤

```text
// 1. 将 (0, source) 放入最小堆。
// 2. 弹出 (distance, u)，若 distance != dist[u]，说明是过期记录，跳过。
// 3. 遍历 u 的邻边，若找到更短路径就更新 dist[v]。
// 4. 每次更新后把新的 (dist[v], v) 压入堆。
// 5. 非负权图上复杂度约 O((V+E)logV)。
```

#### C++ 实现

```cpp
// 小顶堆
class mycomparison {
public:
    bool operator()(const pair<int, int>& lhs, const pair<int, int>& rhs) {
        return lhs.second > rhs.second;
    }
};
// 定义一个结构体来表示带权重的边
struct Edge {
    int to;  // 邻接顶点
    int val; // 边的权重

    Edge(int t, int w): to(t), val(w) {}  // 构造函数
};

int main() {
    int n, m, p1, p2, val;
    cin >> n >> m;

    vector<list<Edge>> grid(n + 1);

    for(int i = 0; i < m; i++){
        cin >> p1 >> p2 >> val;
        // p1 指向 p2，权值为 val
        grid[p1].push_back(Edge(p2, val));

    }

    int start = 1;  // 起点
    int end = n;    // 终点

    // 存储从源点到每个节点的最短距离
    std::vector<int> minDist(n + 1, INT_MAX);

    // 记录顶点是否被访问过
    std::vector<bool> visited(n + 1, false);

    // 优先队列中存放 pair<节点，源点到该节点的权值>
    priority_queue<pair<int, int>, vector<pair<int, int>>, mycomparison> pq;


    // 初始化队列，源点到源点的距离为0，所以初始为0
    pq.push(pair<int, int>(start, 0));

    minDist[start] = 0;  // 起始点到自身的距离为0

    while (!pq.empty()) {
        // 1. 第一步，选源点到哪个节点近且该节点未被访问过 （通过优先级队列来实现）
        // <节点， 源点到该节点的距离>
        pair<int, int> cur = pq.top(); pq.pop();

        if (visited[cur.first]) continue;

        // 2. 第二步，该最近节点被标记访问过
        visited[cur.first] = true;

        // 3. 第三步，更新非访问节点到源点的距离（即更新minDist数组）
        for (Edge edge : grid[cur.first]) { // 遍历 cur指向的节点，cur指向的节点为 edge
            // cur指向的节点edge.to，这条边的权值为 edge.val
            if (!visited[edge.to] && minDist[cur.first] + edge.val < minDist[edge.to]) { // 更新minDist
                minDist[edge.to] = minDist[cur.first] + edge.val;
                pq.push(pair<int, int>(edge.to, minDist[edge.to]));
            }
        }

    }

    if (minDist[end] == INT_MAX) cout << -1 << endl; // 不能到达终点
    else cout << minDist[end] << endl; // 到达终点最短路径
}
```

#### Python 实现

```python
import heapq

def dijkstra(graph, source):
    distance, heap = {source: 0}, [(0, source)]
    while heap:
        cost, node = heapq.heappop(heap)
        if cost != distance[node]: continue
        for neighbor, weight in graph[node]:
            candidate = cost + weight
            if candidate < distance.get(neighbor, float("inf")): distance[neighbor] = candidate; heapq.heappush(heap, (candidate, neighbor))
    return distance
```


### 拓扑排序


**形象理解**：入度为 0 的课程没有任何先修课，可以立即学习。学完它就删除其出边，后续课程的先修数量减少；不断重复直到没有可学课程。

#### 执行步骤

```text
// 1. 建立邻接表并统计每个节点入度。
// 2. 所有入度为 0 的节点入队。
// 3. 弹出节点加入拓扑序，并遍历它指向的邻居。
// 4. 邻居入度减一；降到 0 时入队。
// 5. 最终处理节点数等于 V 则无环，否则存在依赖环。
```

#### C++ 实现

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

#### Python 实现

```python
from collections import deque

def topological_sort(n, edges):
    graph, indegree = [[] for _ in range(n)], [0] * n
    for source, target in edges: graph[source].append(target); indegree[target] += 1
    queue, order = deque(i for i in range(n) if indegree[i] == 0), []
    while queue:
        node = queue.popleft(); order.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0: queue.append(neighbor)
    return order if len(order) == n else []
```


### Bellman-Ford


**形象理解**：最短简单路径最多包含 V-1 条边。每一轮让已知最短距离沿所有边再传播一步，连续做 V-1 轮就能覆盖所有可能的简单路径。

#### 执行步骤

```text
// 1. dist[source] = 0，其余为无穷大。
// 2. 重复 V-1 轮扫描全部边 (u,v,w)。
// 3. u 可达且 dist[u] + w < dist[v] 时松弛 dist[v]。
// 4. 某轮没有任何更新时可提前结束。
// 5. 算法允许负权边，复杂度 O(VE)。
```

#### C++ 实现

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

#### Python 实现

```python
def bellman_ford(n, edges, source):
    distance = [float("inf")] * n; distance[source] = 0
    for _ in range(n - 1):
        changed = False
        for a, b, weight in edges:
            if distance[a] + weight < distance[b]: distance[b] = distance[a] + weight; changed = True
        if not changed: break
    return distance
```


### SPFA 队列优化


**形象理解**：Bellman-Ford 每轮检查所有边，SPFA 只让“距离刚刚变小的节点”去通知邻居；没有新消息的节点无需重复广播。

#### 执行步骤

```text
// 1. source 入队，并用 inQueue 防止同一节点重复排队。
// 2. 弹出 u 后令 inQueue[u] = false。
// 3. 松弛 u 的所有出边；dist[v] 变小时才需要继续传播。
// 4. 若 v 当前不在队列中，将它入队并标记。
// 5. 平均可能更快，但最坏复杂度仍为 O(VE)。
```

#### C++ 实现

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

#### Python 实现

```python
from collections import deque

def spfa(n, graph, source):
    distance, queued, queue = [float("inf")] * n, [False] * n, deque([source])
    distance[source] = 0; queued[source] = True
    while queue:
        node = queue.popleft(); queued[node] = False
        for neighbor, weight in graph[node]:
            if distance[node] + weight < distance[neighbor]:
                distance[neighbor] = distance[node] + weight
                if not queued[neighbor]: queue.append(neighbor); queued[neighbor] = True
    return distance
```


### Bellman-Ford 判断负权回路


**形象理解**：V-1 轮后正常最短路已经稳定；若第 V 轮还能变短，说明路径可以绕某个负权环不断降价，不存在有限最短值。

#### 执行步骤

```text
// 1. 先完成至多 V-1 轮正常松弛。
// 2. 再额外扫描全部边一次。
// 3. 若仍存在 dist[u] + w < dist[v]，说明源点可达的负环存在。
// 4. SPFA 版本也可统计节点进入路径/队列的次数，达到 V 时判负环。
```

#### C++ 实现

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

#### Python 实现

```python
def has_negative_cycle(n, edges):
    distance = [0] * n
    for iteration in range(n):
        changed = False
        for a, b, weight in edges:
            if distance[a] + weight < distance[b]: distance[b] = distance[a] + weight; changed = True
        if not changed: return False
    return True
```


### Bellman-Ford 单源有限最短路径


**形象理解**：题目限制最多经过 k 条边，就让距离只传播 k 轮。每轮必须读取上一轮的快照，避免同一轮连续使用多条边而突破边数限制。

#### 执行步骤

```text
// 1. dist[source] = 0，其余为无穷大。
// 2. 重复 k 轮，先复制 backup = dist。
// 3. 扫描边时使用 backup[u] + w 更新 dist[v]。
// 4. backup 确保第 i 轮只从“最多 i-1 条边”的结果转移。
// 5. k 轮后 dist[target] 即边数受限的最短距离。
```

#### C++ 实现

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

#### Python 实现

```python
def shortest_path_with_at_most_k_edges(n, edges, source, k):
    distance = [float("inf")] * n; distance[source] = 0
    for _ in range(k):
        previous = distance[:]
        for a, b, weight in edges:
            distance[b] = min(distance[b], previous[a] + weight)
    return distance
```


### Floyd 算法


**形象理解**：逐个开放中转站 k。开放第 k 个站后，任意 i 到 j 的路线可以保持原路，也可以改走 `i -> k -> j`，选更短者。

#### 执行步骤

```text
// 1. dist[i][i] = 0，有直接边时写入边权，其余为无穷大。
// 2. 最外层枚举中间节点 k，表示当前只允许使用 0..k 作中转。
// 3. 再枚举起点 i 和终点 j。
// 4. 两段都可达时更新 dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])。
// 5. k 必须放在最外层；完成后得到任意两点最短路。
```

#### C++ 实现

```cpp
vector<vector<long long>> floydWarshall(vector<vector<long long>> dist) {
    int n = dist.size();
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (dist[i][k] != LLONG_MAX && dist[k][j] != LLONG_MAX) {
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
                }
            }
        }
    }
    return dist;
}
```

#### Python 实现

```python
def floyd_warshall(distance):
    n = len(distance)
    for middle in range(n):
        for source in range(n):
            for target in range(n):
                distance[source][target] = min(distance[source][target], distance[source][middle] + distance[middle][target])
    return distance
```


## 系列导航

- [原专题文章索引](#原代码索引)
- 下一篇：无（本篇是当前 LeetCode 专题终点）

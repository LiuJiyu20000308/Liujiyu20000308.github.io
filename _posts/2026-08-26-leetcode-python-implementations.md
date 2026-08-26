---
layout: post
title: LeetCode 例题 Python 实现全集
date: 2026-08-26 00:01 +0800
tags: [数据结构与算法]
toc: true
permalink: /leetcode/python-implementations/
---

这篇文章为站内 LeetCode 专题提供 Python 版本。条目顺序与[全部例题复习导读]({% post_url 2026-8-10-LeetcodeExamples %})一致；代码保留 LeetCode 常用接口，`ListNode`、`TreeNode`、`Node` 等由题目环境提供。原文中的 C++ 推导和复杂度分析继续保留，这里只集中维护 Python 实现。

## 二分查找和双指针

### 基本二分查找 {#py-001}

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

### 二分查找左右边界 {#py-002}

```python
from bisect import bisect_left, bisect_right

def search_range(nums, target):
    left = bisect_left(nums, target)
    if left == len(nums) or nums[left] != target:
        return [-1, -1]
    return [left, bisect_right(nums, target) - 1]
```

### 两个有序数组的中位数 {#py-003}

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

### 链表的中间结点 {#py-004}

```python
def middle_node(head):
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
    return slow
```

### 删除有序数组中的重复项 {#py-005}

```python
def remove_duplicates(nums):
    write = 0
    for value in nums:
        if write == 0 or value != nums[write - 1]:
            nums[write] = value
            write += 1
    return write
```

### 有序数组的平方 {#py-006}

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

### 反转字符串 {#py-007}

```python
def reverse_string(chars):
    left, right = 0, len(chars) - 1
    while left < right:
        chars[left], chars[right] = chars[right], chars[left]
        left, right = left + 1, right - 1
```

### 最长回文子串 {#py-008}

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

### 长度最小的子数组 {#py-009}

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

### 无重复字符的最长子串 {#py-010}

```python
def length_of_longest_substring(s):
    last, left, answer = {}, 0, 0
    for right, char in enumerate(s):
        left = max(left, last.get(char, -1) + 1)
        last[char] = right
        answer = max(answer, right - left + 1)
    return answer
```

### 最大连续 1 的个数 III {#py-011}

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

### 最小覆盖子串 {#py-012}

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

### 水果成篮 {#py-013}

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

### 找出所有字母异位词 {#py-014}

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

### 串联所有单词的子串 {#py-015}

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

### 字符串的排列 {#py-016}

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

### 除自身以外数组的乘积 {#py-017}

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

### 和为 K 的子数组 {#py-018}

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

### 和可被 K 整除的子数组 {#py-019}

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

### 连续数组 {#py-020}

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

## 模拟过程与链表

### 螺旋矩阵 II {#py-021}

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

### 删除链表元素与虚拟头结点 {#py-022}

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

### 设计链表 {#py-023}

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

### 反转链表 {#py-024}

```python
def reverse_list(head):
    previous = None
    while head:
        head.next, previous, head = previous, head, head.next
    return previous
```

### K 个一组反转链表 {#py-025}

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

### 两两交换链表节点 {#py-026}

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

### 删除链表倒数第 N 个节点 {#py-027}

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

### 链表相交 {#py-028}

```python
def get_intersection_node(head_a, head_b):
    a, b = head_a, head_b
    while a is not b:
        a = a.next if a else head_b
        b = b.next if b else head_a
    return a
```

### 环形链表入口 {#py-029}

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

## 哈希表、字符串、栈和队列

### 有效的字母异位词 {#py-030}

```python
from collections import Counter

def is_anagram(s, t):
    return Counter(s) == Counter(t)
```

### 两个数组的交集 {#py-031}

```python
def intersection(nums1, nums2):
    return list(set(nums1) & set(nums2))
```

### 快乐数 {#py-032}

```python
def is_happy(n):
    seen = set()
    while n != 1 and n not in seen:
        seen.add(n)
        n = sum(int(digit) ** 2 for digit in str(n))
    return n == 1
```

### 两数之和 {#py-033}

```python
def two_sum(nums, target):
    seen = {}
    for i, value in enumerate(nums):
        if target - value in seen:
            return [seen[target - value], i]
        seen[value] = i
```

### 四数相加 II {#py-034}

```python
from collections import Counter

def four_sum_count(a, b, c, d):
    pair_sums = Counter(x + y for x in a for y in b)
    return sum(pair_sums[-x - y] for x in c for y in d)
```

### 三数之和 {#py-035}

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

### 四数之和 {#py-036}

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

### 反转字符串 II {#py-037}

```python
def reverse_str(s, k):
    chars = list(s)
    for start in range(0, len(chars), 2 * k):
        chars[start:start + k] = reversed(chars[start:start + k])
    return "".join(chars)
```

### 替换数字 {#py-038}

```python
def replace_digits(s):
    return "".join("number" if char.isdigit() else char for char in s)
```

### 反转字符串中的单词 {#py-039}

```python
def reverse_words(s):
    return " ".join(reversed(s.split()))
```

### 右旋字符串 {#py-040}

```python
def rotate_right(s, k):
    if not s:
        return s
    k %= len(s)
    return s[-k:] + s[:-k]
```

### 找出字符串中第一个匹配下标 {#py-041}

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

### 重复的子字符串 {#py-042}

```python
def repeated_substring_pattern(s):
    return s in (s + s)[1:-1]
```

### 用栈实现队列 {#py-043}

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

### 用队列实现栈 {#py-044}

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

### 有效的括号 {#py-045}

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

### 删除相邻重复项 {#py-046}

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

### 中缀表达式转后缀表达式 {#py-047}

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

### 逆波兰表达式求值 {#py-048}

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

### 前 K 个高频元素 {#py-049}

```python
from collections import Counter

def top_k_frequent(nums, k):
    return [value for value, _ in Counter(nums).most_common(k)]
```

### 滑动窗口最大值 {#py-050}

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

### 和至少为 K 的最短子数组 {#py-051}

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

### 选择排序 {#py-052}

```python
def selection_sort(nums):
    for i in range(len(nums)):
        smallest = min(range(i, len(nums)), key=nums.__getitem__)
        nums[i], nums[smallest] = nums[smallest], nums[i]
    return nums
```

### 冒泡排序 {#py-053}

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

### 插入排序 {#py-054}

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

### 希尔排序 {#py-055}

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

### 归并排序 {#py-056}

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

### 快速排序 {#py-057}

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

### 堆排序与 `priority_queue` {#py-058}

```python
import heapq

def heap_sort(nums):
    heapq.heapify(nums)
    return [heapq.heappop(nums) for _ in range(len(nums))]
```

### 计数排序 {#py-059}

```python
def counting_sort(nums):
    if not nums: return []
    low, high = min(nums), max(nums)
    count = [0] * (high - low + 1)
    for value in nums: count[value - low] += 1
    return [value for i, total in enumerate(count) for value in [i + low] * total]
```

### 桶排序 {#py-060}

```python
def bucket_sort(nums, bucket_size=10):
    if not nums: return []
    low = min(nums)
    buckets = [[] for _ in range((max(nums) - low) // bucket_size + 1)]
    for value in nums: buckets[(value - low) // bucket_size].append(value)
    return [value for bucket in buckets for value in sorted(bucket)]
```

### 基数排序 {#py-061}

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

### Introsort {#py-062}

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

### 滑动窗口中位数 {#py-063}

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

### 并行归并排序与快速排序 {#py-064}

```python
from concurrent.futures import ProcessPoolExecutor

def parallel_sort(chunks):
    with ProcessPoolExecutor() as pool:
        sorted_chunks = list(pool.map(sorted, chunks))
    import heapq
    return list(heapq.merge(*sorted_chunks))
```

### LRU 缓存 {#py-065}

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

### LFU 缓存 {#py-066}

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

### 基于时间的键值存储 TimeMap {#py-067}

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

### 带 TTL 的令牌验证系统 {#py-068}

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

### 全 O(1) 数据结构 AllOne {#py-069}

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

## 二叉树与二叉搜索树

### 递归前序、中序和后序遍历 {#py-070}

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

### 迭代前序、中序和后序遍历 {#py-071}

```python
def inorder_traversal(root):
    answer, stack, node = [], [], root
    while stack or node:
        while node: stack.append(node); node = node.left
        node = stack.pop(); answer.append(node.val); node = node.right
    return answer
```

### 统一格式迭代遍历 {#py-072}

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

### 二叉树层序遍历 {#py-073}

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

### 二叉树的右视图 {#py-074}

```python
def right_side_view(root):
    return [level[-1] for level in level_order(root)]
```

### 二叉树每层的平均值 {#py-075}

```python
def average_of_levels(root):
    return [sum(level) / len(level) for level in level_order(root)]
```

### 填充每个节点的下一个右侧节点指针 {#py-076}

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

### 对称二叉树 {#py-077}

```python
def is_symmetric(root):
    def mirror(left, right):
        if not left or not right: return left is right
        return left.val == right.val and mirror(left.left, right.right) and mirror(left.right, right.left)
    return mirror(root.left, root.right) if root else True
```

### 二叉树和 N 叉树的最大深度 {#py-078}

```python
def max_depth(root):
    if not root: return 0
    children = getattr(root, "children", None)
    if children is not None:
        return 1 + max((max_depth(child) for child in children), default=0)
    return 1 + max(max_depth(root.left), max_depth(root.right))
```

### 二叉树的最小深度 {#py-079}

```python
def min_depth(root):
    if not root: return 0
    if not root.left: return 1 + min_depth(root.right)
    if not root.right: return 1 + min_depth(root.left)
    return 1 + min(min_depth(root.left), min_depth(root.right))
```

### 平衡二叉树 {#py-080}

```python
def is_balanced(root):
    def height(node):
        if not node: return 0
        left, right = height(node.left), height(node.right)
        if left < 0 or right < 0 or abs(left - right) > 1: return -1
        return 1 + max(left, right)
    return height(root) >= 0
```

### 二叉树的所有路径 {#py-081}

```python
def binary_tree_paths(root):
    if not root: return []
    if not root.left and not root.right: return [str(root.val)]
    return [f"{root.val}->{path}" for child in (root.left, root.right) if child for path in binary_tree_paths(child)]
```

### 路径总和 {#py-082}

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

### 左叶子之和 {#py-083}

```python
def sum_of_left_leaves(root):
    if not root: return 0
    left = root.left.val if root.left and not root.left.left and not root.left.right else sum_of_left_leaves(root.left)
    return left + sum_of_left_leaves(root.right)
```

### 找树左下角的值 {#py-084}

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

### 翻转二叉树 {#py-085}

```python
def invert_tree(root):
    if root:
        root.left, root.right = invert_tree(root.right), invert_tree(root.left)
    return root
```

### 根据前序和中序遍历构造二叉树 {#py-086}

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

### 搜索和插入 BST {#py-087}

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

### 删除 BST 节点 {#py-088}

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

### 验证 BST {#py-089}

```python
def is_valid_bst(root):
    def validate(node, low, high):
        return not node or low < node.val < high and validate(node.left, low, node.val) and validate(node.right, node.val, high)
    return validate(root, float("-inf"), float("inf"))
```

### BST 转双向链表 {#py-090}

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

### BST 中的最小绝对差和众数 {#py-091}

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

### 二叉树与 BST 的最近公共祖先 {#py-092}

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

### 修剪 BST {#py-093}

```python
def trim_bst(root, low, high):
    if not root: return None
    if root.val < low: return trim_bst(root.right, low, high)
    if root.val > high: return trim_bst(root.left, low, high)
    root.left, root.right = trim_bst(root.left, low, high), trim_bst(root.right, low, high)
    return root
```

### BST 转累加树 {#py-094}

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

### 有序数组转平衡 BST {#py-095}

```python
def sorted_array_to_bst(nums):
    if not nums: return None
    middle = len(nums) // 2
    return TreeNode(nums[middle], sorted_array_to_bst(nums[:middle]), sorted_array_to_bst(nums[middle + 1:]))
```

### 每日温度 {#py-096}

```python
def daily_temperatures(temperatures):
    answer, stack = [0] * len(temperatures), []
    for i, value in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < value:
            old = stack.pop(); answer[old] = i - old
        stack.append(i)
    return answer
```

### 最大二叉树 {#py-097}

```python
def construct_maximum_binary_tree(nums):
    if not nums: return None
    i = max(range(len(nums)), key=nums.__getitem__)
    return TreeNode(nums[i], construct_maximum_binary_tree(nums[:i]), construct_maximum_binary_tree(nums[i + 1:]))
```

### 接雨水 {#py-098}

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

### 柱状图中最大的矩形 {#py-099}

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

### 组合 {#py-100}

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

### 组合总和 I {#py-101}

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

### 组合总和 II（原记录“组合求和 III”，Leetcode 40） {#py-102}

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

### 组合总和 III（原记录“组合求和 II”，Leetcode 216） {#py-103}

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

### 分割回文串 {#py-104}

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

### 复原 IP 地址 {#py-105}

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

### 子集 I {#py-106}

```python
def subsets(nums):
    answer = [[]]
    for value in nums: answer += [part + [value] for part in answer]
    return answer
```

### 子集 II {#py-107}

```python
def subsets_with_dup(nums):
    nums.sort(); answer = [[]]; previous_size = 0
    for i, value in enumerate(nums):
        start = previous_size if i and nums[i] == nums[i - 1] else 0
        previous_size = len(answer)
        answer += [answer[j] + [value] for j in range(start, previous_size)]
    return answer
```

### 递增子序列 {#py-108}

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

### 全排列 I {#py-109}

```python
def permute(nums):
    if not nums: return [[]]
    return [[nums[i]] + tail for i in range(len(nums)) for tail in permute(nums[:i] + nums[i + 1:])]
```

### 全排列 II {#py-110}

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

### N 皇后 {#py-111}

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

### 数独 {#py-112}

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

### 重新安排行程 {#py-113}

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

### 分发饼干 {#py-114}

```python
def find_content_children(children, cookies):
    children.sort(); cookies.sort(); child = 0
    for cookie in cookies:
        if child < len(children) and cookie >= children[child]: child += 1
    return child
```

### 摆动序列 {#py-115}

```python
def wiggle_max_length(nums):
    up = down = 1
    for a, b in zip(nums, nums[1:]):
        if b > a: up = down + 1
        elif b < a: down = up + 1
    return max(up, down)
```

### 最大子序和 {#py-116}

```python
def max_sub_array(nums):
    current = answer = nums[0]
    for value in nums[1:]:
        current = max(value, current + value); answer = max(answer, current)
    return answer
```

### 买卖股票的最佳时机 II {#py-117}

```python
def max_profit_unlimited(prices):
    return sum(max(0, right - left) for left, right in zip(prices, prices[1:]))
```

### 跳跃游戏 I {#py-118}

```python
def can_jump(nums):
    farthest = 0
    for i, jump in enumerate(nums):
        if i > farthest: return False
        farthest = max(farthest, i + jump)
    return True
```

### 跳跃游戏 II {#py-119}

```python
def jump(nums):
    steps = end = farthest = 0
    for i in range(len(nums) - 1):
        farthest = max(farthest, i + nums[i])
        if i == end: steps, end = steps + 1, farthest
    return steps
```

### K 次取反后最大化数组和 {#py-120}

```python
def largest_sum_after_k_negations(nums, k):
    nums.sort(key=abs, reverse=True)
    for i in range(len(nums)):
        if nums[i] < 0 and k: nums[i], k = -nums[i], k - 1
    if k % 2: nums[-1] = -nums[-1]
    return sum(nums)
```

### 加油站 {#py-121}

```python
def can_complete_circuit(gas, cost):
    total = tank = start = 0
    for i, (supply, expense) in enumerate(zip(gas, cost)):
        delta = supply - expense; total += delta; tank += delta
        if tank < 0: start, tank = i + 1, 0
    return start if total >= 0 else -1
```

### 分发糖果 {#py-122}

```python
def candy(ratings):
    sweets = [1] * len(ratings)
    for i in range(1, len(ratings)):
        if ratings[i] > ratings[i - 1]: sweets[i] = sweets[i - 1] + 1
    for i in range(len(ratings) - 2, -1, -1):
        if ratings[i] > ratings[i + 1]: sweets[i] = max(sweets[i], sweets[i + 1] + 1)
    return sum(sweets)
```

### 柠檬水找零 {#py-123}

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

### 根据身高重建队列 {#py-124}

```python
def reconstruct_queue(people):
    answer = []
    for person in sorted(people, key=lambda item: (-item[0], item[1])):
        answer.insert(person[1], person)
    return answer
```

### 用最少数量的箭引爆气球 {#py-125}

```python
def find_min_arrow_shots(points):
    arrows, end = 0, float("-inf")
    for start, finish in sorted(points, key=lambda point: point[1]):
        if start > end: arrows, end = arrows + 1, finish
    return arrows
```

### 无重叠区间 {#py-126}

```python
def erase_overlap_intervals(intervals):
    kept, end = 0, float("-inf")
    for start, finish in sorted(intervals, key=lambda interval: interval[1]):
        if start >= end: kept, end = kept + 1, finish
    return len(intervals) - kept
```

### 划分字母区间 {#py-127}

```python
def partition_labels(s):
    last, start, end, answer = {char: i for i, char in enumerate(s)}, 0, 0, []
    for i, char in enumerate(s):
        end = max(end, last[char])
        if i == end: answer.append(end - start + 1); start = i + 1
    return answer
```

### 合并区间 {#py-128}

```python
def merge(intervals):
    answer = []
    for interval in sorted(intervals):
        if not answer or interval[0] > answer[-1][1]: answer.append(interval)
        else: answer[-1][1] = max(answer[-1][1], interval[1])
    return answer
```

### 单调递增的数字 {#py-129}

```python
def monotone_increasing_digits(n):
    digits = list(str(n)); marker = len(digits)
    for i in range(len(digits) - 1, 0, -1):
        if digits[i - 1] > digits[i]:
            digits[i - 1] = str(int(digits[i - 1]) - 1); marker = i
    digits[marker:] = "9" * (len(digits) - marker)
    return int("".join(digits))
```

### 监控二叉树 {#py-130}

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

## 动态规划

### 爬楼梯 {#py-131}

```python
def climb_stairs(n):
    first, second = 1, 1
    for _ in range(n): first, second = second, first + second
    return first
```

### 使用最小花费爬楼梯 {#py-132}

```python
def min_cost_climbing_stairs(cost):
    previous, current = 0, 0
    for value in cost: previous, current = current, min(previous, current) + value
    return min(previous, current)
```

### 不同路径 {#py-133}

```python
def unique_paths(m, n):
    dp = [1] * n
    for _ in range(1, m):
        for col in range(1, n): dp[col] += dp[col - 1]
    return dp[-1]
```

### 不同路径 II {#py-134}

```python
def unique_paths_with_obstacles(grid):
    dp = [0] * len(grid[0]); dp[0] = 1
    for row in grid:
        for col, blocked in enumerate(row):
            dp[col] = 0 if blocked else dp[col] + (dp[col - 1] if col else 0)
    return dp[-1]
```

### 整数拆分 {#py-135}

```python
def integer_break(n):
    dp = [0] * (n + 1); dp[1] = 1
    for total in range(2, n + 1):
        dp[total] = max(max(part * (total - part), part * dp[total - part]) for part in range(1, total))
    return dp[n]
```

### 不同的二叉搜索树 {#py-136}

```python
def num_trees(n):
    dp = [1] + [0] * n
    for nodes in range(1, n + 1):
        dp[nodes] = sum(dp[root - 1] * dp[nodes - root] for root in range(1, nodes + 1))
    return dp[n]
```

### 打家劫舍 I {#py-137}

```python
def rob(nums):
    skip = take = 0
    for value in nums: skip, take = max(skip, take), skip + value
    return max(skip, take)
```

### 打家劫舍 II {#py-138}

```python
def rob_circle(nums):
    if len(nums) == 1: return nums[0]
    return max(rob(nums[:-1]), rob(nums[1:]))
```

### 打家劫舍 III {#py-139}

```python
def rob_tree(root):
    def dfs(node):
        if not node: return 0, 0
        left, right = dfs(node.left), dfs(node.right)
        return node.val + left[1] + right[1], max(left) + max(right)
    return max(dfs(root))
```

### 只能买卖一次 {#py-140}

```python
def max_profit_once(prices):
    minimum, answer = float("inf"), 0
    for price in prices: minimum, answer = min(minimum, price), max(answer, price - minimum)
    return answer
```

### 可以买卖任意次 {#py-141}

```python
def max_profit_many(prices):
    cash, hold = 0, float("-inf")
    for price in prices: cash, hold = max(cash, hold + price), max(hold, cash - price)
    return cash
```

### 最多买卖两次 {#py-142}

```python
def max_profit_twice(prices):
    buy1 = buy2 = float("-inf"); sell1 = sell2 = 0
    for price in prices:
        buy1 = max(buy1, -price); sell1 = max(sell1, buy1 + price)
        buy2 = max(buy2, sell1 - price); sell2 = max(sell2, buy2 + price)
    return sell2
```

### 最多买卖 K 次 {#py-143}

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

### 含冷冻期的股票交易 {#py-144}

```python
def max_profit_cooldown(prices):
    hold, sold, rest = float("-inf"), 0, 0
    for price in prices: hold, sold, rest = max(hold, rest - price), hold + price, max(rest, sold)
    return max(sold, rest)
```

### 含手续费的股票交易 {#py-145}

```python
def max_profit_fee(prices, fee):
    cash, hold = 0, -prices[0]
    for price in prices[1:]: cash, hold = max(cash, hold + price - fee), max(hold, cash - price)
    return cash
```

### 二维 0-1 背包 {#py-146}

```python
def knapsack_2d(weights, values, capacity):
    dp = [[0] * (capacity + 1) for _ in range(len(weights) + 1)]
    for i, (weight, value) in enumerate(zip(weights, values), 1):
        for cap in range(capacity + 1):
            dp[i][cap] = dp[i - 1][cap]
            if cap >= weight: dp[i][cap] = max(dp[i][cap], dp[i - 1][cap - weight] + value)
    return dp[-1][-1]
```

### 一维 0-1 背包 {#py-147}

```python
def knapsack(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for weight, value in zip(weights, values):
        for cap in range(capacity, weight - 1, -1): dp[cap] = max(dp[cap], dp[cap - weight] + value)
    return dp[-1]
```

### 分割等和子集 {#py-148}

```python
def can_partition(nums):
    total = sum(nums)
    if total % 2: return False
    reachable = 1
    for value in nums: reachable |= reachable << value
    return bool(reachable >> (total // 2) & 1)
```

### 最后一块石头的重量 II {#py-149}

```python
def last_stone_weight_ii(stones):
    possible = {0}
    for stone in stones: possible |= {total + stone for total in possible}
    half = max(total for total in possible if total <= sum(stones) // 2)
    return sum(stones) - 2 * half
```

### 目标和 {#py-150}

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

### 一和零 {#py-151}

```python
def find_max_form(strings, m, n):
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for string in strings:
        zeros, ones = string.count("0"), string.count("1")
        for i in range(m, zeros - 1, -1):
            for j in range(n, ones - 1, -1): dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)
    return dp[m][n]
```

### 完全背包的遍历顺序 {#py-152}

```python
def complete_knapsack(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for weight, value in zip(weights, values):
        for cap in range(weight, capacity + 1): dp[cap] = max(dp[cap], dp[cap - weight] + value)
    return dp[-1]
```

### 零钱兑换 II {#py-153}

```python
def change(amount, coins):
    dp = [1] + [0] * amount
    for coin in coins:
        for total in range(coin, amount + 1): dp[total] += dp[total - coin]
    return dp[amount]
```

### 组合总和 IV {#py-154}

```python
def combination_sum4(nums, target):
    dp = [1] + [0] * target
    for total in range(1, target + 1): dp[total] = sum(dp[total - value] for value in nums if value <= total)
    return dp[target]
```

### 爬楼梯作为完全背包 {#py-155}

```python
def climb_stairs_complete(n, steps=(1, 2)):
    dp = [1] + [0] * n
    for total in range(1, n + 1): dp[total] = sum(dp[total - step] for step in steps if step <= total)
    return dp[n]
```

### 零钱兑换 {#py-156}

```python
def coin_change(coins, amount):
    dp = [0] + [amount + 1] * amount
    for total in range(1, amount + 1): dp[total] = min((dp[total - coin] + 1 for coin in coins if coin <= total), default=amount + 1)
    return -1 if dp[amount] > amount else dp[amount]
```

### 完全平方数 {#py-157}

```python
def num_squares(n):
    dp = [0] + [n] * n
    for total in range(1, n + 1): dp[total] = 1 + min(dp[total - root * root] for root in range(1, int(total ** 0.5) + 1))
    return dp[n]
```

### 单词拆分 {#py-158}

```python
def word_break(s, word_dict):
    words, dp = set(word_dict), [True] + [False] * len(s)
    for end in range(1, len(s) + 1): dp[end] = any(dp[start] and s[start:end] in words for start in range(end))
    return dp[-1]
```

### 最长递增子序列 {#py-159}

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

### 最长连续递增序列 {#py-160}

```python
def find_length_of_lcis(nums):
    current = answer = int(bool(nums))
    for left, right in zip(nums, nums[1:]):
        current = current + 1 if right > left else 1; answer = max(answer, current)
    return answer
```

### 最长重复子数组 {#py-161}

```python
def find_length(nums1, nums2):
    dp, answer = [0] * (len(nums2) + 1), 0
    for a in nums1:
        for j in range(len(nums2) - 1, -1, -1):
            dp[j + 1] = dp[j] + 1 if a == nums2[j] else 0; answer = max(answer, dp[j + 1])
    return answer
```

### 最长公共子序列 {#py-162}

```python
def longest_common_subsequence(a, b):
    dp = [0] * (len(b) + 1)
    for char_a in a:
        diagonal = 0
        for j, char_b in enumerate(b, 1):
            old = dp[j]; dp[j] = diagonal + 1 if char_a == char_b else max(dp[j], dp[j - 1]); diagonal = old
    return dp[-1]
```

### 不相交的线 {#py-163}

```python
def max_uncrossed_lines(nums1, nums2):
    return longest_common_subsequence(nums1, nums2)
```

### 动态规划版本的最大子序和 {#py-164}

```python
def max_sub_array_dp(nums):
    dp = nums[:]
    for i in range(1, len(nums)): dp[i] = max(nums[i], dp[i - 1] + nums[i])
    return max(dp)
```

### 判断子序列 {#py-165}

```python
def is_subsequence(s, t):
    iterator = iter(t)
    return all(char in iterator for char in s)
```

### 不同的子序列 {#py-166}

```python
def num_distinct(s, t):
    dp = [1] + [0] * len(t)
    for source in s:
        for j in range(len(t) - 1, -1, -1):
            if source == t[j]: dp[j + 1] += dp[j]
    return dp[-1]
```

### 两个字符串的删除操作 {#py-167}

```python
def min_distance_delete(word1, word2):
    common = longest_common_subsequence(word1, word2)
    return len(word1) + len(word2) - 2 * common
```

### 编辑距离 {#py-168}

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

### 回文子串 {#py-169}

```python
def count_substrings(s):
    answer = 0
    for center in range(2 * len(s) - 1):
        left, right = center // 2, (center + 1) // 2
        while left >= 0 and right < len(s) and s[left] == s[right]: answer += 1; left -= 1; right += 1
    return answer
```

### 最长回文子序列 {#py-170}

```python
def longest_palindrome_subseq(s):
    return longest_common_subsequence(s, s[::-1])
```

### 正则表达式匹配 {#py-171}

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

## 图论

### 寻找图中是否存在路径 {#py-172}

```python
def valid_path(n, edges, source, destination):
    parent = list(range(n))
    def find(x):
        while x != parent[x]: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in edges: parent[find(a)] = find(b)
    return find(source) == find(destination)
```

### 冗余连接 {#py-173}

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

### 冗余连接 II {#py-174}

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

### 所有可能的路径 {#py-175}

```python
def all_paths_source_target(graph):
    answer = []
    def dfs(node, path):
        if node == len(graph) - 1: answer.append(path); return
        for neighbor in graph[node]: dfs(neighbor, path + [neighbor])
    dfs(0, [0])
    return answer
```

### 岛屿数量 {#py-176}

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

### 岛屿的最大面积 {#py-177}

```python
def max_area_of_island(grid):
    def flood(row, col):
        if row < 0 or row == len(grid) or col < 0 or col == len(grid[0]) or not grid[row][col]: return 0
        grid[row][col] = 0
        return 1 + sum(flood(row + dr, col + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
    return max((flood(r, c) for r in range(len(grid)) for c in range(len(grid[0]))), default=0)
```

### 孤岛的最大面积 {#py-178}

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

### 沉没孤岛（被围绕的区域） {#py-179}

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

### 太平洋大西洋水流问题 {#py-180}

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

### 建造最大岛屿 {#py-181}

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

### 岛屿的周长 {#py-182}

```python
def island_perimeter(grid):
    land = sum(map(sum, grid)); shared = 0
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col]: shared += (row > 0 and grid[row - 1][col]) + (col > 0 and grid[row][col - 1])
    return 4 * land - 2 * shared
```

### 单词接龙 {#py-183}

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

### Prim 算法 {#py-184}

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

### Kruskal 算法 {#py-185}

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

### Dijkstra 朴素算法 {#py-186}

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

### Dijkstra 堆优化 {#py-187}

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

### 拓扑排序 {#py-188}

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

### Bellman-Ford {#py-189}

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

### SPFA 队列优化 {#py-190}

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

### Bellman-Ford 判断负权回路 {#py-191}

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

### Bellman-Ford 单源有限最短路径 {#py-192}

```python
def shortest_path_with_at_most_k_edges(n, edges, source, k):
    distance = [float("inf")] * n; distance[source] = 0
    for _ in range(k):
        previous = distance[:]
        for a, b, weight in edges:
            distance[b] = min(distance[b], previous[a] + weight)
    return distance
```

### Floyd 算法 {#py-193}

```python
def floyd_warshall(distance):
    n = len(distance)
    for middle in range(n):
        for source in range(n):
            for target in range(n):
                distance[source][target] = min(distance[source][target], distance[source][middle] + distance[middle][target])
    return distance
```

## 导航

- 上一篇：[全部例题直观解释与代码执行步骤]({% post_url 2026-8-10-LeetcodeExamples %})
- [算法笔记总览]({{ '/notes/' | relative_url }})
- 下一篇：无（本篇是当前 LeetCode 专题终点）

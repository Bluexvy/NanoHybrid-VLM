#include <bits/stdc++.h>
using namespace std;

// 链表节点
struct ListNode {
    int val;
    ListNode* next;

    ListNode(int value) {
        val = value;
        next = nullptr;
    }
};

// 根据数组建立链表
ListNode* buildList(vector<int>& nums) {
    ListNode dummy(0);
    ListNode* tail = &dummy;

    for (int value : nums) {
        ListNode* newNode = new ListNode(value);

        tail->next = newNode;
        tail = newNode;
    }

    return dummy.next;
}

// 从 start 开始向后移动 k 次
// 返回移动之后到达的节点
ListNode* getKthNode(ListNode* start, int k) {
    while (start != nullptr && k > 0) {
        start = start->next;
        --k;
    }

    return start;
}

// K 个一组翻转链表
ListNode* reverseKGroup(ListNode* head, int k) {
    ListNode dummy(0);
    dummy.next = head;

    // 当前这一组前面的节点
    ListNode* groupPrev = &dummy;

    while (true) {
        // 找到当前这一组的第 k 个节点
        ListNode* kthNode = getKthNode(groupPrev, k);

        // 如果不足 k 个节点，直接结束
        if (kthNode == nullptr) {
            break;
        }

        // 保存下一组的第一个节点
        ListNode* groupNext = kthNode->next;

        // 当前这一组的第一个节点
        ListNode* current = groupPrev->next;

        // 当前组翻转后，尾部需要指向下一组
        ListNode* prev = groupNext;

        // 翻转当前这一组
        while (current != groupNext) {
            ListNode* nextNode = current->next;

            current->next = prev;

            prev = current;
            current = nextNode;
        }

        /*
            翻转前：

            groupPrev -> 原来的组头 -> ... -> kthNode -> groupNext

            翻转后：

            groupPrev -> kthNode -> ... -> 原来的组头 -> groupNext
        */

        // 保存原来的组头
        // 它在翻转之后会变成当前组的尾节点
        ListNode* oldGroupHead = groupPrev->next;

        // 将上一部分连接到当前组的新头节点
        groupPrev->next = kthNode;

        // 当前组原来的头节点，变成下一组的 groupPrev
        groupPrev = oldGroupHead;
    }

    return dummy.next;
}

// 输出链表
void printList(ListNode* head) {
    ListNode* current = head;

    while (current != nullptr) {
        cout << current->val;

        if (current->next != nullptr) {
            cout << " ";
        }

        current = current->next;
    }

    cout << "\n";
}

int main() {
    int n;
    int k;

    cin >> n >> k;

    vector<int> nums(n);

    for (int i = 0; i < n; ++i) {
        cin >> nums[i];
    }

    // 根据数组建立链表
    ListNode* head = buildList(nums);

    // K 个一组翻转
    head = reverseKGroup(head, k);

    // 输出结果
    printList(head);

    return 0;
}
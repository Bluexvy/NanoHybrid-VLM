#include<bits/stdc++.h>

using namespace std;

struct TreeNode{
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x){
        val = x;
        left = nullptr;
        right = nullptr;
    }
};

TreeNode* buildTree(vector<int>& nums, int index){
    if(index >= nums.size()) return nullptr;
    if(nums[index] == -1) return nullptr;

    TreeNode* root = new TreeNode(nums[index]);
    root->left = buildTree(nums, index * 2 + 1);
    root->right = buildTree(nums, index * 2 + 2);

    return root;
}

void inorderTraversal(TreeNode* root){
    if(root == nullptr) return;
    inorderTraversal(root->left);
    cout<< root->val<< " ";
    inorderTraversal(root->right);
}

int main(){
    vector<int> nums = {1,2,3,4,5,6,76,8,98,7678};
    TreeNode* root = buildTree(nums, 0);
    inorderTraversal(root);
    return 0;
}
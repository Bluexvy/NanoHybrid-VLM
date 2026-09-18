#include <bits/stdc++.h>
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

TreeNode* buildTree(){
    int value;
    cin >> value;

    if(value == -1) return nullptr;

    TreeNode* root = new TreeNode(value);
    root->left = buildTree();
    root->right = buildTree();
    return root;
}

void inorder(TreeNode* root){
    if(root == nullptr) return;
    inorder(root->left);
    cout << root->val << " ";
    inorder(root->right);
}

int main(){
    TreeNode* root = buildTree();
    inorder(root);
    return 0;
}
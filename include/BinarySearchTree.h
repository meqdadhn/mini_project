
#ifndef __BINARY_SEARCH__
#define __BINARY_SEARCH__

#include <iostream>

struct Node
{

  Node(int v)
  {
    val = v;
    left =nullptr;
    right = nullptr;
  }

  Node* left;
  Node* right;
  int val;
};



class BinarySearchTree
{
public:

  Node* Insert(Node* cur_node, int new_val)
  {
    if (!cur_node)
    {
      cur_node = new Node(new_val);
    }
    else
    {
      if (new_val < cur_node->val)
      {
        cur_node->left = Insert(cur_node->left, new_val);
      }
      else
      {
        cur_node->right = Insert(cur_node->right, new_val);
      }
    }

    return cur_node;
  }

  void Log(Node* root)

  {
    if (root)
    {
      Log(root->left);
      std::cout << root->val << std::endl;
      Log(root->right);
    }
  }


private:

};


#endif
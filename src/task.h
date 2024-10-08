#include <iostream>
#include <memory>


struct Node
{
  Node(int val)
  {
    value = val; 
    left = nullptr;
    right = nullptr;
  }
  int value;
  Node* left;
  Node* right;
};

class Entity
{

public:
  Entity()
  {
    std::cout << "CONSTRUCT ENTITY" << std::endl;
  }
  virtual ~Entity() 
  {
    std::cout << "Destruct Entity" << std::endl;
  }
  virtual std::string GetName() {return "Entity";}

};

class Player: public Entity
{
public:

  Player(std::string name):
        Entity(),
        name_(name) 
  {
    std::cout << "CONSTRUCT PLAYER:" << name_ << std::endl;
  }
  virtual ~Player() override 
  {
    std::cout << "Destruct Player: " << name_ << std::endl;
  }
  std::string GetName() override {return name_;}
  std::string name_;
  float x;
  float y;
};

class NPC: public Player
{
public:

  NPC():
    Player("Non Player Charachter")
  {
    std::cout << "CONSTRUCT NPC" << std::endl;
  }
  ~NPC() 
  {
    std::cout << "Destruct NPC" << std::endl;
  }
};

void LOG(Entity* e)
{
  std::cout << "LOG: " << e->GetName() << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////

class B;
class A
{
public:
  A()
  {
    std::cout << "Class A constructor" << std::endl;
  }
  ~A()
  {
    std::cout << "Class A Destruct" << std::endl;
  }
  std::shared_ptr<B> b_obj;
};

class B
{
public:
  B()
  {
    std::cout << "Class B constructor" << std::endl;
  }
  ~B()
  {
    std::cout << "Class B Destruct" << std::endl;
  }

  std::weak_ptr<A> a_obj;
};




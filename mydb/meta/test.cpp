#include "bptree.h"
#include <iostream>

int main() {
  BPlusTreeMap<int, int> bp;
  bp.Insert(1, 11);
  std::cout << bp.Find(1).second << std::endl;
  bp.Insert(2, 22);
  std::cout << bp.Find(2).second << std::endl;
  bp.Insert(3, 33);
  std::cout << bp.Find(3).second << std::endl;
  bp.Insert(4, 44);
  std::cout << bp.Find(4).second << std::endl;
  bp.Insert(5, 55);
  std::cout << bp.Find(5).second << std::endl;
  bp.Insert(6, 66);
  std::cout << bp.Find(6).second << std::endl;
  json j = bp.Serialize();
  std::cout << j << std::endl;

  BPlusTreeMap<int, int> bp_d;
  bp_d.Deserialize(j);
  std::cout << bp_d.Serialize() << std::endl;
}

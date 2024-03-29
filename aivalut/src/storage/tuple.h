#pragma once
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "../storage/base.h"
#include "../utils/bptree.h"
#include "data.pb.h"
#include "tran.h"

const int TupleSize = 128;

struct Item {
  ColType coltype;
  int num_value;
  float float_value;
  std::string str_value;

  Item(int num_value) : num_value(num_value), coltype(ColType::Int) {}
  Item(std::string str_value)
      : str_value(str_value), coltype(ColType::Varchar) {}
  Item(float float_value) : float_value(float_value), coltype(ColType::Float) {}
};

inline storage::Tuple *NewTuple(uint64_t minTxId,
                                const std::vector<Item> &values) {
  storage::Tuple *t = new storage::Tuple();
  t->set_mintxid(minTxId);
  t->set_maxtxid(minTxId);

  for (const auto &v : values) {
    storage::TupleData *td = t->add_data();
    if (v.coltype == ColType::Int) {
      td->set_type(storage::TupleData_Type_INT);
      td->set_toi(v.num_value);
    } else if (v.coltype == ColType::Varchar) {
      td->set_type(storage::TupleData_Type_STRING);
      td->set_tos(v.str_value);
    } else if (v.coltype == ColType::Float) {
      td->set_type(storage::TupleData_Type_FLOAT);
      td->set_tof(v.float_value);
    }
  }

  return t;
}

inline std::array<uint8_t, TupleSize> SerializeTuple(const storage::Tuple *t) {
  std::array<uint8_t, TupleSize> buffer{};

  if (t != nullptr) {
    std::string serializedData;
    if (!t->SerializeToString(&serializedData)) {
      try {
        throw std::runtime_error("Failed to serialize storage::Tuple");
      } catch (std::runtime_error e) {
        std::cerr << "runtime_error: " << e.what() << std::endl;
      }
    }
    const size_t dataSize =
        std::min(serializedData.size(), TupleSize - sizeof(size_t));
    std::memcpy(buffer.data() + sizeof(size_t), serializedData.data(),
                dataSize);
    *reinterpret_cast<size_t *>(buffer.data()) = dataSize;
    // std::memcpy(buffer.data(), serializedData.c_str(),
    // serializedData.size());
  } else {
    std::cerr << "Skip a null tuple when serializing a tuple\n";
  }
  return buffer;
}

inline storage::Tuple *
DeserializeTuple(const std::array<uint8_t, TupleSize> &buffer) {
  storage::Tuple *t = new storage::Tuple();

  size_t dataSize = *reinterpret_cast<const size_t *>(buffer.data());
  const char *serializedData =
      reinterpret_cast<const char *>(buffer.data() + sizeof(size_t));
  // std::string serializedData(reinterpret_cast<const char *>(buffer.data()),
  //                           buffer.size());
  if (!t->ParseFromArray(serializedData, dataSize)) {
    try {
      throw std::runtime_error("Failed to deserialize storage::Tuple");
    } catch (std::runtime_error e) {
      std::cerr << "runtime_error: " << e.what() << std::endl;
      return nullptr;
    }
  }
  return t;
}

inline bool TupleEqual(const storage::Tuple *tuple, int order, int n) {
  const storage::TupleData &tupleData = tuple->data()[order];
  if (tupleData.type() == storage::TupleData_Type_INT) {
    return tupleData.toi() == n;
  }
  return false;
}

inline bool TupleEqual(const storage::Tuple *tuple, int order, float v) {
  const storage::TupleData &tupleData = tuple->data()[order];
  if (tupleData.type() == storage::TupleData_Type_FLOAT) {
    return tupleData.tof() == v;
  }
  return false;
}

inline bool TupleEqual(const storage::Tuple *tuple, int order,
                       const std::string &s) {
  const storage::TupleData &tupleData = tuple->data()[order];
  if (tupleData.type() == storage::TupleData_Type_STRING) {
    return tupleData.tos() == s;
  }
  return false;
}

inline bool TupleGreaterEq(const storage::Tuple *tuple, int order, int n) {
  const storage::TupleData &tupleData = tuple->data()[order];
  if (tupleData.type() == storage::TupleData_Type_INT) {
    return tupleData.toi() >= n;
  }
  return false;
}

inline bool TupleGreaterEq(const storage::Tuple *tuple, int order, float v) {
  const storage::TupleData &tupleData = tuple->data()[order];
  if (tupleData.type() == storage::TupleData_Type_FLOAT) {
    return tupleData.tof() >= v;
  }
  return false;
}

inline bool TupleGreaterEq(const storage::Tuple *tuple, int order,
                           const std::string &s) {
  const storage::TupleData &tupleData = tuple->data()[order];
  if (tupleData.type() == storage::TupleData_Type_STRING) {
    return tupleData.tos() >= s;
  }
  return false;
}

inline bool TupleLessEq(const storage::Tuple *tuple, int order, int n) {
  const storage::TupleData &tupleData = tuple->data()[order];
  if (tupleData.type() == storage::TupleData_Type_INT) {
    return tupleData.toi() <= n;
  }
  return false;
}

inline bool TupleLessEq(const storage::Tuple *tuple, int order, float v) {
  const storage::TupleData &tupleData = tuple->data()[order];
  if (tupleData.type() == storage::TupleData_Type_FLOAT) {
    return tupleData.tof() <= v;
  }
  return false;
}

inline bool TupleLessEq(const storage::Tuple *tuple, int order,
                        const std::string &s) {
  const storage::TupleData &tupleData = tuple->data()[order];
  if (tupleData.type() == storage::TupleData_Type_STRING) {
    return tupleData.tos() <= s;
  }
  return false;
}

inline bool TupleIsUnused(const storage::Tuple *tuple) {
  return tuple->mintxid() == 0;
}

inline bool TupleCanSee(const storage::Tuple *tuple, const Transaction *tran) {
  if (tuple->mintxid() == tran->Txid()) {
    return true;
  }

  if (tuple->maxtxid() < tran->Txid()) {
    return false;
  }

  if (tuple->mintxid() > tran->Txid() &&
      tran->GetState() != TransactionState::Commited) {
    return false;
  }

  return true;
}

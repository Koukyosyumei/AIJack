#pragma once
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "../meta/btree.h"
#include "../meta/meta.h"
#include "data.pb.h"
#include "tran.h"

inline storage::Tuple *NewTuple(uint64_t minTxId,
                                const std::vector<int> &values) {
  storage::Tuple *t = new storage::Tuple();
  t->set_mintxid(minTxId);
  t->set_maxtxid(minTxId);

  for (const auto &v : values) {
    storage::TupleData *td = t->add_data();
    td->set_type(storage::TupleData_Type_INT);
    td->set_number(v);
  }

  return t;
}

inline storage::Tuple *NewTuple(uint64_t minTxId,
                                const std::vector<std::string> &values) {
  storage::Tuple *t = new storage::Tuple();
  t->set_mintxid(minTxId);
  t->set_maxtxid(minTxId);

  for (const auto &v : values) {
    storage::TupleData *td = t->add_data();

    td->set_type(storage::TupleData_Type_STRING);
    td->set_string(v);
  }

  return t;
}

inline bool TupleLess(const storage::Tuple *t1, const IntItem *item2) {
  if (t1 == nullptr || item2 == nullptr) {
    return false;
  }

  int32_t left = t1->data()[0].number();

  return left < item2->value;
}

inline std::array<uint8_t, 128> SerializeTuple(const storage::Tuple *t) {
  std::array<uint8_t, 128> buffer;

  storage::Tuple tuple;
  tuple.set_mintxid(t->mintxid());
  tuple.set_maxtxid(t->maxtxid());

  for (const auto &td : t->data()) {
    storage::TupleData *tupleData = tuple.add_data();

    if (td.type() == storage::TupleData_Type_INT) {
      tupleData->set_type(storage::TupleData_Type_INT);
      tupleData->set_number(td.number());
    } else if (td.type() == storage::TupleData_Type_STRING) {
      tupleData->set_type(storage::TupleData_Type_STRING);
      tupleData->set_string(td.string());
    } else {
      throw std::runtime_error("Invalid TupleData type");
    }
  }

  std::string serializedData = tuple.SerializeAsString();
  std::memcpy(buffer.data(), serializedData.c_str(), serializedData.size());

  return buffer;
}

inline storage::Tuple *
DeserializeTuple(const std::array<uint8_t, 128> &buffer) {
  storage::Tuple tuple;

  std::string serializedData(reinterpret_cast<const char *>(buffer.data()),
                             buffer.size());
  if (!tuple.ParseFromString(serializedData)) {
    throw std::runtime_error("Failed to deserialize Tuple");
  }

  storage::Tuple *t = new storage::Tuple();
  t->set_mintxid(tuple.mintxid());
  t->set_maxtxid(tuple.maxtxid());

  for (int i = 0; i < tuple.data_size(); i++) {
    const storage::TupleData &tupleData = tuple.data(i);
    storage::TupleData *td = t->add_data();

    if (tupleData.type() == storage::TupleData_Type_INT) {
      td->set_type(storage::TupleData_Type_INT);
      td->set_number(tupleData.number());
    } else if (tupleData.type() == storage::TupleData_Type_STRING) {
      td->set_type(storage::TupleData_Type_STRING);
      td->set_string(tupleData.string());
    } else {
      throw std::runtime_error("Invalid TupleData type");
    }
  }

  return t;
}

inline bool TupleEqual(const storage::Tuple *tuple, int order,
                       const std::string &s, int n) {
  const storage::TupleData &tupleData = tuple->data()[order];

  if (tupleData.type() == storage::TupleData_Type_STRING) {
    return tupleData.string() == s;
  } else if (tupleData.type() == storage::TupleData_Type_INT) {
    return tupleData.number() == n;
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

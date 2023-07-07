#pragma once
#include "tuple.h"
#include <array>
#include <cstring>
#include <iostream>

const int TupleNumber = 32;
const int PageSize = 4096;
const int TupleSize = 128;

class Page {
public:
  std::array<storage::Tuple, TupleNumber> Tuples;
};

inline Page *NewPage() {
  Page *page = new Page();
  return page;
}

inline uint64_t NewPgid(const std::string &tableName) {
  // FIXME: Implement the logic to generate a new Pgid
  return 0;
}

inline std::array<char, PageSize> SerializePage(const Page *p) {
  std::array<char, PageSize> buffer;

  for (int i = 0; i < TupleNumber; i++) {
    std::array<uint8_t, TupleSize> tupleBytes = SerializeTuple(&p->Tuples[i]);
    std::memcpy(buffer.data() + i * TupleSize, tupleBytes.data(), TupleSize);
  }

  return buffer;
}

inline Page *DeserializePage(const std::array<char, PageSize> &buffer) {
  Page *p = new Page();

  for (int i = 0; i < TupleNumber; i++) {
    std::array<uint8_t, TupleSize> tupleBytes;
    std::memcpy(tupleBytes.data(), buffer.data() + i * TupleSize, TupleSize);
    storage::Tuple *t = DeserializeTuple(tupleBytes);
    if (t == nullptr) {
      std::cerr << "Failed to deserialize tuple (offset=" << i << ")\n";
    } else {
      p->Tuples[i] = *t;
    }
  }

  return p;
}

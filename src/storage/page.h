#pragma once
#include "tuple.h"
#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>

const int TupleNumber = 32;
const int PageSize = 4096;

typedef std::pair<int, int> TID;

class Page {
public:
  std::array<storage::Tuple, TupleNumber> Tuples;
  int front;
};

inline Page *NewPage() {
  Page *page = new Page();
  page->front = 0;
  return page;
}

inline std::array<char, PageSize> SerializePage(const Page *p) {
  std::array<char, PageSize> buffer{};

  for (int i = 0; i < TupleNumber; i++) {
    std::array<uint8_t, TupleSize> tupleBytes = SerializeTuple(&p->Tuples[i]);
    std::memcpy(buffer.data() + i * TupleSize, tupleBytes.data(), TupleSize);
  }

  return buffer;
}

inline Page *DeserializePage(const std::array<char, PageSize> &buffer) {
  Page *p = new Page();
  bool front_set = false;
  for (int i = 0; i < TupleNumber; i++) {
    std::array<uint8_t, TupleSize> tupleBytes{};
    std::memcpy(tupleBytes.data(), buffer.data() + i * TupleSize, TupleSize);
    storage::Tuple *t = DeserializeTuple(tupleBytes);
    if (t == nullptr) {
      try {
        throw std::runtime_error(
            "Failed to deserialize tuple (offset=" + std::to_string(i) + ")");
      } catch (std::runtime_error &e) {
        std::cerr << "runtime_error: " << e.what() << std::endl;
      }
    } else {
      p->Tuples[i] = *t;
      if (!front_set && TupleIsUnused(t)) {
        p->front = i;
        front_set = true;
      }
    }
  }

  return p;
}

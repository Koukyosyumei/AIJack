#pragma once
#include <atomic>
#include <unordered_map>

enum class TransactionState { Commited = 1, InProgress, Abort };

class Transaction {
public:
  Transaction(uint64_t txid, TransactionState state)
      : txid(txid), state(state) {}

  uint64_t Txid() const { return txid; }

  TransactionState GetState() const { return state; }

  uint64_t txid;
  TransactionState state;
};

class TransactionManager {
public:
  TransactionManager() : currentTxid(0) {}

  Transaction *BeginTransaction() {
    uint64_t txid = newTxid();
    Transaction *tx = new Transaction(txid, TransactionState::InProgress);
    clogs[txid] = tx;
    return tx;
  }

  void Commit(Transaction *tran) {
    if (!tran) {
      tran->state = TransactionState::Commited;
    }
  }

  void Abort(Transaction *tran) {
    if (!tran) {
      tran->state = TransactionState::Abort;
    }
  }

  uint64_t GetCurrentTxID() { return currentTxid; }

  uint64_t newTxid() { return std::atomic_fetch_add(&currentTxid, 1) + 1; }

private:
  std::unordered_map<uint64_t, Transaction *> clogs;
  std::atomic<uint64_t> currentTxid;
};

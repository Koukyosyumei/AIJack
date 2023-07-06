#pragma once
#include <atomic>
#include <mutex>
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
    std::lock_guard<std::mutex> lock(clogsMutex);
    clogs[txid] = tx;
    return tx;
  }

  void Commit(Transaction *tran) { tran->state = TransactionState::Commited; }

  void Abort(Transaction *tran) { tran->state = TransactionState::Abort; }

  uint64_t GetCurrentTxID() { return currentTxid; }

  uint64_t newTxid() { return ++currentTxid; }

private:
  std::unordered_map<uint64_t, Transaction *> clogs;
  std::atomic<uint64_t> currentTxid;
  std::mutex clogsMutex;
};

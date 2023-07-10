#pragma once
#include "token.h"
#include <string>
#include <vector>

// Forward declarations
struct Stmt;
struct Expr;

// Stmt interface
struct Stmt {
  virtual void stmtNode() = 0;
};

// Statements
struct CreateTableStmt : public Stmt {
  std::string TableName;
  std::vector<std::string> ColNames;
  std::vector<std::string> ColTypes;
  std::string PrimaryKey;

  void stmtNode() override {}
};

struct InsertStmt : public Stmt {
  std::string TableName;
  std::vector<Expr *> Values;

  void stmtNode() override {}
};

struct SelectStmt : public Stmt {
  std::vector<std::string> ColNames;
  std::vector<std::string> From;
  std::vector<Expr *> Wheres;
  std::vector<std::pair<std::string, std::pair<std::string, std::string>>>
      Joins;

  void stmtNode() override {}
};

struct LogregStmt : public Stmt {
  std::string model_name;
  std::string index_col;
  std::string target_col;
  int num_iterations;
  float lr;
  SelectStmt *selectstmt;
  void stmtNode() override {}
};

struct ComplaintStmt : public Stmt {
  std::string complaint_name;
  int k;
  LogregStmt *logregstmt;
  void stmtNode() override {}
};

struct UpdateStmt : public Stmt {
  std::string TableName;
  std::vector<std::string> ColNames;
  std::vector<std::string *> Set;
  std::vector<Expr *> Where;

  void stmtNode() override {}
};

struct BeginStmt : public Stmt {
  void stmtNode() override {}
};

struct CommitStmt : public Stmt {
  void stmtNode() override {}
};

struct AbortStmt : public Stmt {
  void stmtNode() override {}
};

// Expr interface
struct Expr {
  Expr *left;
  Expr *right;
  std::string v;
  TokenKind op;
  bool is_primary;

  Expr() : is_primary(false){};
  Expr(std::string v) : is_primary(false), v(v) {}
  Expr(Expr *left, Expr *right) : left(left), right(right), is_primary(false) {}
  Expr(TokenKind op, Expr *left, Expr *right)
      : op(op), left(left), right(right), is_primary(false) {}
  bool IsLit() { return (left == nullptr) && (right == nullptr); }
};

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

  void stmtNode() override {}
};

struct UpdateStmt : public Stmt {
  std::string TableName;
  std::vector<std::string> ColNames;
  std::vector<std::string *>
      Set; // Assuming a placeholder for the actual type of set values
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

  Expr(){};
  Expr(std::string v) : v(v) {}
  Expr(Expr *left, Expr *right) : left(left), right(right) {}
  Expr(TokenKind op, Expr *left, Expr *right)
      : op(op), left(left), right(right) {}
  bool IsLit() { return (left == nullptr) && (right == nullptr); }
  // virtual void exprNode() = 0;
};

/*
// Expressions
struct Eq : public Expr {
  Expr *left;
  Expr *right;

  void exprNode() override {}
};

struct Lit : public Expr {
  std::string v;

  Lit() {}
  Lit(std::string v) : v(v) {}
  void exprNode() override {}
};
*/

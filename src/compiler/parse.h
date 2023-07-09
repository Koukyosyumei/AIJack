#include "ast.h"
#include "token.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct Parser {
  std::vector<Token *> tokens;
  int pos;
  std::vector<std::string> errors;

  Parser(std::vector<Token *> &tokens) : tokens(tokens), pos(0) {}

  bool isPosValid(TokenKind kind) {
    if (tokens.size() <= pos) {
      errors.emplace_back("Expected " + TokenKindToString(kind) +
                          " at position=" + std::to_string(pos) +
                          ", but the query is broken");
      return false;
    }
    return true;
  }

  bool isPosValid(std::vector<TokenKind> kinds) {
    std::string expectedKinds;
    for (int i = 0; i < kinds.size() - 1; i++) {
      expectedKinds += TokenKindToString(kinds[i]) + " or ";
    }
    expectedKinds += TokenKindToString(kinds[kinds.size() - 1]);
    if (tokens.size() <= pos) {
      errors.emplace_back("Expected " + expectedKinds + " at position=" +
                          std::to_string(pos) + ", but the query is broken");
      return false;
    }
    return true;
  }

  Token *expect(TokenKind kind) {
    if (!isPosValid(kind)) {
      return nullptr;
    }

    Token *token = tokens[pos];
    if (token->kind == kind) {
      pos++;
      return token;
    }

    errors.emplace_back("Expected " + TokenKindToString(kind) + ", but found " +
                        TokenKindToString(token->kind) +
                        " at position=" + std::to_string(pos));
    return nullptr;
  }

  Token *expectOr(const std::vector<TokenKind> &kinds) {
    if (!isPosValid(kinds)) {
      return nullptr;
    }

    Token *token = tokens[pos];
    for (TokenKind kind : kinds) {
      if (token->kind == kind) {
        pos++;
        return token;
      }
    }

    std::string expectedKinds;
    for (int i = 0; i < kinds.size() - 1; i++) {
      expectedKinds += TokenKindToString(kinds[i]) + " or ";
    }
    expectedKinds += TokenKindToString(kinds[kinds.size() - 1]);

    errors.emplace_back("Expected " + expectedKinds + ", but found " +
                        TokenKindToString(token->kind) +
                        " at position=" + std::to_string(pos));
    return nullptr;
  }

  bool consume(TokenKind kind) {
    if (pos < tokens.size() && tokens[pos]->kind == kind) {
      pos++;
      return true;
    }
    return false;
  }

  Expr *eq() {
    Expr *left = expr();

    if (consume(EQ)) {
      Expr *right = expr();
      return new Expr(EQ, left, right);
    }
    if (consume(GEQ)) {
      Expr *right = expr();
      return new Expr(GEQ, left, right);
    }
    return left;
  }

  Expr *expr() {
    if (!isPosValid({EQ, GEQ})) {
      return nullptr;
    }

    Token *token = tokens[pos];

    if (consume(NUMBER) || consume(STRING)) {
      return new Expr(token->str);
    }
    errors.emplace_back("Broken Expression at posittion=" +
                        std::to_string(pos));
    return nullptr;
  }

  std::vector<std::string> fromClause() {
    std::vector<std::string> tablenames;
    while (true) {
      Token *s = expect(STRING);
      if (s) {
        tablenames.emplace_back(s->str);
      }

      if (!consume(COMMA)) {
        break;
      }
    }
    return tablenames;
  }

  std::vector<std::pair<std::string, std::pair<std::string, std::string>>>
  joinClause() {
    Token *right = expect(STRING);
    consume(ON);
    Token *left_idx = expect(STRING);
    consume(EQ);
    Token *right_idx = expect(STRING);
    if ((right == nullptr) || (left_idx == nullptr) || (right_idx == nullptr)) {
      errors.emplace_back("Join clause is broken");
      return {};
    }
    return {std::make_pair(right->str,
                           std::make_pair(left_idx->str, right_idx->str))};
  }

  std::vector<Expr *> whereClause() {
    std::vector<Expr *> exprs;
    exprs.push_back(eq());
    return exprs;
  }

  SelectStmt *selectStmt() {
    SelectStmt *selectNode = new SelectStmt();

    while (true) {
      Token *tkn = expectOr({STAR, STRING});
      if (!tkn) {
        return nullptr;
      }
      selectNode->ColNames.push_back(tkn->str);

      if (!consume(COMMA)) {
        break;
      }
    }

    if (expect(FROM)) {
      std::vector<std::string> from = fromClause();
      if (from.size() == 0) {
        return nullptr;
      }
      selectNode->From = from;
    } else {
      return nullptr;
    }

    if (consume(JOIN)) {
      selectNode->Joins = joinClause();
    }

    if (consume(WHERE)) {
      selectNode->Wheres = whereClause();
    }

    return selectNode;
  }

  LogregStmt *logregStmt() {
    LogregStmt *logregNode = new LogregStmt();
    Token *modelName = expect(STRING);
    Token *indexName = expect(STRING);
    Token *targetName = expect(STRING);
    Token *numitr = expect(NUMBER);
    Token *lr = expect(NUMBER);
    if ((modelName == nullptr) || (indexName == nullptr) ||
        (targetName == nullptr) || (numitr == nullptr) || (lr == nullptr)) {
      return nullptr;
    }
    logregNode->model_name = modelName->str;
    logregNode->index_col = indexName->str;
    logregNode->target_col = targetName->str;
    logregNode->num_iterations = std::stoi(numitr->str);
    logregNode->lr = std::stof(lr->str);
    consume(FROM);
    if (!consume(SELECT)) {
      return nullptr;
    }
    logregNode->selectstmt = selectStmt();
    return logregNode;
  }

  UpdateStmt *updateTableStmt() {
    Token *tblName = expect(STRING);
    if (!tblName) {
      return nullptr;
    }
    expect(SET);

    std::vector<std::string> cols;
    std::vector<std::string *> sets;

    while (true) {
      Token *col = expect(STRING);
      if (!col) {
        return nullptr;
      }
      cols.push_back(col->str);

      expect(EQ);

      Token *set = expectOr({STRING, NUMBER});
      if (!set) {
        return nullptr;
      }
      sets.push_back(&set->str);

      if (!consume(COMMA)) {
        break;
      }
    }

    std::vector<Expr *> exprs;
    if (consume(WHERE)) {
      exprs = whereClause();
    }

    UpdateStmt *updateNode = new UpdateStmt();
    updateNode->TableName = tblName->str;
    updateNode->ColNames = cols;
    updateNode->Set = sets;
    updateNode->Where = exprs;

    return updateNode;
  }

  InsertStmt *insertTableStmt() {
    expect(INTO);
    Token *tblName = expect(STRING);
    if (!tblName) {
      return nullptr;
    }
    expect(VALUES);
    expect(LPAREN);

    std::vector<Expr *> exprs;
    while (true) {
      exprs.push_back(eq());
      if (consume(RPAREN)) {
        break;
      }
      expect(COMMA);
    }

    InsertStmt *insertNode = new InsertStmt();
    insertNode->TableName = tblName->str;
    insertNode->Values = exprs;

    return insertNode;
  }

  CreateTableStmt *createTableStmt() {
    expect(TABLE);
    Token *tblName = expect(STRING);
    expect(LBRACE);

    std::vector<std::string> colNames;
    std::vector<std::string> colTypes;
    std::string pk;

    while (true) {
      Token *colName = expect(STRING);
      if (colName == nullptr) {
        return nullptr;
      }
      colNames.push_back(colName->str);
      if (consume(INT)) {
        colTypes.push_back("int");
      } else if (consume(FLOAT)) {
        colTypes.push_back("float");
      } else if (consume(VARCHAR)) {
        colTypes.push_back("varchar");
      } else {
        errors.push_back("supported data type is not specified at position=" +
                         std::to_string(pos));
        return nullptr;
      }

      if (consume(PRIMARY)) {
        expect(KEY);
        pk = colName->str;
      }

      if (!consume(COMMA)) {
        break;
      }
    }

    expect(RBRACE);

    CreateTableStmt *createNode = new CreateTableStmt();
    createNode->TableName = tblName->str;
    createNode->ColNames = colNames;
    createNode->ColTypes = colTypes;
    createNode->PrimaryKey = pk;

    return createNode;
  }

  Stmt *Parse() {
    if (consume(CREATE)) {
      return createTableStmt();
    }

    if (consume(LOGREG)) {
      return logregStmt();
    }

    if (consume(SELECT)) {
      return selectStmt();
    }

    if (consume(INSERT)) {
      return insertTableStmt();
    }

    if (consume(UPDATE)) {
      return updateTableStmt();
    }

    if (consume(BEGIN)) {
      return new BeginStmt();
    }

    if (consume(COMMIT)) {
      return new CommitStmt();
    }

    if (consume(ROLLBACK)) {
      return new AbortStmt();
    }

    errors.push_back("Unexpected query");
    return nullptr;
  }
};

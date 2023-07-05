#include "ast.h"
#include "token.h"
#include <iostream>
#include <string>
#include <vector>

struct Parser {
  std::vector<Token *> tokens;
  int pos;
  std::vector<std::string> errors;

  Parser(std::vector<Token *> &tokens) : tokens(tokens), pos(0) {}

  Token *expect(TokenKind kind) {
    Token *token = tokens[pos];
    if (token->kind == kind) {
      pos++;
      return token;
    }

    errors.push_back("Expected " + TokenKindToString(kind) + ", but found " +
                     TokenKindToString(token->kind));
    std::cout << errors[errors.size() - 1] << std::endl;
    return nullptr;
  }

  Token *expectOr(const std::vector<TokenKind> &kinds) {
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

    errors.push_back("Expected " + expectedKinds + ", but found " +
                     TokenKindToString(token->kind));
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
      return new Expr(left, right);
    }
    return left;
  }

  Expr *expr() {
    Token *token = tokens[pos];

    if (consume(NUMBER) || consume(STRING)) {
      return new Expr(token->str);
    }

    errors.push_back("Expression failed");
    return nullptr;
  }

  std::vector<std::string> fromClause() {
    Token *s = expect(STRING);
    return std::vector<std::string>{s->str};
  }

  std::vector<Expr *> whereClause() {
    std::vector<Expr *> exprs;
    exprs.push_back(eq());
    return exprs;
  }

  SelectStmt *selectStmt() {
    Token *tkn = expectOr({STAR, STRING});

    SelectStmt *selectNode = new SelectStmt();
    selectNode->ColNames.push_back(tkn->str);

    if (consume(FROM)) {
      std::vector<std::string> from = fromClause();
      selectNode->From = from;
    }

    if (consume(WHERE)) {
      selectNode->Wheres = whereClause();
    }

    return selectNode;
  }

  UpdateStmt *updateTableStmt() {
    Token *tblName = expect(STRING);
    expect(SET);

    std::vector<std::string> cols;
    std::vector<std::string *> sets;

    while (true) {
      Token *col = expect(STRING);
      cols.push_back(col->str);

      expect(EQ);

      Token *set = expectOr({STRING, NUMBER});
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
    std::cout << 1 << std::endl;
    Token *tblName = expect(STRING);
    std::cout << 2 << std::endl;
    expect(LBRACE);
    std::cout << 3 << std::endl;

    std::vector<std::string> colNames;
    std::vector<std::string> colTypes;
    std::string pk;

    while (true) {
      std::cout << 4 << std::endl;
      Token *colName = expect(STRING);
      expect(INT);
      std::cout << 5 << std::endl;
      colNames.push_back(colName->str);
      colTypes.push_back("int");

      std::cout << 6 << std::endl;
      if (consume(PRIMARY)) {
        expect(KEY);
        std::cout << 7 << std::endl;
        pk = colName->str;
        std::cout << 8 << std::endl;
      }

      std::cout << 9 << std::endl;
      if (!consume(COMMA)) {
        break;
      }
      std::cout << 10 << std::endl;
    }

    std::cout << 11 << std::endl;
    expect(RBRACE);

    std::cout << 12 << std::endl;
    CreateTableStmt *createNode = new CreateTableStmt();
    createNode->TableName = tblName->str;
    createNode->ColNames = colNames;
    createNode->ColTypes = colTypes;
    createNode->PrimaryKey = pk;

    std::cout << 13 << std::endl;
    return createNode;
  }

  Stmt *Parse(std::vector<std::string> &parseErrors) {
    if (consume(CREATE)) {
      return createTableStmt();
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

    parseErrors.push_back("Unexpected query");
    return nullptr;
  }
};

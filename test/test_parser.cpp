#include "mydb/compiler/parse.h"
#include <gtest/gtest.h>
#include <vector>

TEST(ParserTest, ParseEq) {
  std::vector<Token *> tokens = {NewToken(TokenKind::STRING, "1"),
                                 NewToken(TokenKind::EQ),
                                 NewToken(TokenKind::STRING, "2")};

  Parser parser(tokens);
  Expr *node = parser.eq();
  ASSERT_EQ(node->left->v, "1");
  ASSERT_EQ(node->right->v, "2");
}

TEST(ParserTest, ParseCreateTable) {
  std::vector<Token *> tokens = {
      NewToken(TokenKind::CREATE),          NewToken(TokenKind::TABLE),
      NewToken(TokenKind::STRING, "users"), NewToken(TokenKind::LBRACE),
      NewToken(TokenKind::STRING, "id"),    NewToken(TokenKind::INT, "int"),
      NewToken(TokenKind::RBRACE)};

  Parser parser(tokens);
  CreateTableStmt *node = dynamic_cast<CreateTableStmt *>(parser.Parse());
  ASSERT_EQ(parser.errors.size(), 0);
  ASSERT_EQ(node->TableName, "users");
  ASSERT_EQ(node->ColNames[0], "id");
  ASSERT_EQ(node->ColTypes[0], "int");
}

TEST(ParserTest, ParseSelect) {
  std::vector<Token *> tokens = {NewToken(TokenKind::SELECT),
                                 NewToken(TokenKind::STRING, "id"),
                                 NewToken(TokenKind::FROM),
                                 NewToken(TokenKind::STRING, "users"),
                                 NewToken(TokenKind::WHERE, "where"),
                                 NewToken(TokenKind::STRING, "1"),
                                 NewToken(TokenKind::EQ),
                                 NewToken(TokenKind::STRING, "2")};

  Parser parser(tokens);
  SelectStmt *node = dynamic_cast<SelectStmt *>(parser.Parse());
  ASSERT_EQ(parser.errors.size(), 0);
  ASSERT_EQ(node->From[0], "users");
  ASSERT_EQ(node->ColNames[0], "id");
}

TEST(ParserTest, ParseInsert) {
  std::vector<Token *> tokens = {
      NewToken(TokenKind::INSERT),          NewToken(TokenKind::INTO),
      NewToken(TokenKind::STRING, "users"), NewToken(TokenKind::VALUES),
      NewToken(TokenKind::LPAREN),          NewToken(TokenKind::NUMBER, "1"),
      NewToken(TokenKind::RPAREN)};

  Parser parser(tokens);
  ASSERT_NO_THROW(parser.Parse());
}

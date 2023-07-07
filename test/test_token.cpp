#include "mydb/query/token.h"
#include <gtest/gtest.h>

TEST(TokenizeTest, CreateTableTokenize) {
  Tokenizer tokenizer("create table{int}");
  std::vector<Token *> tkns;
  ASSERT_NO_THROW(tkns = tokenizer.Tokenize());
  ASSERT_EQ(tkns.size(), 5);
  ASSERT_EQ(tkns[0]->kind, TokenKind::CREATE);
  ASSERT_EQ(tkns[1]->kind, TokenKind::TABLE);
  ASSERT_EQ(tkns[2]->kind, TokenKind::LBRACE);
  ASSERT_EQ(tkns[3]->kind, TokenKind::INT);
  ASSERT_EQ(tkns[4]->kind, TokenKind::RBRACE);
}

TEST(TokenizeTest, LitTokenize) {
  Tokenizer tokenizer("a def 1 123");
  std::vector<Token *> tkns;
  ASSERT_NO_THROW(tkns = tokenizer.Tokenize());
  ASSERT_EQ(tkns.size(), 4);
  ASSERT_EQ(tkns[0]->kind, TokenKind::STRING);
  ASSERT_EQ(tkns[0]->str, "a");
  ASSERT_EQ(tkns[1]->kind, TokenKind::STRING);
  ASSERT_EQ(tkns[1]->str, "def");
  ASSERT_EQ(tkns[2]->kind, TokenKind::NUMBER);
  ASSERT_EQ(tkns[2]->str, "1");
  ASSERT_EQ(tkns[3]->kind, TokenKind::NUMBER);
  ASSERT_EQ(tkns[3]->str, "123");
}

TEST(TokenizeTest, OperatorTokenize) {
  Tokenizer tokenizer("{},*()");
  std::vector<Token *> tkns;
  ASSERT_NO_THROW(tkns = tokenizer.Tokenize());
  ASSERT_EQ(tkns.size(), 6);
  ASSERT_EQ(tkns[0]->kind, TokenKind::LBRACE);
  ASSERT_EQ(tkns[1]->kind, TokenKind::RBRACE);
  ASSERT_EQ(tkns[2]->kind, TokenKind::COMMA);
  ASSERT_EQ(tkns[3]->kind, TokenKind::STAR);
  ASSERT_EQ(tkns[4]->kind, TokenKind::LPAREN);
  ASSERT_EQ(tkns[5]->kind, TokenKind::RPAREN);
}

TEST(TokenizeTest, KeywordTokenize) {
  Tokenizer tokenizer("create table select where from insert into values "
                      "update set begin commit rollback eq");
  std::vector<Token *> tkns;
  ASSERT_NO_THROW(tkns = tokenizer.Tokenize());
  ASSERT_EQ(tkns.size(), 14);
  ASSERT_EQ(tkns[0]->kind, TokenKind::CREATE);
  ASSERT_EQ(tkns[1]->kind, TokenKind::TABLE);
  ASSERT_EQ(tkns[2]->kind, TokenKind::SELECT);
  ASSERT_EQ(tkns[3]->kind, TokenKind::WHERE);
  ASSERT_EQ(tkns[4]->kind, TokenKind::FROM);
  ASSERT_EQ(tkns[5]->kind, TokenKind::INSERT);
  ASSERT_EQ(tkns[6]->kind, TokenKind::INTO);
  ASSERT_EQ(tkns[7]->kind, TokenKind::VALUES);
  ASSERT_EQ(tkns[8]->kind, TokenKind::UPDATE);
  ASSERT_EQ(tkns[9]->kind, TokenKind::SET);
  ASSERT_EQ(tkns[10]->kind, TokenKind::BEGIN);
  ASSERT_EQ(tkns[11]->kind, TokenKind::COMMIT);
  ASSERT_EQ(tkns[12]->kind, TokenKind::ROLLBACK);
  ASSERT_EQ(tkns[13]->kind, TokenKind::EQ);
}

TEST(TokenizeTest, UpperKeywordTokenize) {
  Tokenizer tokenizer("CREATE TABLE");
  std::vector<Token *> tkns;
  ASSERT_NO_THROW(tkns = tokenizer.Tokenize());
  ASSERT_EQ(tkns.size(), 2);
  ASSERT_EQ(tkns[0]->kind, TokenKind::CREATE);
  ASSERT_EQ(tkns[1]->kind, TokenKind::TABLE);
}

TEST(TokenizeTest, TokenKindToString) {
  ASSERT_EQ(TokenKindToString(TokenKind::CREATE), "Create");
}

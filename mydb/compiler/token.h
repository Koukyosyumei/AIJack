#pragma once
#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

enum TokenKind {
  ILLEGAL,
  EOF_TOKEN,

  lit_begin,
  STRING,
  NUMBER,
  lit_end,

  type_begin,
  INT,
  type_end,

  operator_begin,
  LBRACE,
  RBRACE,
  LPAREN,
  RPAREN,
  COMMA,
  STAR,
  EQ,
  GEQ,
  operator_end,

  keyword_begin,
  SELECT,
  FROM,
  WHERE,
  JOIN,
  CREATE,
  TABLE,
  INSERT,
  INTO,
  VALUES,
  UPDATE,
  SET,
  BEGIN,
  COMMIT,
  ROLLBACK,
  PRIMARY,
  KEY,
  keyword_end
};

const std::unordered_map<int, std::string> tokenmap = {
    {ILLEGAL, "Illegal"}, {EOF, "Eor"},         {STRING, "String"},
    {NUMBER, "Number"},   {INT, "Int"},         {LBRACE, "{"},
    {RBRACE, "}"},        {LPAREN, "("},        {RPAREN, ")"},
    {COMMA, ","},         {STAR, "*"},          {SELECT, "Select"},
    {FROM, "From"},       {WHERE, "Where"},     {JOIN, "Join"},
    {CREATE, "Create"},   {TABLE, "Table"},     {INSERT, "Insert"},
    {INTO, "Into"},       {VALUES, "Values"},   {UPDATE, "Update"},
    {SET, "Set"},         {BEGIN, "Begin"},     {COMMIT, "Commit"},
    {ROLLBACK, "Abort"},  {PRIMARY, "Primary"}, {KEY, "Key"},
    {EQ, "Eq"},           {GEQ, "Geq"}};

struct Token {
  TokenKind kind;
  std::string str;
};

inline Token *NewToken(TokenKind kind, std::string str = "") {
  return new Token{kind, str};
}

inline std::string TokenKindToString(TokenKind kind) {
  auto it = tokenmap.find(static_cast<int>(kind));
  if (it != tokenmap.end()) {
    return it->second;
  }

  return "Unknown";
}

class Tokenizer {
private:
  std::string input;
  size_t pos;

  bool isSpace() {
    return input[pos] == ' ' || input[pos] == '\n' || input[pos] == '\t';
  }

  void skipSpace() {
    while (isSpace()) {
      pos++;
    }
  }

  bool isEnd() { return pos >= input.length(); }

  bool isSeq() { return input[pos] == ','; }

  bool matchKeyword(const std::string &keyword) {
    bool ok = pos + keyword.length() <= input.length() &&
              std::equal(keyword.begin(), keyword.end(), input.begin() + pos,
                         [](char a, char b) {
                           return std::tolower(a) == std::tolower(b);
                         });

    if (ok) {
      pos += keyword.length();
    }
    return ok;
  }

  bool isAsciiChar() {
    return (input[pos] >= 'a' && input[pos] <= 'z') ||
           (input[pos] >= 'A' && input[pos] <= 'Z');
  }

  bool isNumber() { return input[pos] >= '0' && input[pos] <= '9'; }

  std::string scanNumber() {
    std::string out;
    while (!isEnd() && !isSpace() && isNumber()) {
      out.push_back(input[pos]);
      pos++;
    }
    return out;
  }

  std::string scanString() {
    std::string out;
    while (!isEnd() && !isSpace() && !isSeq()) {
      out.push_back(input[pos]);
      pos++;
    }
    return out;
  }

public:
  Tokenizer(const std::string &input) : input(input), pos(0) {}

  std::vector<Token *> Tokenize() {
    std::vector<Token *> tokens;
    for (pos = 0; pos < input.length();) {
      skipSpace();

      if (matchKeyword("create")) {
        tokens.push_back(NewToken(CREATE, ""));
        continue;
      }

      if (matchKeyword("table")) {
        tokens.push_back(NewToken(TABLE, ""));
        continue;
      }

      if (matchKeyword("insert")) {
        tokens.push_back(NewToken(INSERT, ""));
        continue;
      }

      if (matchKeyword("into")) {
        tokens.push_back(NewToken(INTO, ""));
        continue;
      }

      if (matchKeyword("values")) {
        tokens.push_back(NewToken(VALUES, ""));
        continue;
      }

      if (matchKeyword("int")) {
        tokens.push_back(NewToken(INT, ""));
        continue;
      }

      if (matchKeyword("select")) {
        tokens.push_back(NewToken(SELECT, ""));
        continue;
      }

      if (matchKeyword("from")) {
        tokens.push_back(NewToken(FROM, ""));
        continue;
      }

      if (matchKeyword("where")) {
        tokens.push_back(NewToken(WHERE, ""));
        continue;
      }

      if (matchKeyword("join")) {
        tokens.push_back(NewToken(JOIN, ""));
        continue;
      }

      if (matchKeyword("update")) {
        tokens.push_back(NewToken(UPDATE, ""));
        continue;
      }

      if (matchKeyword("set")) {
        tokens.push_back(NewToken(SET, ""));
        continue;
      }

      if (matchKeyword("begin")) {
        tokens.push_back(NewToken(BEGIN, ""));
        continue;
      }

      if (matchKeyword("commit")) {
        tokens.push_back(NewToken(COMMIT, ""));
        continue;
      }

      if (matchKeyword("rollback")) {
        tokens.push_back(NewToken(ROLLBACK, ""));
        continue;
      }

      if (matchKeyword("primary")) {
        tokens.push_back(NewToken(PRIMARY, ""));
        continue;
      }

      if (matchKeyword("key")) {
        tokens.push_back(NewToken(KEY, ""));
        continue;
      }

      if (matchKeyword("eq")) {
        tokens.push_back(NewToken(EQ, ""));
        continue;
      }

      if (matchKeyword("geq")) {
        tokens.push_back(NewToken(GEQ, ""));
        continue;
      }

      if (isNumber()) {
        std::string num = scanNumber();
        Token *tkn = NewToken(NUMBER, num);
        tokens.push_back(tkn);
        continue;
      }

      if (isAsciiChar()) {
        std::string ascii = scanString();
        Token *tkn = NewToken(STRING, ascii);
        tokens.push_back(tkn);
        continue;
      }

      switch (input[pos]) {
      case '{':
        tokens.push_back(NewToken(LBRACE, ""));
        break;
      case '}':
        tokens.push_back(NewToken(RBRACE, ""));
        break;
      case '(':
        tokens.push_back(NewToken(LPAREN, ""));
        break;
      case ')':
        tokens.push_back(NewToken(RPAREN, ""));
        break;
      case ',':
        tokens.push_back(NewToken(COMMA, ""));
        break;
      case '*':
        tokens.push_back(NewToken(STAR, "*"));
        break;
      default:
        // error
        break;
      }

      pos++;
    }
    return tokens;
  }
};

inline bool IsType(TokenKind kind) {
  return kind > type_begin && kind < type_end;
}

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "mydb/core/db.h"
#include "mydb/core/http.h"
#include "mydb/storage/tuple.h"

std::vector<std::string> readFileLines(const std::string &filename) {
  std::vector<std::string> lines;
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    return lines;
  }

  std::string line;
  while (std::getline(file, line)) {
    lines.push_back(line);
  }

  file.close();
  return lines;
}

void showTitle() {
  std::string title =
      "MyDB : A simple database implemented from scratch in C++.";
  std::cout << title << std::endl;
}

bool client_exit() {
  httplib::Client client("localhost", 32198);
  auto res = client.Get("/exit");
  if (res && res->status == 200) {
    // Request succeeded
    std::cout << "Server exit requested\n";
    return true;
  } else {
    std::cerr << "Error requesting server exit\n";
    return false;
  }
}

void client_query(std::string input) {
  std::string query = "/execute?query=" + input;
  httplib::Client client("localhost", 32198);
  auto res = client.Get(query.c_str());
  if (!res || res->status != 200) {
    std::cerr << "Error executing query\n";
  } else {
    std::cout << res->body;
  }
}

void client() {
  showTitle();
  std::string input;
  while (true) {
    std::cout << ">>";
    std::getline(std::cin, input);

    if (input.substr(0, 4) == "exit") {
      if (client_exit()) {
        break;
      }
    } else if (input.substr(0, 7) == "source ") {
      std::string filename = input.substr(7);
      std::vector<std::string> lines = readFileLines(filename);
      for (const std::string &line : lines) {
        client_query(line);
      }
    } else {
      client_query(input);
    }
  }
}

void server() {
  MyDb *db = NewMyDb();
  db->Init();

  std::shared_ptr<ApiServer> apiServer = std::make_shared<ApiServer>(db);
  apiServer->Host();
}

int main(int argc, char *argv[]) {
  if (argc > 1 && std::string(argv[1]) == "server") {
    server();
  } else {
    client();
  }

  return 0;
}

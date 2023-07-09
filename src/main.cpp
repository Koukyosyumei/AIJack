#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/db.h"
#include "core/http.h"
#include "storage/tuple.h"

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
  std::string title = "AIValut : A simple database for debugging ML.";
  std::cout << title << std::endl;
}

bool client_exit(std::string address = "localhost", int port = 8889) {
  httplib::Client client(address, port);
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

void client_query(std::string input, std::string address = "localhost",
                  int port = 8889) {
  std::string query = "/execute?query=" + input;
  httplib::Client client(address, port);
  auto res = client.Get(query.c_str());
  if (!res || res->status != 200) {
    std::cerr << "Error executing query\n";
  } else {
    std::cout << res->body;
  }
}

void client(std::string address = "localhost", int port = 8889) {
  showTitle();
  std::string input;
  while (true) {
    std::cout << ">>";
    std::getline(std::cin, input);

    if (input.substr(0, 4) == "exit") {
      if (client_exit(address, port)) {
        break;
      }
    } else if (input.substr(0, 7) == "source ") {
      std::string filename = input.substr(7);
      std::vector<std::string> lines = readFileLines(filename);
      for (const std::string &line : lines) {
        client_query(line, address, port);
      }
    } else {
      client_query(input, address, port);
    }
  }
}

void server(std::string home = ".db/") {
  MyDb *db = NewMyDb(home);
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

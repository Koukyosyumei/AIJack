#include <fstream>
//#include <log>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "mydb/db/db.h"
#include "mydb/db/http.h"
#include "mydb/storage/tuple.h"

void showTitle() {
  std::string title = "BogoDb : A toy database management system.";
  std::cout << title << std::endl;
}

void client() {
  showTitle();
  std::string input;
  while (true) {
    std::cout << ">>";
    std::getline(std::cin, input);

    std::string err;
    if (input.substr(0, 4) == "exit") {
      httplib::Client client("localhost", 32198);
      auto res = client.Get("/exit");
      if (res && res->status == 200) {
        // Request succeeded
        std::cout << "Server exit requested" << std::endl;
      } else {
        err = "Error requesting server exit";
      }
    } else {
      std::string query = "/execute?query=" + input;
      httplib::Client client("localhost", 32198);
      auto res = client.Get(query.c_str());
      if (!res || res->status != 200) {
        err = "Error executing query";
      } else {
        std::cout << res->body;
      }
    }

    if (!err.empty()) {
      std::cout << err << std::endl;
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

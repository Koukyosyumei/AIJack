#pragma once
#include <cstring>
#include <iostream>
// #include <log>
#include "db.h"
#include "httplib.h"
#include <string>
#include <unordered_map>
#include <vector>

class ApiServer {
private:
  MyDb *db;

public:
  ApiServer(MyDb *db) : db(db) {}

  void executeHandler(const httplib::Request &req, httplib::Response &res) {
    // logger->info("/execute requested");
    // logger->info(req.path);
    // logger->info(req.body);

    std::string query = req.get_param_value("query");
    if (query.empty()) {
      res.status = 400;
      res.set_content("Bad Request: Missing query parameter", "text/plain");
      return;
    }

    std::string userAgent = req.get_header_value("User-Agent");

    std::string result;
    std::string error;
    db->Execute(query, userAgent, result, error);

    if (!error.empty()) {
      res.status = 400;
      res.set_content(error, "text/plain");
      // logger->error(error);
      return;
    }

    res.set_content(result, "text/plain");
  }

  void exitHandler(const httplib::Request &req, httplib::Response &res) {
    // logger->info("/exit requested");
    db->Terminate();
  }

  void Host() {

    httplib::Server server;
    server.Get("/execute",
               [&](const httplib::Request &req, httplib::Response &res) {
                 executeHandler(req, res);
               });
    server.Get("/exit", [&](const httplib::Request &req,
                            httplib::Response &res) { exitHandler(req, res); });

    // log::info("Starting API server on port 32198...");
    server.listen("0.0.0.0", 32198);
  }
};

inline ApiServer *NewApiServer(MyDb *db) { return new ApiServer(db); }

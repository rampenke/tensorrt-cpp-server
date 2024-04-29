#include "crow.h"
#include <fstream>
#include <sstream>
#include  "mnist.h"


inline std::unique_ptr<std::vector<uint8_t>> parsePGMData(const std::string& pgm) {
    std::string magic, w, h, max;
    auto istrream = std::istringstream(pgm);
    istrream >> magic >> w >> h >> max;
    istrream.seekg(1, istrream.cur);
    auto data = std::make_unique<std::vector<uint8_t>>(std::stoi(w) * std::stoi(h));

    istrream.read(reinterpret_cast<char*>(data->data()), std::stoi(w) * std::stoi(h));
    return data;
}

crow::response handleUpload(MnistApi mnistApi, const crow::request& req) {
    crow::multipart::message file_message(req);
    for (const auto& part : file_message.part_map) {
        const auto& part_name = part.first;
        const auto& part_value = part.second;
        CROW_LOG_DEBUG << "Part: " << part_name;
        if ("file" == part_name) {
            // Extract the file name
            auto headers_it = part_value.headers.find("Content-Disposition");
            if (headers_it == part_value.headers.end()) {
                CROW_LOG_ERROR << "No Content-Disposition found";
                return crow::response(400);
            }
            auto params_it = headers_it->second.params.find("filename");
            if (params_it == headers_it->second.params.end()) {
                CROW_LOG_ERROR << "Part with name \"filename\" should have a file";
                return crow::response(400);
            }
            const std::string outfile_name = params_it->second;

            for (const auto& part_header : part_value.headers) {
                const auto& part_header_name = part_header.first;
                const auto& part_header_val = part_header.second;
                CROW_LOG_DEBUG << "Header: " << part_header_name << '=' << part_header_val.value;
                for (const auto& param : part_header_val.params) {
                    const auto& param_key = param.first;
                    const auto& param_val = param.second;
                    CROW_LOG_DEBUG << " Param: " << param_key << ',' << param_val;
                }
            }

            // Create a new file with the extracted file name and write file contents to it
            /* Test: Save to local file
            std::ofstream out_file(outfile_name);
            if (!out_file) {
                CROW_LOG_ERROR << " Write to file failed\n";
                continue;
            }
            out_file << part_value.body;
            out_file.close();
            CROW_LOG_INFO << " Contents written to " << outfile_name << '\n';
            */
            auto data = parsePGMData(part_value.body);
            auto result = mnistApi.infer(reinterpret_cast<char*>(data->data()));
            CROW_LOG_DEBUG << " Inference reuslt: " << result << '\n';
            //return crow::response(200);
            return crow::json::wvalue({
                {"Result", result}
            });

        } else {
            CROW_LOG_DEBUG << " Value: " << part_value.body << '\n';
        }
    }
    return crow::response(200);
}

int main() {
    crow::SimpleApp app;
    MnistApi mnistApi;
    if (!mnistApi.load()) {
        CROW_LOG_DEBUG << " Failed to load model " << '\n';
    };

    CROW_ROUTE(app, "/api/upload")
      .methods(crow::HTTPMethod::Post)([mnistApi](const crow::request& req) {
        return handleUpload(mnistApi, req);
      });

    // enables all log
    app.loglevel(crow::LogLevel::Debug);

    app.port(18080)
      .multithreaded()
      .run();

    return 0;
}
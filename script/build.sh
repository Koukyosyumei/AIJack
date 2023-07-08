protoc --proto_path=mydb/storage --cpp_out=mydb/storage mydb/storage/data.proto

cmake -S . -B build
cmake --build build

cp build/script/mydbapp mydbapp

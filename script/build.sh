protoc --proto_path=src/storage --cpp_out=src/storage src/storage/data.proto

cmake -S . -B build
cmake --build build

cp build/script/mydbapp mydbapp

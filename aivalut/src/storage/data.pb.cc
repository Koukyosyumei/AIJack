// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: data.proto

#include "data.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
extern PROTOBUF_INTERNAL_EXPORT_data_2eproto ::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_TupleData_data_2eproto;
namespace storage {
class TupleDataDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<TupleData> _instance;
} _TupleData_default_instance_;
class TupleDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<Tuple> _instance;
} _Tuple_default_instance_;
}  // namespace storage
static void InitDefaultsscc_info_Tuple_data_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::storage::_Tuple_default_instance_;
    new (ptr) ::storage::Tuple();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::storage::Tuple::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<1> scc_info_Tuple_data_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 1, 0, InitDefaultsscc_info_Tuple_data_2eproto}, {
      &scc_info_TupleData_data_2eproto.base,}};

static void InitDefaultsscc_info_TupleData_data_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::storage::_TupleData_default_instance_;
    new (ptr) ::storage::TupleData();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::storage::TupleData::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_TupleData_data_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 0, 0, InitDefaultsscc_info_TupleData_data_2eproto}, {}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_data_2eproto[2];
static const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* file_level_enum_descriptors_data_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_data_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_data_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::storage::TupleData, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::storage::TupleData, type_),
  PROTOBUF_FIELD_OFFSET(::storage::TupleData, toi_),
  PROTOBUF_FIELD_OFFSET(::storage::TupleData, tos_),
  PROTOBUF_FIELD_OFFSET(::storage::TupleData, tof_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::storage::Tuple, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::storage::Tuple, mintxid_),
  PROTOBUF_FIELD_OFFSET(::storage::Tuple, maxtxid_),
  PROTOBUF_FIELD_OFFSET(::storage::Tuple, data_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::storage::TupleData)},
  { 9, -1, sizeof(::storage::Tuple)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::storage::_TupleData_default_instance_),
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::storage::_Tuple_default_instance_),
};

const char descriptor_table_protodef_data_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\ndata.proto\022\007storage\"\201\001\n\tTupleData\022%\n\004t"
  "ype\030\003 \001(\0162\027.storage.TupleData.Type\022\013\n\003to"
  "i\030\004 \001(\005\022\013\n\003tos\030\005 \001(\t\022\013\n\003tof\030\006 \001(\002\"&\n\004Typ"
  "e\022\007\n\003INT\020\000\022\n\n\006STRING\020\001\022\t\n\005FLOAT\020\002\"K\n\005Tup"
  "le\022\017\n\007minTxId\030\001 \001(\004\022\017\n\007maxTxId\030\002 \001(\004\022 \n\004"
  "data\030\003 \003(\0132\022.storage.TupleDatab\006proto3"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_data_2eproto_deps[1] = {
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_data_2eproto_sccs[2] = {
  &scc_info_Tuple_data_2eproto.base,
  &scc_info_TupleData_data_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_data_2eproto_once;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_data_2eproto = {
  false, false, descriptor_table_protodef_data_2eproto, "data.proto", 238,
  &descriptor_table_data_2eproto_once, descriptor_table_data_2eproto_sccs, descriptor_table_data_2eproto_deps, 2, 0,
  schemas, file_default_instances, TableStruct_data_2eproto::offsets,
  file_level_metadata_data_2eproto, 2, file_level_enum_descriptors_data_2eproto, file_level_service_descriptors_data_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_data_2eproto = (static_cast<void>(::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_data_2eproto)), true);
namespace storage {
const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* TupleData_Type_descriptor() {
  ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&descriptor_table_data_2eproto);
  return file_level_enum_descriptors_data_2eproto[0];
}
bool TupleData_Type_IsValid(int value) {
  switch (value) {
    case 0:
    case 1:
    case 2:
      return true;
    default:
      return false;
  }
}

#if (__cplusplus < 201703) && (!defined(_MSC_VER) || _MSC_VER >= 1900)
constexpr TupleData_Type TupleData::INT;
constexpr TupleData_Type TupleData::STRING;
constexpr TupleData_Type TupleData::FLOAT;
constexpr TupleData_Type TupleData::Type_MIN;
constexpr TupleData_Type TupleData::Type_MAX;
constexpr int TupleData::Type_ARRAYSIZE;
#endif  // (__cplusplus < 201703) && (!defined(_MSC_VER) || _MSC_VER >= 1900)

// ===================================================================

void TupleData::InitAsDefaultInstance() {
}
class TupleData::_Internal {
 public:
};

TupleData::TupleData(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:storage.TupleData)
}
TupleData::TupleData(const TupleData& from)
  : ::PROTOBUF_NAMESPACE_ID::Message() {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  tos_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (!from._internal_tos().empty()) {
    tos_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), from._internal_tos(),
      GetArena());
  }
  ::memcpy(&type_, &from.type_,
    static_cast<size_t>(reinterpret_cast<char*>(&tof_) -
    reinterpret_cast<char*>(&type_)) + sizeof(tof_));
  // @@protoc_insertion_point(copy_constructor:storage.TupleData)
}

void TupleData::SharedCtor() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&scc_info_TupleData_data_2eproto.base);
  tos_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  ::memset(&type_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&tof_) -
      reinterpret_cast<char*>(&type_)) + sizeof(tof_));
}

TupleData::~TupleData() {
  // @@protoc_insertion_point(destructor:storage.TupleData)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void TupleData::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
  tos_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void TupleData::ArenaDtor(void* object) {
  TupleData* _this = reinterpret_cast< TupleData* >(object);
  (void)_this;
}
void TupleData::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void TupleData::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const TupleData& TupleData::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_TupleData_data_2eproto.base);
  return *internal_default_instance();
}


void TupleData::Clear() {
// @@protoc_insertion_point(message_clear_start:storage.TupleData)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  tos_.ClearToEmpty(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
  ::memset(&type_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&tof_) -
      reinterpret_cast<char*>(&type_)) + sizeof(tof_));
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* TupleData::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  ::PROTOBUF_NAMESPACE_ID::Arena* arena = GetArena(); (void)arena;
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // .storage.TupleData.Type type = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 24)) {
          ::PROTOBUF_NAMESPACE_ID::uint64 val = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
          _internal_set_type(static_cast<::storage::TupleData_Type>(val));
        } else goto handle_unusual;
        continue;
      // int32 toi = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 32)) {
          toi_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // string tos = 5;
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 42)) {
          auto str = _internal_mutable_tos();
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParser(str, ptr, ctx);
          CHK_(::PROTOBUF_NAMESPACE_ID::internal::VerifyUTF8(str, "storage.TupleData.tos"));
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // float tof = 6;
      case 6:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 53)) {
          tof_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag,
            _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
            ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* TupleData::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:storage.TupleData)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // .storage.TupleData.Type type = 3;
  if (this->type() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteEnumToArray(
      3, this->_internal_type(), target);
  }

  // int32 toi = 4;
  if (this->toi() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(4, this->_internal_toi(), target);
  }

  // string tos = 5;
  if (this->tos().size() > 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_tos().data(), static_cast<int>(this->_internal_tos().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "storage.TupleData.tos");
    target = stream->WriteStringMaybeAliased(
        5, this->_internal_tos(), target);
  }

  // float tof = 6;
  if (!(this->tof() <= 0 && this->tof() >= 0)) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(6, this->_internal_tof(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:storage.TupleData)
  return target;
}

size_t TupleData::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:storage.TupleData)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // string tos = 5;
  if (this->tos().size() > 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_tos());
  }

  // .storage.TupleData.Type type = 3;
  if (this->type() != 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::EnumSize(this->_internal_type());
  }

  // int32 toi = 4;
  if (this->toi() != 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
        this->_internal_toi());
  }

  // float tof = 6;
  if (!(this->tof() <= 0 && this->tof() >= 0)) {
    total_size += 1 + 4;
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void TupleData::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:storage.TupleData)
  GOOGLE_DCHECK_NE(&from, this);
  const TupleData* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<TupleData>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:storage.TupleData)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:storage.TupleData)
    MergeFrom(*source);
  }
}

void TupleData::MergeFrom(const TupleData& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:storage.TupleData)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.tos().size() > 0) {
    _internal_set_tos(from._internal_tos());
  }
  if (from.type() != 0) {
    _internal_set_type(from._internal_type());
  }
  if (from.toi() != 0) {
    _internal_set_toi(from._internal_toi());
  }
  if (!(from.tof() <= 0 && from.tof() >= 0)) {
    _internal_set_tof(from._internal_tof());
  }
}

void TupleData::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:storage.TupleData)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void TupleData::CopyFrom(const TupleData& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:storage.TupleData)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool TupleData::IsInitialized() const {
  return true;
}

void TupleData::InternalSwap(TupleData* other) {
  using std::swap;
  _internal_metadata_.Swap<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(&other->_internal_metadata_);
  tos_.Swap(&other->tos_, &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArena());
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(TupleData, tof_)
      + sizeof(TupleData::tof_)
      - PROTOBUF_FIELD_OFFSET(TupleData, type_)>(
          reinterpret_cast<char*>(&type_),
          reinterpret_cast<char*>(&other->type_));
}

::PROTOBUF_NAMESPACE_ID::Metadata TupleData::GetMetadata() const {
  return GetMetadataStatic();
}


// ===================================================================

void Tuple::InitAsDefaultInstance() {
}
class Tuple::_Internal {
 public:
};

Tuple::Tuple(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(arena),
  data_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:storage.Tuple)
}
Tuple::Tuple(const Tuple& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      data_(from.data_) {
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::memcpy(&mintxid_, &from.mintxid_,
    static_cast<size_t>(reinterpret_cast<char*>(&maxtxid_) -
    reinterpret_cast<char*>(&mintxid_)) + sizeof(maxtxid_));
  // @@protoc_insertion_point(copy_constructor:storage.Tuple)
}

void Tuple::SharedCtor() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&scc_info_Tuple_data_2eproto.base);
  ::memset(&mintxid_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&maxtxid_) -
      reinterpret_cast<char*>(&mintxid_)) + sizeof(maxtxid_));
}

Tuple::~Tuple() {
  // @@protoc_insertion_point(destructor:storage.Tuple)
  SharedDtor();
  _internal_metadata_.Delete<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

void Tuple::SharedDtor() {
  GOOGLE_DCHECK(GetArena() == nullptr);
}

void Tuple::ArenaDtor(void* object) {
  Tuple* _this = reinterpret_cast< Tuple* >(object);
  (void)_this;
}
void Tuple::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void Tuple::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const Tuple& Tuple::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_Tuple_data_2eproto.base);
  return *internal_default_instance();
}


void Tuple::Clear() {
// @@protoc_insertion_point(message_clear_start:storage.Tuple)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  data_.Clear();
  ::memset(&mintxid_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&maxtxid_) -
      reinterpret_cast<char*>(&mintxid_)) + sizeof(maxtxid_));
  _internal_metadata_.Clear<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>();
}

const char* Tuple::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  ::PROTOBUF_NAMESPACE_ID::Arena* arena = GetArena(); (void)arena;
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // uint64 minTxId = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          mintxid_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // uint64 maxTxId = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 16)) {
          maxtxid_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint64(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // repeated .storage.TupleData data = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 26)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(_internal_add_data(), ptr);
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::ExpectTag<26>(ptr));
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag,
            _internal_metadata_.mutable_unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(),
            ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* Tuple::_InternalSerialize(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:storage.Tuple)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // uint64 minTxId = 1;
  if (this->mintxid() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteUInt64ToArray(1, this->_internal_mintxid(), target);
  }

  // uint64 maxTxId = 2;
  if (this->maxtxid() != 0) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteUInt64ToArray(2, this->_internal_maxtxid(), target);
  }

  // repeated .storage.TupleData data = 3;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->_internal_data_size()); i < n; i++) {
    target = stream->EnsureSpace(target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessage(3, this->_internal_data(i), target, stream);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(::PROTOBUF_NAMESPACE_ID::UnknownFieldSet::default_instance), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:storage.Tuple)
  return target;
}

size_t Tuple::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:storage.Tuple)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated .storage.TupleData data = 3;
  total_size += 1UL * this->_internal_data_size();
  for (const auto& msg : this->data_) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(msg);
  }

  // uint64 minTxId = 1;
  if (this->mintxid() != 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::UInt64Size(
        this->_internal_mintxid());
  }

  // uint64 maxTxId = 2;
  if (this->maxtxid() != 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::UInt64Size(
        this->_internal_maxtxid());
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void Tuple::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:storage.Tuple)
  GOOGLE_DCHECK_NE(&from, this);
  const Tuple* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<Tuple>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:storage.Tuple)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:storage.Tuple)
    MergeFrom(*source);
  }
}

void Tuple::MergeFrom(const Tuple& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:storage.Tuple)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  data_.MergeFrom(from.data_);
  if (from.mintxid() != 0) {
    _internal_set_mintxid(from._internal_mintxid());
  }
  if (from.maxtxid() != 0) {
    _internal_set_maxtxid(from._internal_maxtxid());
  }
}

void Tuple::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:storage.Tuple)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Tuple::CopyFrom(const Tuple& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:storage.Tuple)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Tuple::IsInitialized() const {
  return true;
}

void Tuple::InternalSwap(Tuple* other) {
  using std::swap;
  _internal_metadata_.Swap<::PROTOBUF_NAMESPACE_ID::UnknownFieldSet>(&other->_internal_metadata_);
  data_.InternalSwap(&other->data_);
  ::PROTOBUF_NAMESPACE_ID::internal::memswap<
      PROTOBUF_FIELD_OFFSET(Tuple, maxtxid_)
      + sizeof(Tuple::maxtxid_)
      - PROTOBUF_FIELD_OFFSET(Tuple, mintxid_)>(
          reinterpret_cast<char*>(&mintxid_),
          reinterpret_cast<char*>(&other->mintxid_));
}

::PROTOBUF_NAMESPACE_ID::Metadata Tuple::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace storage
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::storage::TupleData* Arena::CreateMaybeMessage< ::storage::TupleData >(Arena* arena) {
  return Arena::CreateMessageInternal< ::storage::TupleData >(arena);
}
template<> PROTOBUF_NOINLINE ::storage::Tuple* Arena::CreateMaybeMessage< ::storage::Tuple >(Arena* arena) {
  return Arena::CreateMessageInternal< ::storage::Tuple >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>

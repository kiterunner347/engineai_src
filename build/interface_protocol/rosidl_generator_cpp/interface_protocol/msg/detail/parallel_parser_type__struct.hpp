// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from interface_protocol:msg/ParallelParserType.idl
// generated code does not contain a copyright notice

#ifndef INTERFACE_PROTOCOL__MSG__DETAIL__PARALLEL_PARSER_TYPE__STRUCT_HPP_
#define INTERFACE_PROTOCOL__MSG__DETAIL__PARALLEL_PARSER_TYPE__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__interface_protocol__msg__ParallelParserType __attribute__((deprecated))
#else
# define DEPRECATED__interface_protocol__msg__ParallelParserType __declspec(deprecated)
#endif

namespace interface_protocol
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct ParallelParserType_
{
  using Type = ParallelParserType_<ContainerAllocator>;

  explicit ParallelParserType_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  explicit ParallelParserType_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->structure_needs_at_least_one_member = 0;
    }
  }

  // field types and members
  using _structure_needs_at_least_one_member_type =
    uint8_t;
  _structure_needs_at_least_one_member_type structure_needs_at_least_one_member;


  // constant declarations
  static constexpr uint8_t CLASSIC_PARSER =
    0u;
  static constexpr uint8_t RL_PARSER =
    1u;

  // pointer types
  using RawPtr =
    interface_protocol::msg::ParallelParserType_<ContainerAllocator> *;
  using ConstRawPtr =
    const interface_protocol::msg::ParallelParserType_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<interface_protocol::msg::ParallelParserType_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<interface_protocol::msg::ParallelParserType_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      interface_protocol::msg::ParallelParserType_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<interface_protocol::msg::ParallelParserType_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      interface_protocol::msg::ParallelParserType_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<interface_protocol::msg::ParallelParserType_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<interface_protocol::msg::ParallelParserType_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<interface_protocol::msg::ParallelParserType_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__interface_protocol__msg__ParallelParserType
    std::shared_ptr<interface_protocol::msg::ParallelParserType_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__interface_protocol__msg__ParallelParserType
    std::shared_ptr<interface_protocol::msg::ParallelParserType_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const ParallelParserType_ & other) const
  {
    if (this->structure_needs_at_least_one_member != other.structure_needs_at_least_one_member) {
      return false;
    }
    return true;
  }
  bool operator!=(const ParallelParserType_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct ParallelParserType_

// alias to use template instance with default allocator
using ParallelParserType =
  interface_protocol::msg::ParallelParserType_<std::allocator<void>>;

// constant definitions
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t ParallelParserType_<ContainerAllocator>::CLASSIC_PARSER;
#endif  // __cplusplus < 201703L
#if __cplusplus < 201703L
// static constexpr member variable definitions are only needed in C++14 and below, deprecated in C++17
template<typename ContainerAllocator>
constexpr uint8_t ParallelParserType_<ContainerAllocator>::RL_PARSER;
#endif  // __cplusplus < 201703L

}  // namespace msg

}  // namespace interface_protocol

#endif  // INTERFACE_PROTOCOL__MSG__DETAIL__PARALLEL_PARSER_TYPE__STRUCT_HPP_

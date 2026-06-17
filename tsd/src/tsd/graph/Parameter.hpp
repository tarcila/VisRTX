// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tsd/core/Any.hpp"
#include "tsd/core/Token.hpp"
// std
#include <cstdint>
#include <string>
#include <vector>

namespace tsd::graph {

struct Parameter
{
  tsd::core::Token name;
  tsd::core::Any value;
};

// An ordered name->Any map owned by a Node. `hash()` feeds the evaluator's
// cache-validity check (a node recomputes when its param hash changes).
struct ParameterList
{
  template <typename T>
  void set(tsd::core::Token name, T v)
  {
    for (auto &p : m_params) {
      if (p.name == name) {
        p.value = tsd::core::Any(v);
        return;
      }
    }
    m_params.push_back(Parameter{name, tsd::core::Any(v)});
  }

  template <typename T>
  T get(tsd::core::Token name) const
  {
    for (const auto &p : m_params) {
      if (p.name == name)
        return p.value.get<T>();
    }
    return T{};
  }

  template <typename T>
  T getOr(tsd::core::Token name, const T &alt) const
  {
    for (const auto &p : m_params) {
      if (p.name == name)
        return p.value.getValueOr<T>(alt);
    }
    return alt;
  }

  bool has(tsd::core::Token name) const
  {
    for (const auto &p : m_params)
      if (p.name == name)
        return true;
    return false;
  }

  const std::vector<Parameter> &items() const
  {
    return m_params;
  }

  // Order-independent per-param mixing of name + type + the bytes the type
  // occupies (exact-bit for floats; the full string for strings).
  uint64_t hash() const
  {
    uint64_t h = 1469598103934665603ull; // FNV-1a offset basis
    auto mix = [&](const void *data, size_t n) {
      const auto *bytes = static_cast<const uint8_t *>(data);
      for (size_t i = 0; i < n; ++i) {
        h ^= bytes[i];
        h *= 1099511628211ull;
      }
    };
    for (const auto &p : m_params) {
      const void *namePtr = p.name.value();
      mix(&namePtr, sizeof(namePtr));
      if (!p.value.valid())
        continue;
      auto type = p.value.type();
      mix(&type, sizeof(type));
      if (type == ANARI_STRING) {
        const std::string s = p.value.getString();
        mix(s.data(), s.size());
      } else {
        // Hash exactly the bytes this ANARI type occupies — never the full
        // fixed-size Any storage, which would mix uninitialized padding and
        // (for strings) over-read the heap buffer.
        mix(p.value.data(), anari::sizeOf(type));
      }
    }
    return h;
  }

 private:
  std::vector<Parameter> m_params;
};

} // namespace tsd::graph

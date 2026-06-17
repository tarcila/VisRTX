// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/graph/TestBackend.hpp"
// std
#include <memory>
#include <vector>

namespace tsd::graph {

namespace {

size_t floatBufferBytes(const Value &v)
{
  auto b = std::static_pointer_cast<std::vector<float>>(v.payload);
  return b ? b->size() * sizeof(float) : 0;
}

Value copyRetag(const Value &src, const Residency &target)
{
  auto in = std::static_pointer_cast<std::vector<float>>(src.payload);
  auto out = std::make_shared<std::vector<float>>(*in); // real copy
  Value v = src;
  v.payload = out;
  v.residency = target;
  return v;
}

} // namespace

void registerTestBackendTransfers(TransferRegistry &reg)
{
  PortType arrayT{tsd::core::Token("array")};
  reg.registerTransfer(arrayT,
      tsd::core::Token("host"),
      tsd::core::Token("test"),
      copyRetag,
      floatBufferBytes);
  reg.registerTransfer(arrayT,
      tsd::core::Token("test"),
      tsd::core::Token("host"),
      copyRetag,
      floatBufferBytes);
}

} // namespace tsd::graph

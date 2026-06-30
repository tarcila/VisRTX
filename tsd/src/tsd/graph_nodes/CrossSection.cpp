// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <memory>
#include <mutex>
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
// viskores
#include <viskores/ImplicitFunction.h>
#include <viskores/cont/ArrayCopy.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/CellSetSingleType.h>
#include <viskores/cont/DataSet.h>
#include <viskores/cont/DataSetBuilderUniform.h>
#include <viskores/cont/Field.h>
#include <viskores/cont/Initialize.h>
#include <viskores/filter/contour/Slice.h>

namespace tsd::graph_nodes {
namespace {

using namespace tsd::graph;
using tsd::core::Token;
using float3 = tsd::core::math::float3;
using float4 = tsd::core::math::float4;
using uint3 = tsd::core::math::uint3;

void ensureViskoresInit()
{
  static std::once_flag once;
  std::call_once(once, [] { viskores::cont::Initialize(); });
}

// Look up a scalar in [valueRange.x, valueRange.y] against the TF colormap.
float4 sampleColormap(const TransferFunctionData &tf, float s)
{
  const size_t n = tf.colormap.size();
  if (n == 0)
    return float4(0.8f, 0.8f, 0.8f, 1.f);
  const float lo = tf.valueRange.x, hi = tf.valueRange.y;
  float t = (hi > lo) ? (s - lo) / (hi - lo) : 0.f;
  t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
  const size_t idx = size_t(t * float(n - 1) + 0.5f);
  return tf.colormap.get<float4>(idx < n ? idx : n - 1);
}

struct CrossSection : Node
{
  ParameterList params;
  CrossSection()
  {
    params.set(Token("origin"), float3(0.f, 0.f, 0.f));
    params.set(Token("normal"), float3(0.f, 0.f, 1.f));
    params.set(Token("color"), float3(0.8f, 0.8f, 0.8f)); // flat fallback
  }
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("CrossSection");
    i.category = Token("processor");
    i.inputs.push_back({Token("in"), PortType{portField()}, true, {}});
    // Optional: when connected, the slice is colored by the same TF as the
    // volume; otherwise it uses the flat `color` param.
    i.inputs.push_back({Token("tf"), PortType{portTF()}, false, {}});
    i.outputs.push_back({Token("out"), PortType{portSurface()}, true, {}});
    return i;
  }
  ParameterList &parameters() override
  {
    return params;
  }
  void evaluate(EvalContext &ctx) override
  {
    auto f = std::static_pointer_cast<Field>(
        ctx.input(Token("in"), hostResidency()).payload);
    if (!f) {
      ctx.fail("CrossSection: missing field input");
      return;
    }
    const uint3 dims = f->dims;
    const size_t n = size_t(dims.x) * dims.y * dims.z;
    if (dims.x == 0u || dims.y == 0u || dims.z == 0u || f->data.size() != n) {
      ctx.fail("CrossSection: field needs non-zero dims and matching data");
      return;
    }
    if (f->data.elementType() != ANARI_FLOAT32) {
      ctx.fail("CrossSection: field data must be ANARI_FLOAT32");
      return;
    }

    std::shared_ptr<TransferFunctionData> tf;
    if (ctx.hasInput(Token("tf")))
      tf = std::static_pointer_cast<TransferFunctionData>(
          ctx.input(Token("tf"), hostResidency()).payload);

    ensureViskoresInit();

    viskores::cont::DataSetBuilderUniform builder;
    auto ds = builder.Create(
        viskores::Id3(
            viskores::Id(dims.x), viskores::Id(dims.y), viskores::Id(dims.z)),
        viskores::Vec3f(f->origin.x, f->origin.y, f->origin.z),
        viskores::Vec3f(f->spacing.x, f->spacing.y, f->spacing.z));

    viskores::cont::ArrayHandle<viskores::Float32> scalars;
    scalars.Allocate(viskores::Id(n));
    {
      auto portal = scalars.WritePortal();
      for (size_t i = 0; i < n; ++i)
        portal.Set(viskores::Id(i), f->data.get<float>(i));
    }
    ds.AddField(viskores::cont::Field(
        "scalars", viskores::cont::Field::Association::Points, scalars));

    const float3 origin = params.getOr<float3>(Token("origin"), float3(0.f));
    const float3 normal =
        params.getOr<float3>(Token("normal"), float3(0.f, 0.f, 1.f));

    viskores::filter::contour::Slice slice;
    slice.SetActiveField("scalars");
    slice.SetImplicitFunction(
        viskores::Plane(viskores::Vec3f(origin.x, origin.y, origin.z),
            viskores::Vec3f(normal.x, normal.y, normal.z)));
    const viskores::cont::DataSet result = slice.Execute(ds);

    auto s = std::make_shared<SurfaceData>();
    s->geomSubtype = Token("triangle");

    if (result.GetNumberOfPoints() > 0) {
      viskores::cont::ArrayHandle<viskores::Vec3f> pts;
      viskores::cont::ArrayCopy(result.GetCoordinateSystem().GetData(), pts);
      const viskores::Id nPts = pts.GetNumberOfValues();
      tsd::core::AnyArray pos(ANARI_FLOAT32_VEC3, size_t(nPts));
      {
        auto pp = pts.ReadPortal();
        for (viskores::Id i = 0; i < nPts; ++i) {
          const auto v = pp.Get(i);
          pos.get<float3>(size_t(i)) = float3(v[0], v[1], v[2]);
        }
      }
      s->prim.arrays.push_back({Token("vertex.position"), pos});

      // Flat plane: one constant normal lights the cut surface.
      const float nlen = std::sqrt(
          normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
      const float3 nrm = nlen > 0.f
          ? float3(normal.x / nlen, normal.y / nlen, normal.z / nlen)
          : float3(0.f, 0.f, 1.f);
      tsd::core::AnyArray normals(ANARI_FLOAT32_VEC3, size_t(nPts));
      for (viskores::Id i = 0; i < nPts; ++i)
        normals.get<float3>(size_t(i)) = nrm;
      s->prim.arrays.push_back({Token("vertex.normal"), normals});

      viskores::cont::CellSetSingleType<> cells;
      result.GetCellSet().AsCellSet(cells);
      const auto conn =
          cells.GetConnectivityArray(viskores::TopologyElementTagCell{},
              viskores::TopologyElementTagPoint{});
      const viskores::Id nConn = conn.GetNumberOfValues();
      const size_t nTris = size_t(nConn / 3);
      tsd::core::AnyArray idx(ANARI_UINT32_VEC3, nTris);
      {
        auto cp = conn.ReadPortal();
        for (size_t t = 0; t < nTris; ++t)
          idx.get<uint3>(t) = uint3(uint32_t(cp.Get(viskores::Id(3 * t + 0))),
              uint32_t(cp.Get(viskores::Id(3 * t + 1))),
              uint32_t(cp.Get(viskores::Id(3 * t + 2))));
      }
      s->prim.arrays.push_back({Token("primitive.index"), idx});

      // Color by the shared TF: map each slice vertex's interpolated scalar
      // through the colormap into a per-vertex color.
      if (tf && result.HasPointField("scalars")) {
        viskores::cont::ArrayHandle<viskores::Float32> sv;
        viskores::cont::ArrayCopy(result.GetField("scalars").GetData(), sv);
        const viskores::Id nS = sv.GetNumberOfValues();
        tsd::core::AnyArray colors(ANARI_FLOAT32_VEC4, size_t(nPts));
        auto sp = sv.ReadPortal();
        for (viskores::Id i = 0; i < nPts && i < nS; ++i)
          colors.get<float4>(size_t(i)) = sampleColormap(*tf, sp.Get(i));
        s->prim.arrays.push_back({Token("vertex.color"), colors});
      }
    }

    s->appearance.scalars.push_back({Token("color"),
        tsd::core::Any(params.getOr<float3>(Token("color"), float3(0.8f)))});

    Value out;
    out.type = PortType{portSurface()};
    out.residency = hostResidency();
    out.payload = s;
    ctx.setOutput(Token("out"), out);
  }
};

} // namespace

void registerCrossSection(NodeRegistry &reg)
{
  reg.registerType(
      Token("CrossSection"), [] { return std::make_unique<CrossSection>(); });
}

} // namespace tsd::graph_nodes

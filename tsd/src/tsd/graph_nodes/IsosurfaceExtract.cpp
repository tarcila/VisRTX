// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include <memory>
#include <mutex>
#include "tsd/graph/Evaluator.hpp"
#include "tsd/graph_nodes/BuiltinNodes.hpp"
#include "tsd/graph_nodes/Descriptors.hpp"
// viskores
#include <viskores/cont/ArrayCopy.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/CellSetSingleType.h>
#include <viskores/cont/DataSet.h>
#include <viskores/cont/DataSetBuilderUniform.h>
#include <viskores/cont/Field.h>
#include <viskores/cont/Initialize.h>
#include <viskores/filter/contour/Contour.h>

namespace tsd::graph_nodes {
namespace {

using namespace tsd::graph;
using tsd::core::Token;
using float3 = tsd::core::math::float3;
using uint3 = tsd::core::math::uint3;

void ensureViskoresInit()
{
  static std::once_flag once;
  std::call_once(once, [] { viskores::cont::Initialize(); });
}

struct IsosurfaceExtract : Node
{
  ParameterList params;
  IsosurfaceExtract()
  {
    params.set(Token("isovalue"), 0.5f);
    params.set(Token("computeNormals"), true);
  }
  NodeTypeInfo typeInfo() const override
  {
    NodeTypeInfo i;
    i.name = Token("IsosurfaceExtract");
    i.category = Token("processor");
    i.inputs.push_back({Token("in"), PortType{portField()}, true, {}});
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
      ctx.fail("IsosurfaceExtract: missing field input");
      return;
    }
    const uint3 dims = f->dims;
    const size_t n = size_t(dims.x) * dims.y * dims.z;
    if (dims.x == 0u || dims.y == 0u || dims.z == 0u || f->data.size() != n) {
      ctx.fail("IsosurfaceExtract: field needs non-zero dims and matching data");
      return;
    }
    if (f->data.elementType() != ANARI_FLOAT32) {
      ctx.fail("IsosurfaceExtract: field data must be ANARI_FLOAT32");
      return;
    }

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

    const float isovalue = params.getOr<float>(Token("isovalue"), 0.5f);
    const bool computeNormals =
        params.getOr<bool>(Token("computeNormals"), true);
    viskores::filter::contour::Contour c;
    c.SetActiveField("scalars");
    c.SetIsoValue(viskores::Float64(isovalue));
    c.SetGenerateNormals(computeNormals);
    c.SetMergeDuplicatePoints(true);
    const viskores::cont::DataSet result = c.Execute(ds);

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

      viskores::cont::CellSetSingleType<> cells;
      result.GetCellSet().AsCellSet(cells);
      const auto conn = cells.GetConnectivityArray(
          viskores::TopologyElementTagCell{},
          viskores::TopologyElementTagPoint{});
      const viskores::Id nConn = conn.GetNumberOfValues();
      const size_t nTris = size_t(nConn / 3);
      tsd::core::AnyArray idx(ANARI_UINT32_VEC3, nTris);
      {
        auto cp = conn.ReadPortal();
        for (size_t t = 0; t < nTris; ++t)
          idx.get<uint3>(t) =
              uint3(uint32_t(cp.Get(viskores::Id(3 * t + 0))),
                  uint32_t(cp.Get(viskores::Id(3 * t + 1))),
                  uint32_t(cp.Get(viskores::Id(3 * t + 2))));
      }
      s->prim.arrays.push_back({Token("primitive.index"), idx});

      if (computeNormals && result.HasPointField("normals")) {
        viskores::cont::ArrayHandle<viskores::Vec3f> nrm;
        viskores::cont::ArrayCopy(result.GetField("normals").GetData(), nrm);
        const viskores::Id nN = nrm.GetNumberOfValues();
        tsd::core::AnyArray normals(ANARI_FLOAT32_VEC3, size_t(nN));
        auto np = nrm.ReadPortal();
        for (viskores::Id i = 0; i < nN; ++i) {
          const auto v = np.Get(i);
          normals.get<float3>(size_t(i)) = float3(v[0], v[1], v[2]);
        }
        s->prim.arrays.push_back({Token("vertex.normal"), normals});
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

void registerIsosurfaceExtract(NodeRegistry &reg)
{
  reg.registerType(Token("IsosurfaceExtract"),
      [] { return std::make_unique<IsosurfaceExtract>(); });
}

} // namespace tsd::graph_nodes

// Copyright 2026 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tsd/ui/imgui/windows/TFCurveEditor.hpp"
#include "tsd/graph_nodes/GraphEditModel.hpp"
#include "tsd/ui/imgui/Application.h"
// imgui
#include "imgui.h"
// std
#include <algorithm>
#include <vector>

namespace tsd::ui::imgui {

using tsd::core::ColorPoint;
using tsd::core::OpacityPoint;
using float3 = tsd::core::math::float3;
using float4 = tsd::core::math::float4;

namespace {

std::vector<ColorPoint> presetToColorPoints(const std::vector<float3> &rgb)
{
  std::vector<ColorPoint> pts;
  if (rgb.empty())
    return pts;
  const float denom = float(rgb.size() - 1);
  pts.reserve(rgb.size());
  for (size_t i = 0; i < rgb.size(); ++i)
    pts.push_back(ColorPoint(float(i) / denom, rgb[i].x, rgb[i].y, rgb[i].z));
  return pts;
}

// Returns the insertion index for a new point at position p (sorted by .x).
template <typename T>
int findIdx(const std::vector<T> &pts, float p)
{
  auto it =
      std::upper_bound(pts.begin(), pts.end(), p, [](float val, const T &b) {
        return val < b.x;
      });
  return int(std::distance(pts.begin(), it));
}

} // namespace

TFCurveEditor::TFCurveEditor(Application *app) : m_app(app) {}

TFCurveEditor::~TFCurveEditor()
{
  if (m_preview)
    SDL_DestroyTexture(m_preview);
}

// Build preview texture: RGBA32 (R=byte0, G=byte1, B=byte2, A=byte3).
// SDL_PIXELFORMAT_RGBA32 is endian-independent: byte order is always R,G,B,A.
// The uint32 value on a little-endian machine is: R|(G<<8)|(B<<16)|(A<<24).
void TFCurveEditor::refreshPreview(
    const tsd::core::TransferFunction &tf, int samples)
{
  const int w = std::max(2, samples);
  if (!m_preview || m_previewWidth != w) {
    if (m_preview)
      SDL_DestroyTexture(m_preview);
    m_preview = SDL_CreateTexture(m_app->sdlRenderer(),
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING,
        w,
        1);
    m_previewWidth = w;
  }
  auto cm = tsd::graph_nodes::GraphEditModel::sampleColormap(
      tf.colorPoints, tf.opacityPoints, w);
  std::vector<uint32_t> px(static_cast<size_t>(w));
  for (int i = 0; i < w; ++i) {
    const auto c = cm[size_t(i)];
    auto to8 = [](float f) {
      return uint32_t(std::clamp(f, 0.f, 1.f) * 255.f + 0.5f);
    };
    // RGBA32: byte0=R, byte1=G, byte2=B, byte3=A -> uint32 =
    // R|(G<<8)|(B<<16)|(A<<24)
    px[size_t(i)] =
        to8(c.x) | (to8(c.y) << 8) | (to8(c.z) << 16) | (to8(c.w) << 24);
  }
  SDL_UpdateTexture(m_preview, nullptr, px.data(), w * int(sizeof(uint32_t)));
}

void TFCurveEditor::drawPresetCombo(
    tsd::core::TransferFunction &tf, bool &changed)
{
  struct Preset
  {
    const char *name;
    const std::vector<float3> *rgb;
  };
  static const Preset presets[] = {
      {"viridis", &tsd::core::colormap::viridis},
      {"cool to warm", &tsd::core::colormap::cool_to_warm},
      {"jet", &tsd::core::colormap::jet},
      {"inferno", &tsd::core::colormap::inferno},
      {"grayscale", &tsd::core::colormap::grayscale},
  };
  if (ImGui::BeginCombo("Load preset", "presets")) {
    for (const auto &p : presets) {
      if (ImGui::Selectable(p.name)) {
        tf.colorPoints = presetToColorPoints(*p.rgb);
        changed = true;
      }
    }
    ImGui::EndCombo();
  }
}

// Port of TransferFunctionEditor::buildUI_drawEditor — retargeted to
// tf.opacityPoints. Keeps the InvisibleButton+IsItemActive+MouseDelta pattern.
void TFCurveEditor::drawOpacityCurve(
    tsd::core::TransferFunction &tf, bool &changed)
{
  ImGui::TextUnformatted("Opacity curve");

  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  const float canvas_x = ImGui::GetCursorScreenPos().x;
  float canvas_y = ImGui::GetCursorScreenPos().y;
  const float canvas_avail_x = ImGui::GetContentRegionAvail().x;
  const float mouse_x = ImGui::GetMousePos().x;
  const float mouse_y = ImGui::GetMousePos().y;
  const float scroll_x = ImGui::GetScrollX();
  const float scroll_y = ImGui::GetScrollY();
  const float margin = 10.f;
  const float width = canvas_avail_x - 2.f * margin;
  const float height = 120.f;
  const float opacity_len = 7.f;

  if (width <= 0.f || tf.opacityPoints.empty()) {
    ImGui::Dummy(ImVec2(canvas_avail_x, height));
    return;
  }

  // Draw filled rect background (visual only — interactive button comes after
  // the point loop so point buttons win the hover race).
  draw_list->AddRectFilled(ImVec2(canvas_x + margin, canvas_y),
      ImVec2(canvas_x + margin + width, canvas_y + height),
      0xFF303030);

  {
    std::vector<ImVec2> poly;
    poly.reserve(4);
    const auto &ops = tf.opacityPoints;
    for (int i = 0; i < int(ops.size()) - 1; ++i) {
      poly.clear();
      poly.emplace_back(
          canvas_x + margin + ops[i].x * width, canvas_y + height);
      poly.emplace_back(canvas_x + margin + ops[i].x * width,
          canvas_y + height - ops[i].y * height);
      poly.emplace_back(canvas_x + margin + ops[i + 1].x * width + 1.f,
          canvas_y + height - ops[i + 1].y * height);
      poly.emplace_back(
          canvas_x + margin + ops[i + 1].x * width + 1.f, canvas_y + height);
      draw_list->AddConvexPolyFilled(poly.data(), int(poly.size()), 0xc8d8d8d8);
    }
  }

  canvas_y += height + margin;

  // Draw and interact with each opacity control point (FIRST so point buttons
  // win the hover race against the background InvisibleButton drawn below).
  ImGui::SetCursorScreenPos(ImVec2(canvas_x, canvas_y));
  for (int i = 0; i < int(tf.opacityPoints.size()); ++i) {
    const ImVec2 pos(canvas_x + width * tf.opacityPoints[i].x + margin,
        canvas_y - height * tf.opacityPoints[i].y - margin);

    ImGui::SetCursorScreenPos(ImVec2(pos.x - opacity_len, pos.y - opacity_len));
    ImGui::InvisibleButton(("##op" + std::to_string(i)).c_str(),
        ImVec2(2.f * opacity_len, 2.f * opacity_len));
    ImGui::SetCursorScreenPos(ImVec2(canvas_x, canvas_y));

    // Visual: dark border -> light fill -> highlight on hover.
    draw_list->AddCircleFilled(pos, opacity_len, 0xFF565656);
    draw_list->AddCircleFilled(pos, 0.8f * opacity_len, 0xFFD8D8D8);
    draw_list->AddCircleFilled(pos,
        0.6f * opacity_len,
        ImGui::IsItemHovered() ? 0xFF051c33 : 0xFFD8D8D8);

    // Double right-click: delete interior points (guard endpoints).
    if (ImGui::IsMouseDoubleClicked(1) && ImGui::IsItemHovered()) {
      if (i > 0 && i < int(tf.opacityPoints.size()) - 1) {
        tf.opacityPoints.erase(tf.opacityPoints.begin() + i);
        changed = true;
        --i; // recheck index after erase
      }
    } else if (ImGui::IsItemActive()) {
      const ImVec2 delta = ImGui::GetIO().MouseDelta;
      tf.opacityPoints[i].y -= delta.y / height;
      tf.opacityPoints[i].y = std::clamp(tf.opacityPoints[i].y, 0.f, 1.f);
      // Allow horizontal drag only for interior points.
      if (i > 0 && i < int(tf.opacityPoints.size()) - 1) {
        tf.opacityPoints[i].x += delta.x / width;
        tf.opacityPoints[i].x = std::clamp(tf.opacityPoints[i].x,
            tf.opacityPoints[i - 1].x,
            tf.opacityPoints[i + 1].x);
      }
      changed = true;
    }
  }

  // Background button drawn AFTER the point loop — loses hover race to any
  // overlapping point button, which is the intended behavior.
  ImGui::SetCursorScreenPos(
      ImVec2(canvas_x + margin, canvas_y - height - margin));
  ImGui::InvisibleButton("##opacity_bg", ImVec2(width, height));

  // Double left-click on background to add a new opacity point.
  if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
    const float x =
        std::clamp((mouse_x - canvas_x - margin - scroll_x) / width, 0.f, 1.f);
    const float y = std::clamp(
        1.f - (mouse_y - canvas_y + margin - scroll_y) / height, 0.f, 1.f);
    const int idx = findIdx(tf.opacityPoints, x);
    tf.opacityPoints.insert(tf.opacityPoints.begin() + idx, OpacityPoint(x, y));
    changed = true;
  }
}

void TFCurveEditor::drawColorStops(
    tsd::core::TransferFunction &tf, bool &changed)
{
  ImGui::TextUnformatted("Color stops");
  bool needSort = false;
  for (int i = 0; i < int(tf.colorPoints.size()); ++i) {
    ImGui::PushID(i);
    auto &cp = tf.colorPoints[size_t(i)];
    float pos = cp.x;
    if (ImGui::SliderFloat("pos", &pos, 0.f, 1.f)) {
      cp.x = std::clamp(pos, 0.f, 1.f);
      needSort = true;
      changed = true;
    }
    float rgb[3] = {cp.y, cp.z, cp.w};
    if (ImGui::ColorEdit3("color", rgb)) {
      cp.y = rgb[0];
      cp.z = rgb[1];
      cp.w = rgb[2];
      changed = true;
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("X")) {
      tf.colorPoints.erase(tf.colorPoints.begin() + i);
      changed = true;
      ImGui::PopID();
      --i;
      continue;
    }
    ImGui::PopID();
  }
  if (ImGui::Button("Add stop")) {
    tf.colorPoints.push_back(ColorPoint(0.5f, 1.f, 1.f, 1.f));
    needSort = true;
    changed = true;
  }
  if (needSort) {
    std::sort(tf.colorPoints.begin(),
        tf.colorPoints.end(),
        [](const ColorPoint &a, const ColorPoint &b) { return a.x < b.x; });
  }
}

void TFCurveEditor::draw(
    tsd::core::TransferFunction &tf, int &samples, bool &changed)
{
  changed = false;
  drawPresetCombo(tf, changed);
  ImGui::Separator();
  drawColorStops(tf, changed);
  ImGui::Separator();
  drawOpacityCurve(tf, changed);
  ImGui::Separator();

  refreshPreview(tf, samples);
  if (m_preview) {
    ImGui::Image(reinterpret_cast<ImTextureID>(m_preview),
        ImVec2(float(m_previewWidth > 256 ? 256 : m_previewWidth), 24.f));
  }

  int s = samples;
  if (ImGui::InputInt(
          "Samples", &s, 1, 10, ImGuiInputTextFlags_EnterReturnsTrue)) {
    samples = std::max(2, s);
    changed = true;
  }
}

} // namespace tsd::ui::imgui

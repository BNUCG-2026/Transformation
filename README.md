# 计算机图形学实验2 - 3D 几何变换与渲染 (Taichi 版)

本项目是基于 **Taichi Lang** 实现的计算机图形学基础实验。通过高性能并行计算，实现了完整的 3D 渲染管线模拟，包括模型变换、视图变换、透视投影变换以及视口映射。

## ✨ 技术亮点
- **高性能内核**: 使用 Taichi 的 `@ti.kernel` 和 `@ti.func` 实现顶点变换的并行加速。
- **MVP 矩阵实现**: 手写实现了绕 Z 轴旋转矩阵、相机视图矩阵以及透视平截头体挤压（Perspective-to-Orthographic）投影矩阵。
- **现代包管理**: 采用 `uv` 工具链，秒级构建 Python 3.12 运行环境。

## 🛠️ 环境要求
- **Python**: 3.12+
- **包管理工具**: [uv](https://github.com/astral-sh/uv)
- **依赖库**: `taichi` (将通过 uv 自动安装)

## 📥 快速启动

### 1. 克隆项目
```bash
git clone https://github.com/BNUCG-2026/Transformation.git
cd Transformation
```

### 2. 同步并安装依赖
`uv` 会自动为你配置 Python 3.12 虚拟环境并安装 Taichi：
```bash
uv sync
```

### 3. 运行渲染器
```bash
uv run main.py
```

## 🎮 交互指南
程序启动后会弹出一个 700x700 的 GUI 窗口：
- **A 键**: 逆时针旋转三角形。
- **D 键**: 顺时针旋转三角形。
- **Esc 键**: 退出程序。

## 🔍 实现细节说明
项目模拟了标准图形管线的顶点处理阶段：
1. **Model Matrix**: 负责物体的局部旋转。
2. **View Matrix**: 将场景平移至以 `eye_pos` 为中心的相机空间。

3. **Projection Matrix**: 
   - 将透视平截头体挤压为长方体。
   - 通过正交变换缩放至 NDC 空间 (Standard Cube $[-1, 1]^3$)。

4. **Viewport Transform**: 将 NDC 坐标映射到屏幕坐标系 $[0, 1] \times [0, 1]$。

## 实验结果展示
![实验演示视频](https://github.com/BNUCG-2026/Transformation/blob/main/video/实验结果展示.gif)

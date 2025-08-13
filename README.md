# 编译器技术博客

一个专注于现代编译器技术的技术博客，涵盖 MLIR、LLVM、GPU 编译器、ASIC 编译器以及硬件架构等前沿技术领域。

## 🚀 项目简介

本博客致力于深入探讨现代编译器技术的理论与实践，为编译器开发者、硬件工程师和研究人员提供高质量的技术内容。我们专注于以下几个核心领域：

- **MLIR (Multi-Level Intermediate Representation)** - 多级中间表示技术
- **LLVM** - 编译器基础设施
- **GPU 编译器** - 并行计算编译技术
- **ASIC 编译器** - 专用芯片编译器
- **硬件架构** - 底层硬件设计与优化

## 📚 内容结构

### MLIR 技术
- MLIR 基本概念与设计理念
- Dialect 系统设计与实现
- Operations、Types & Attributes
- Regions & Blocks
- Pass 系统与优化框架

### LLVM 技术
- LLVM 架构概览
- LLVM IR 中间表示
- Pass 系统与优化技术
- 后端代码生成
- LLVM 工具链

### GPU 编译器
- CUDA 编译器技术
- OpenCL 跨平台编译
- ROCm AMD GPU 编译
- GPU 性能优化
- 内存管理与性能分析

### ASIC 编译器
- 高层次综合 (HLS)
- 领域特定架构 (DSA)
- 资源调度与优化
- 设计验证方法
- 实际项目案例分析

### 硬件架构
- GPU 架构分析
- AI 加速器 (TPU、NPU)
- FPGA 可重构计算
- 内存系统设计
- 片上网络与互连

## 🛠️ 技术栈

- **静态站点生成器**: Hugo
- **主题**: Hextra
- **部署**: GitHub Pages / Netlify
- **版本控制**: Git

## 🏃‍♂️ 快速开始

### 环境要求

- Go 1.19+
- Hugo Extended 0.112.0+
- Git

### 本地运行

1. 克隆仓库
```bash
git clone <repository-url>
cd compiler-blog
```

2. 安装依赖
```bash
go mod tidy
```

3. 启动开发服务器
```bash
hugo server -D
```

4. 访问 http://localhost:1313 查看博客

### 构建生产版本

```bash
hugo --minify
```

## 🎯 目标读者

- 编译器开发工程师
- 硬件架构师
- GPU/ASIC 开发者
- 高性能计算研究人员
- 计算机科学学生和研究者

## 📄 许可证

本项目采用 [MIT License](LICENSE) 许可证。

---

**让我们一起探索编译器技术的无限可能！** 🚀
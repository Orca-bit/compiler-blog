---
title: 硬件架构深入
weight: 5
next: hardware/introduction
---

# 硬件架构深入

深入理解目标硬件架构是编译器优化的基础。本章节详细分析GPU、TPU、FPGA等现代加速器的硬件架构，为编译器设计提供硬件层面的洞察。

## 📋 章节概览

{{< cards >}}
  {{< card link="hardware/introduction" title="硬件架构概述" icon="chip" subtitle="现代计算架构分类" >}}
  {{< card link="hardware/gpu-arch" title="GPU 架构分析" icon="chip" subtitle="并行计算架构详解" >}}
  {{< card link="hardware/ai-accelerators" title="AI 加速器" icon="chip" subtitle="TPU、NPU 架构分析" >}}
  {{< card link="hardware/fpga" title="FPGA 架构" icon="server" subtitle="可重构计算平台" >}}
  {{< card link="hardware/memory-systems" title="内存系统" icon="server" subtitle="存储层次与优化" >}}
  {{< card link="hardware/interconnect" title="互连网络" icon="globe" subtitle="片上网络与通信" >}}
{{< /cards >}}

## 🎯 学习目标

通过本章节的学习，你将掌握：

- **架构分类**: 理解不同类型处理器的架构特点
- **性能模型**: 建立硬件性能分析的理论基础
- **优化策略**: 针对特定硬件的编译器优化方法
- **设计权衡**: 理解硬件设计中的性能与功耗权衡
- **发展趋势**: 了解硬件架构的发展方向

## 🏗️ 现代计算架构分类

### 按并行度分类
```
计算架构
├── 标量处理器 (Scalar)
│   └── 传统 CPU 核心
├── 向量处理器 (Vector)
│   └── SIMD 单元
├── 并行处理器 (Parallel)
│   ├── 多核 CPU
│   ├── GPU (SIMT)
│   └── 众核处理器
└── 专用处理器 (Specialized)
    ├── DSP
    ├── AI 加速器
    └── ASIC
```

### 按存储访问模式分类
```
存储架构
├── 冯·诺依曼架构
│   └── 指令和数据共享存储
├── 哈佛架构
│   └── 指令和数据分离存储
├── 数据流架构
│   └── 数据驱动计算
└── 近存储计算
    └── Processing-in-Memory
```

## 🖥️ GPU 架构深入分析

### NVIDIA GPU 架构演进
```
架构演进时间线:
Fermi (2010) → Kepler (2012) → Maxwell (2014) → Pascal (2016) 
→ Volta (2017) → Turing (2018) → Ampere (2020) → Ada Lovelace (2022) → Hopper (2022)
```

### GPU 层次结构
```
GPU 芯片
├── GPC (Graphics Processing Cluster)
│   ├── TPC (Texture Processing Cluster)
│   │   └── SM (Streaming Multiprocessor)
│   │       ├── CUDA Cores
│   │       ├── Tensor Cores (Volta+)
│   │       ├── RT Cores (Turing+)
│   │       ├── Shared Memory
│   │       ├── Register File
│   │       └── Warp Scheduler
│   └── Raster Engine
├── Memory Controller
├── L2 Cache
└── NVLink/PCIe Interface
```

### SM (Streaming Multiprocessor) 详细结构
```
SM 内部组织 (Ampere GA102):
├── 128 CUDA Cores (FP32)
├── 64 CUDA Cores (INT32)
├── 4 Third-gen Tensor Cores
├── 1 RT Core (2nd gen)
├── 4 Warp Schedulers
├── 8 Dispatch Units
├── 128 KB Shared Memory/L1 Cache
├── 65536 x 32-bit Registers
└── 16 Load/Store Units
```

## 🧠 AI 加速器架构

### Google TPU 架构
```
TPU v4 架构:
├── Matrix Multiply Unit (MXU)
│   └── 128x128 systolic array
├── Vector Processing Unit (VPU)
│   ├── Vector ALUs
│   ├── Vector Registers
│   └── Scalar Unit
├── High Bandwidth Memory (HBM)
│   └── 32 GB HBM2
├── Interconnect
│   └── 2D Torus Network
└── Host Interface
    └── PCIe Gen4
```

### 脉动阵列 (Systolic Array) 原理
```
脉动阵列计算模式:

输入 A →  [PE] → [PE] → [PE] → 输出
         ↓     ↓     ↓
输入 B → [PE] → [PE] → [PE] → 输出
         ↓     ↓     ↓  
输入 B → [PE] → [PE] → [PE] → 输出
         ↓     ↓     ↓
        输出  输出  输出

PE (Processing Element):
├── 乘法器
├── 加法器
├── 累加器
└── 寄存器
```

### 华为昇腾 NPU 架构
```
昇腾 910 架构:
├── AI Core
│   ├── Cube Unit (矩阵计算)
│   ├── Vector Unit (向量计算)
│   ├── Scalar Unit (标量计算)
│   └── Local Memory
├── AI CPU
│   └── ARM Cortex-A55
├── DVPP (Digital Vision Pre-Processing)
├── HBM2e Memory
└── PCIe 4.0 Interface
```

## 🔧 FPGA 架构分析

### Xilinx FPGA 架构
```
Xilinx Versal ACAP:
├── Programmable Logic (PL)
│   ├── CLB (Configurable Logic Block)
│   ├── DSP Slices
│   ├── Block RAM
│   └── UltraRAM
├── Processing System (PS)
│   ├── ARM Cortex-A72 (Dual-core)
│   ├── ARM Cortex-R5F (Dual-core)
│   └── DDR4/LPDDR4 Controller
├── AI Engine Array
│   ├── AI Engine Tiles
│   ├── Memory Tiles
│   └── Interface Tiles
└── Hardened IP
    ├── PCIe Gen4
    ├── Ethernet
    └── Interlaken
```

### CLB (Configurable Logic Block) 结构
```
CLB 内部组织:
├── 8 x LUT6 (6-input Look-Up Table)
├── 16 x Flip-Flops
├── 2 x Distributed RAM
├── 2 x SRL (Shift Register LUT)
├── Carry Logic
└── Multiplexers
```

### AI Engine 架构
```
AI Engine Tile:
├── VLIW Vector Processor
│   ├── Vector ALU
│   ├── Scalar ALU
│   └── Load/Store Unit
├── Program Memory (16 KB)
├── Data Memory (32 KB)
├── AXI4-Stream Interface
└── Cascade Connections
```

## 💾 内存系统架构

### GPU 内存层次
```
GPU 内存层次 (延迟/带宽):

寄存器 (Registers)
├── 延迟: 1 cycle
├── 带宽: ~20 TB/s
└── 容量: 64KB per SM

共享内存 (Shared Memory)
├── 延迟: ~20 cycles
├── 带宽: ~19 TB/s
└── 容量: 128KB per SM

L1 缓存
├── 延迟: ~80 cycles
├── 带宽: ~9 TB/s
└── 容量: 128KB per SM

L2 缓存
├── 延迟: ~200 cycles
├── 带宽: ~7 TB/s
└── 容量: 40MB (A100)

全局内存 (HBM2e)
├── 延迟: ~300 cycles
├── 带宽: ~2 TB/s
└── 容量: 80GB (A100)
```

### HBM (High Bandwidth Memory) 架构
```
HBM2e 堆栈结构:
├── Logic Die (底层)
│   ├── PHY Interface
│   ├── Memory Controller
│   └── TSV (Through-Silicon Via)
├── DRAM Dies (8层)
│   ├── Bank Groups
│   ├── Banks
│   └── Subarrays
└── Microbumps
    └── 1024-bit Interface

性能特性:
├── 带宽: 460 GB/s per stack
├── 容量: 16/24/32 GB per stack
├── 功耗: ~5W per stack
└── 延迟: ~100ns
```

## 🌐 片上网络 (NoC) 架构

### 2D Mesh 网络
```
2D Mesh 拓扑:
[R]─[R]─[R]─[R]
 │   │   │   │
[R]─[R]─[R]─[R]
 │   │   │   │  
[R]─[R]─[R]─[R]
 │   │   │   │
[R]─[R]─[R]─[R]

R = Router
每个 Router 连接:
├── 本地处理单元
├── 4个相邻 Router
└── 路由逻辑
```

### Torus 网络
```
Torus 拓扑 (环形连接):
[R]─[R]─[R]─[R]
 │ ╲ │ ╱ │ ╲ │
[R]─[R]─[R]─[R]
 │ ╱ │ ╲ │ ╱ │
[R]─[R]─[R]─[R]
 │ ╲ │ ╱ │ ╲ │
[R]─[R]─[R]─[R]

优势:
├── 更短的平均路径
├── 更好的负载均衡
└── 更高的容错性
```

## 📊 性能建模

### Roofline 模型
```
性能 (FLOPS) = min(
    Peak Performance,
    Arithmetic Intensity × Memory Bandwidth
)

其中:
- Arithmetic Intensity = FLOPs / Bytes
- Memory Bandwidth = 内存带宽
- Peak Performance = 峰值计算性能
```

### GPU 性能分析
```cpp
// GPU 性能计算示例
struct GPUPerf {
    float compute_throughput;  // TFLOPS
    float memory_bandwidth;    // TB/s
    int sm_count;             // SM 数量
    int cores_per_sm;         // 每个 SM 的核心数
    float base_clock;         // 基础时钟频率 (GHz)
    float boost_clock;        // 加速时钟频率 (GHz)
};

// A100 规格
GPUPerf a100 = {
    .compute_throughput = 19.5,  // FP32
    .memory_bandwidth = 2.0,
    .sm_count = 108,
    .cores_per_sm = 64,
    .base_clock = 1.41,
    .boost_clock = 1.73
};

// 理论峰值性能计算
float peak_flops = a100.sm_count * a100.cores_per_sm * 
                   a100.boost_clock * 2; // 2 ops per cycle
```

## 🔮 未来发展趋势

### 新兴架构
```
发展方向:
├── 近存储计算 (Near-Data Computing)
│   ├── Processing-in-Memory (PIM)
│   ├── Computational Storage
│   └── Smart Memory
├── 神经形态计算 (Neuromorphic)
│   ├── Spiking Neural Networks
│   ├── Memristor Arrays
│   └── Event-driven Processing
├── 量子计算 (Quantum)
│   ├── 量子比特
│   ├── 量子门
│   └── 量子纠错
└── 光学计算 (Optical)
    ├── 光子处理器
    ├── 光学互连
    └── 光电混合
```

### 技术挑战
```
主要挑战:
├── 功耗墙 (Power Wall)
├── 内存墙 (Memory Wall)
├── 指令级并行墙 (ILP Wall)
├── 可靠性挑战
├── 设计复杂度
└── 制造成本
```

## 🔗 相关资源

- [NVIDIA GPU 架构白皮书](https://developer.nvidia.com/gpu-architecture)
- [AMD GPU 架构文档](https://www.amd.com/en/graphics/rdna-architecture)
- [Intel GPU 架构指南](https://www.intel.com/content/www/us/en/docs/graphics-for-linux/developer-guide/current/overview.html)
- [Google TPU 论文集](https://cloud.google.com/tpu/docs/system-architecture)
- [Xilinx FPGA 架构手册](https://www.xilinx.com/support/documentation/user_guides/ug574-ultrascale-clb.pdf)
- [ARM 处理器架构参考](https://developer.arm.com/documentation/ddi0487/latest/)
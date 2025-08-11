---
title: LLVM 编译器基础设施
weight: 2
next: llvm/introduction
---

# LLVM (Low Level Virtual Machine)

LLVM 是一个模块化和可重用的编译器和工具链技术集合。它提供了现代化的编译器基础设施，支持静态和动态编译，广泛应用于各种编程语言和硬件平台。

## 📋 章节概览

{{< cards >}}
  {{< card link="llvm/introduction" title="LLVM 简介" icon="book-open" subtitle="编译器基础设施概述" >}}
  {{< card link="llvm/ir" title="LLVM IR" icon="code" subtitle="中间表示语言" >}}
  {{< card link="llvm/passes" title="Pass 系统" icon="arrow-right" subtitle="优化与变换框架" >}}
  {{< card link="llvm/backend" title="后端代码生成" icon="cog" subtitle="目标代码生成" >}}
  {{< card link="llvm/optimization" title="优化技术" icon="fire" subtitle="编译器优化策略" >}}
  {{< card link="llvm/tools" title="工具链" icon="cog" subtitle="LLVM 工具集" >}}
{{< /cards >}}

## 🎯 学习目标

通过本章节的学习，你将掌握：

- **LLVM 架构**: 理解 LLVM 的整体架构和设计原则
- **LLVM IR**: 熟练读写 LLVM 中间表示
- **Pass 开发**: 编写自定义的分析和变换 Pass
- **后端开发**: 了解目标代码生成的基本原理
- **优化技术**: 掌握常见的编译器优化方法

## 🏗️ LLVM 架构概览

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Optimizer      │    │    Backend      │
│                 │    │                  │    │                 │
│ C/C++   → IR    │───▶│ IR → IR (Opts)   │───▶│ IR → Machine    │
│ Rust    → IR    │    │                  │    │     Code        │
│ Swift   → IR    │    │ Pass Manager     │    │                 │
│ ...     → IR    │    │                  │    │ x86/ARM/GPU/... │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔧 环境搭建

### 从源码构建 LLVM
```bash
# 克隆 LLVM 项目
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

# 创建构建目录
mkdir build && cd build

# 配置构建
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="clang;lld" \
  -DLLVM_TARGETS_TO_BUILD="X86;ARM;AArch64;NVPTX;AMDGPU" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON

# 编译
ninja
```

### 验证安装
```bash
# 检查版本
./bin/llvm-config --version
./bin/clang --version

# 运行测试
ninja check-llvm
```

## 📝 第一个 LLVM IR 程序

### C 源码
```c
// hello.c
#include <stdio.h>

int main() {
    printf("Hello, LLVM!\n");
    return 0;
}
```

### 生成 LLVM IR
```bash
# 生成 LLVM IR
clang -S -emit-llvm hello.c -o hello.ll

# 查看生成的 IR
cat hello.ll
```

### LLVM IR 示例
```llvm
; hello.ll
@.str = private unnamed_addr constant [13 x i8] c"Hello, LLVM!\00"

declare i32 @printf(i8*, ...)

define i32 @main() {
entry:
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds 
    ([13 x i8], [13 x i8]* @.str, i32 0, i32 0))
  ret i32 0
}
```

## 🚀 核心组件

### 1. LLVM Core
- **Module**: 编译单元的顶层容器
- **Function**: 函数定义和声明
- **BasicBlock**: 基本块，控制流的基本单位
- **Instruction**: 指令，计算的基本单位

### 2. Pass Manager
- **ModulePass**: 模块级别的 Pass
- **FunctionPass**: 函数级别的 Pass  
- **BasicBlockPass**: 基本块级别的 Pass
- **LoopPass**: 循环级别的 Pass

### 3. 代码生成器
- **SelectionDAG**: 指令选择的中间表示
- **MachineFunction**: 机器级别的函数表示
- **Register Allocation**: 寄存器分配
- **Instruction Scheduling**: 指令调度

## 🎨 优化示例

### 死代码消除 (DCE)
```llvm
; 优化前
define i32 @example() {
  %1 = add i32 1, 2      ; 死代码
  %2 = add i32 3, 4
  ret i32 %2
}

; 优化后
define i32 @example() {
  ret i32 7              ; 常量折叠 + DCE
}
```

### 循环优化
```llvm
; 循环展开前
for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %sum = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %add = add i32 %sum, %i
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %inc, 4
  br i1 %cmp, label %for.body, label %for.end
```

## 🔗 相关资源

- [LLVM 官方文档](https://llvm.org/docs/)
- [LLVM IR 语言参考](https://llvm.org/docs/LangRef.html)
- [LLVM Pass 编写指南](https://llvm.org/docs/WritingAnLLVMPass.html)
- [LLVM 代码生成器](https://llvm.org/docs/CodeGenerator.html)
- [LLVM 优化指南](https://llvm.org/docs/Passes.html)
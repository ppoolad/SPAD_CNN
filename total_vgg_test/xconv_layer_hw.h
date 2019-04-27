// ==============================================================
// File generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2017.2
// Copyright (C) 1986-2017 Xilinx, Inc. All Rights Reserved.
// 
// ==============================================================

// CTRL_BUS
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read)
//        bit 7  - auto_restart (Read/Write)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
// 0x10 : Data signal of input_offset
//        bit 31~0 - input_offset[31:0] (Read/Write)
// 0x14 : reserved
// 0x18 : Data signal of output_offset
//        bit 31~0 - output_offset[31:0] (Read/Write)
// 0x1c : reserved
// 0x20 : Data signal of b
//        bit 31~0 - b[31:0] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of od
//        bit 31~0 - od[31:0] (Read/Write)
// 0x2c : reserved
// 0x30 : Data signal of ox
//        bit 31~0 - ox[31:0] (Read/Write)
// 0x34 : reserved
// 0x38 : Data signal of oy
//        bit 31~0 - oy[31:0] (Read/Write)
// 0x3c : reserved
// 0x40 : Data signal of id
//        bit 31~0 - id[31:0] (Read/Write)
// 0x44 : reserved
// 0x48 : Data signal of ix
//        bit 31~0 - ix[31:0] (Read/Write)
// 0x4c : reserved
// 0x50 : Data signal of iy
//        bit 31~0 - iy[31:0] (Read/Write)
// 0x54 : reserved
// 0x58 : Data signal of s
//        bit 31~0 - s[31:0] (Read/Write)
// 0x5c : reserved
// 0x60 : Data signal of k
//        bit 31~0 - k[31:0] (Read/Write)
// 0x64 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XCONV_LAYER_CTRL_BUS_ADDR_AP_CTRL            0x00
#define XCONV_LAYER_CTRL_BUS_ADDR_GIE                0x04
#define XCONV_LAYER_CTRL_BUS_ADDR_IER                0x08
#define XCONV_LAYER_CTRL_BUS_ADDR_ISR                0x0c
#define XCONV_LAYER_CTRL_BUS_ADDR_INPUT_OFFSET_DATA  0x10
#define XCONV_LAYER_CTRL_BUS_BITS_INPUT_OFFSET_DATA  32
#define XCONV_LAYER_CTRL_BUS_ADDR_OUTPUT_OFFSET_DATA 0x18
#define XCONV_LAYER_CTRL_BUS_BITS_OUTPUT_OFFSET_DATA 32
#define XCONV_LAYER_CTRL_BUS_ADDR_B_DATA             0x20
#define XCONV_LAYER_CTRL_BUS_BITS_B_DATA             32
#define XCONV_LAYER_CTRL_BUS_ADDR_OD_DATA            0x28
#define XCONV_LAYER_CTRL_BUS_BITS_OD_DATA            32
#define XCONV_LAYER_CTRL_BUS_ADDR_OX_DATA            0x30
#define XCONV_LAYER_CTRL_BUS_BITS_OX_DATA            32
#define XCONV_LAYER_CTRL_BUS_ADDR_OY_DATA            0x38
#define XCONV_LAYER_CTRL_BUS_BITS_OY_DATA            32
#define XCONV_LAYER_CTRL_BUS_ADDR_ID_DATA            0x40
#define XCONV_LAYER_CTRL_BUS_BITS_ID_DATA            32
#define XCONV_LAYER_CTRL_BUS_ADDR_IX_DATA            0x48
#define XCONV_LAYER_CTRL_BUS_BITS_IX_DATA            32
#define XCONV_LAYER_CTRL_BUS_ADDR_IY_DATA            0x50
#define XCONV_LAYER_CTRL_BUS_BITS_IY_DATA            32
#define XCONV_LAYER_CTRL_BUS_ADDR_S_DATA             0x58
#define XCONV_LAYER_CTRL_BUS_BITS_S_DATA             32
#define XCONV_LAYER_CTRL_BUS_ADDR_K_DATA             0x60
#define XCONV_LAYER_CTRL_BUS_BITS_K_DATA             32
#define XCONV_LAYER_CTRL_BUS_ADDR_RELU_DATA          0x68
#define XCONV_LAYER_CTRL_BUS_BITS_RELU_DATA          32

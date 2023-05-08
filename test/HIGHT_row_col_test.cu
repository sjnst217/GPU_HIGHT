#include "HIGHT.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define _CRT_SECURE_NO_WARNINGS
#define HIGHT_BLOCK_SIZE    (8)

BYTE Delta[128] = {
       0x5A,0x6D,0x36,0x1B,0x0D,0x06,0x03,0x41,
       0x60,0x30,0x18,0x4C,0x66,0x33,0x59,0x2C,
       0x56,0x2B,0x15,0x4A,0x65,0x72,0x39,0x1C,
       0x4E,0x67,0x73,0x79,0x3C,0x5E,0x6F,0x37,
       0x5B,0x2D,0x16,0x0B,0x05,0x42,0x21,0x50,
       0x28,0x54,0x2A,0x55,0x6A,0x75,0x7A,0x7D,
       0x3E,0x5F,0x2F,0x17,0x4B,0x25,0x52,0x29,
       0x14,0x0A,0x45,0x62,0x31,0x58,0x6C,0x76,
       0x3B,0x1D,0x0E,0x47,0x63,0x71,0x78,0x7C,
       0x7E,0x7F,0x3F,0x1F,0x0F,0x07,0x43,0x61,
       0x70,0x38,0x5C,0x6E,0x77,0x7B,0x3D,0x1E,
       0x4F,0x27,0x53,0x69,0x34,0x1A,0x4D,0x26,
       0x13,0x49,0x24,0x12,0x09,0x04,0x02,0x01,
       0x40,0x20,0x10,0x08,0x44,0x22,0x11,0x48,
       0x64,0x32,0x19,0x0C,0x46,0x23,0x51,0x68,
       0x74,0x3A,0x5D,0x2E,0x57,0x6B,0x35,0x5A };

// __device__ BYTE HIGHT_F0[256] = {
//        0x00,0x86,0x0D,0x8B,0x1A,0x9C,0x17,0x91,
//        0x34,0xB2,0x39,0xBF,0x2E,0xA8,0x23,0xA5,
//        0x68,0xEE,0x65,0xE3,0x72,0xF4,0x7F,0xF9,
//        0x5C,0xDA,0x51,0xD7,0x46,0xC0,0x4B,0xCD,
//        0xD0,0x56,0xDD,0x5B,0xCA,0x4C,0xC7,0x41,
//        0xE4,0x62,0xE9,0x6F,0xFE,0x78,0xF3,0x75,
//        0xB8,0x3E,0xB5,0x33,0xA2,0x24,0xAF,0x29,
//        0x8C,0x0A,0x81,0x07,0x96,0x10,0x9B,0x1D,
//        0xA1,0x27,0xAC,0x2A,0xBB,0x3D,0xB6,0x30,
//        0x95,0x13,0x98,0x1E,0x8F,0x09,0x82,0x04,
//        0xC9,0x4F,0xC4,0x42,0xD3,0x55,0xDE,0x58,
//        0xFD,0x7B,0xF0,0x76,0xE7,0x61,0xEA,0x6C,
//        0x71,0xF7,0x7C,0xFA,0x6B,0xED,0x66,0xE0,
//        0x45,0xC3,0x48,0xCE,0x5F,0xD9,0x52,0xD4,
//        0x19,0x9F,0x14,0x92,0x03,0x85,0x0E,0x88,
//        0x2D,0xAB,0x20,0xA6,0x37,0xB1,0x3A,0xBC,
//        0x43,0xC5,0x4E,0xC8,0x59,0xDF,0x54,0xD2,
//        0x77,0xF1,0x7A,0xFC,0x6D,0xEB,0x60,0xE6,
//        0x2B,0xAD,0x26,0xA0,0x31,0xB7,0x3C,0xBA,
//        0x1F,0x99,0x12,0x94,0x05,0x83,0x08,0x8E,
//        0x93,0x15,0x9E,0x18,0x89,0x0F,0x84,0x02,
//        0xA7,0x21,0xAA,0x2C,0xBD,0x3B,0xB0,0x36,
//        0xFB,0x7D,0xF6,0x70,0xE1,0x67,0xEC,0x6A,
//        0xCF,0x49,0xC2,0x44,0xD5,0x53,0xD8,0x5E,
//        0xE2,0x64,0xEF,0x69,0xF8,0x7E,0xF5,0x73,
//        0xD6,0x50,0xDB,0x5D,0xCC,0x4A,0xC1,0x47,
//        0x8A,0x0C,0x87,0x01,0x90,0x16,0x9D,0x1B,
//        0xBE,0x38,0xB3,0x35,0xA4,0x22,0xA9,0x2F,
//        0x32,0xB4,0x3F,0xB9,0x28,0xAE,0x25,0xA3,
//        0x06,0x80,0x0B,0x8D,0x1C,0x9A,0x11,0x97,
//        0x5A,0xDC,0x57,0xD1,0x40,0xC6,0x4D,0xCB,
//        0x6E,0xE8,0x63,0xE5,0x74,0xF2,0x79,0xFF };
//
//__device__ BYTE HIGHT_F1[256] = {
//        0x00,0x58,0xB0,0xE8,0x61,0x39,0xD1,0x89,
//        0xC2,0x9A,0x72,0x2A,0xA3,0xFB,0x13,0x4B,
//        0x85,0xDD,0x35,0x6D,0xE4,0xBC,0x54,0x0C,
//        0x47,0x1F,0xF7,0xAF,0x26,0x7E,0x96,0xCE,
//        0x0B,0x53,0xBB,0xE3,0x6A,0x32,0xDA,0x82,
//        0xC9,0x91,0x79,0x21,0xA8,0xF0,0x18,0x40,
//        0x8E,0xD6,0x3E,0x66,0xEF,0xB7,0x5F,0x07,
//        0x4C,0x14,0xFC,0xA4,0x2D,0x75,0x9D,0xC5,
//        0x16,0x4E,0xA6,0xFE,0x77,0x2F,0xC7,0x9F,
//        0xD4,0x8C,0x64,0x3C,0xB5,0xED,0x05,0x5D,
//        0x93,0xCB,0x23,0x7B,0xF2,0xAA,0x42,0x1A,
//        0x51,0x09,0xE1,0xB9,0x30,0x68,0x80,0xD8,
//        0x1D,0x45,0xAD,0xF5,0x7C,0x24,0xCC,0x94,
//        0xDF,0x87,0x6F,0x37,0xBE,0xE6,0x0E,0x56,
//        0x98,0xC0,0x28,0x70,0xF9,0xA1,0x49,0x11,
//        0x5A,0x02,0xEA,0xB2,0x3B,0x63,0x8B,0xD3,
//        0x2C,0x74,0x9C,0xC4,0x4D,0x15,0xFD,0xA5,
//        0xEE,0xB6,0x5E,0x06,0x8F,0xD7,0x3F,0x67,
//        0xA9,0xF1,0x19,0x41,0xC8,0x90,0x78,0x20,
//        0x6B,0x33,0xDB,0x83,0x0A,0x52,0xBA,0xE2,
//        0x27,0x7F,0x97,0xCF,0x46,0x1E,0xF6,0xAE,
//        0xE5,0xBD,0x55,0x0D,0x84,0xDC,0x34,0x6C,
//        0xA2,0xFA,0x12,0x4A,0xC3,0x9B,0x73,0x2B,
//        0x60,0x38,0xD0,0x88,0x01,0x59,0xB1,0xE9,
//        0x3A,0x62,0x8A,0xD2,0x5B,0x03,0xEB,0xB3,
//        0xF8,0xA0,0x48,0x10,0x99,0xC1,0x29,0x71,
//        0xBF,0xE7,0x0F,0x57,0xDE,0x86,0x6E,0x36,
//        0x7D,0x25,0xCD,0x95,0x1C,0x44,0xAC,0xF4,
//        0x31,0x69,0x81,0xD9,0x50,0x08,0xE0,0xB8,
//        0xF3,0xAB,0x43,0x1B,0x92,0xCA,0x22,0x7A,
//        0xB4,0xEC,0x04,0x5C,0xD5,0x8D,0x65,0x3D,
//        0x76,0x2E,0xC6,0x9E,0x17,0x4F,0xA7,0xFF };

__device__ DWORD use_F0(DWORD X_before)
{
    DWORD use_XX1 = X_before;
    DWORD use_XX2 = X_before;
    DWORD use_XX3 = X_before;
    DWORD use_XX_F0;

    use_XX1 = (use_XX1 << 1) | (use_XX1 >> 7);
    use_XX2 = (use_XX2 << 2) | (use_XX2 >> 6);
    use_XX3 = (use_XX3 << 7) | (use_XX3 >> 1);
    use_XX_F0 = use_XX1 ^ use_XX2 ^ use_XX3;

    return use_XX_F0;
}

__device__ DWORD use_F1(DWORD X_before)
{
    DWORD use_XX4 = X_before;
    DWORD use_XX5 = X_before;
    DWORD use_XX6 = X_before;
    DWORD use_XX_F1;

    use_XX4 = (use_XX4 << 3) | (use_XX4 >> 5);
    use_XX5 = (use_XX5 << 4) | (use_XX5 >> 4);
    use_XX6 = (use_XX6 << 6) | (use_XX6 >> 2);
    use_XX_F1 = use_XX4 ^ use_XX5 ^ use_XX6;

    return use_XX_F1;
}



__device__ void    HIGHT_Encrypt(
    BYTE* roundkey,
    BYTE* us_pt,
    BYTE* ct)

{
    DWORD   XX[8];
    BYTE pt[8];
    pt[0] = *(us_pt + 0);
    pt[1] = *(us_pt + 1);
    pt[2] = *(us_pt + 2);
    pt[3] = *(us_pt + 3);
    pt[4] = *(us_pt + 4);
    pt[5] = *(us_pt + 5);
    pt[6] = *(us_pt + 6);
    pt[7] = *(us_pt + 7);
    printf("%p\n", us_pt[0]);
    printf("%02X, %02X, %02X, %02X, %02X, %02X, %02X, %02X\n", pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], pt[6], pt[7]);
    // First Round
    XX[1] = pt[1];
    XX[3] = pt[3];
    XX[5] = pt[5];
    XX[7] = pt[7];

    XX[0] = (pt[0] + roundkey[0]) & 0xFF;
    XX[2] = (pt[2] ^ roundkey[1]);
    XX[4] = (pt[4] + roundkey[2]) & 0xFF;
    XX[6] = (pt[6] ^ roundkey[3]);

    // Encryption Round 
#define HIGHT_ENC(k, i0,i1,i2,i3,i4,i5,i6,i7) {                         \
        XX[i0] = (XX[i0] ^ (use_F0(XX[i1]) + roundkey[4*k+3])) & 0xFF;    \
        XX[i2] = (XX[i2] + (use_F1(XX[i3]) ^ roundkey[4*k+2])) & 0xFF;    \
        XX[i4] = (XX[i4] ^ (use_F0(XX[i5]) + roundkey[4*k+1])) & 0xFF;    \
        XX[i6] = (XX[i6] + (use_F1(XX[i7]) ^ roundkey[4*k+0])) & 0xFF;    \
    }

    HIGHT_ENC(2, 7, 6, 5, 4, 3, 2, 1, 0);
    HIGHT_ENC(3, 6, 5, 4, 3, 2, 1, 0, 7);
    HIGHT_ENC(4, 5, 4, 3, 2, 1, 0, 7, 6);
    HIGHT_ENC(5, 4, 3, 2, 1, 0, 7, 6, 5);
    HIGHT_ENC(6, 3, 2, 1, 0, 7, 6, 5, 4);
    HIGHT_ENC(7, 2, 1, 0, 7, 6, 5, 4, 3);
    HIGHT_ENC(8, 1, 0, 7, 6, 5, 4, 3, 2);
    HIGHT_ENC(9, 0, 7, 6, 5, 4, 3, 2, 1);
    HIGHT_ENC(10, 7, 6, 5, 4, 3, 2, 1, 0);
    HIGHT_ENC(11, 6, 5, 4, 3, 2, 1, 0, 7);
    HIGHT_ENC(12, 5, 4, 3, 2, 1, 0, 7, 6);
    HIGHT_ENC(13, 4, 3, 2, 1, 0, 7, 6, 5);
    HIGHT_ENC(14, 3, 2, 1, 0, 7, 6, 5, 4);
    HIGHT_ENC(15, 2, 1, 0, 7, 6, 5, 4, 3);
    HIGHT_ENC(16, 1, 0, 7, 6, 5, 4, 3, 2);
    HIGHT_ENC(17, 0, 7, 6, 5, 4, 3, 2, 1);
    HIGHT_ENC(18, 7, 6, 5, 4, 3, 2, 1, 0);
    HIGHT_ENC(19, 6, 5, 4, 3, 2, 1, 0, 7);
    HIGHT_ENC(20, 5, 4, 3, 2, 1, 0, 7, 6);
    HIGHT_ENC(21, 4, 3, 2, 1, 0, 7, 6, 5);
    HIGHT_ENC(22, 3, 2, 1, 0, 7, 6, 5, 4);
    HIGHT_ENC(23, 2, 1, 0, 7, 6, 5, 4, 3);
    HIGHT_ENC(24, 1, 0, 7, 6, 5, 4, 3, 2);
    HIGHT_ENC(25, 0, 7, 6, 5, 4, 3, 2, 1);
    HIGHT_ENC(26, 7, 6, 5, 4, 3, 2, 1, 0);
    HIGHT_ENC(27, 6, 5, 4, 3, 2, 1, 0, 7);
    HIGHT_ENC(28, 5, 4, 3, 2, 1, 0, 7, 6);
    HIGHT_ENC(29, 4, 3, 2, 1, 0, 7, 6, 5);
    HIGHT_ENC(30, 3, 2, 1, 0, 7, 6, 5, 4);
    HIGHT_ENC(31, 2, 1, 0, 7, 6, 5, 4, 3);
    HIGHT_ENC(32, 1, 0, 7, 6, 5, 4, 3, 2);
    HIGHT_ENC(33, 0, 7, 6, 5, 4, 3, 2, 1);

    // Final Round
    ct[1] = (BYTE)XX[2];
    ct[3] = (BYTE)XX[4];
    ct[5] = (BYTE)XX[6];
    ct[7] = (BYTE)XX[0];

    ct[0] = (BYTE)(XX[1] + roundkey[4]);
    ct[2] = (BYTE)(XX[3] ^ roundkey[5]);
    ct[4] = (BYTE)(XX[5] + roundkey[6]);
    ct[6] = (BYTE)(XX[7] ^ roundkey[7]);
}

__device__ void HIGHT_Decrypt(BYTE* RoundKey, BYTE* ct, BYTE* dt)
{
    DWORD   XX[8];



    XX[2] = (BYTE)ct[1];
    XX[4] = (BYTE)ct[3];
    XX[6] = (BYTE)ct[5];
    XX[0] = (BYTE)ct[7];

    XX[1] = (BYTE)(ct[0] - RoundKey[4]);
    XX[3] = (BYTE)(ct[2] ^ RoundKey[5]);
    XX[5] = (BYTE)(ct[4] - RoundKey[6]);
    XX[7] = (BYTE)(ct[6] ^ RoundKey[7]);

#define HIGHT_DEC(k, i0,i1,i2,i3,i4,i5,i6,i7) {                         \
        XX[i1] = (XX[i1] - (use_F1(XX[i2]) ^ RoundKey[4*k+2])) & 0xFF;    \
        XX[i3] = (XX[i3] ^ (use_F0(XX[i4]) + RoundKey[4*k+1])) & 0xFF;    \
        XX[i5] = (XX[i5] - (use_F1(XX[i6]) ^ RoundKey[4*k+0])) & 0xFF;    \
        XX[i7] = (XX[i7] ^ (use_F0(XX[i0]) + RoundKey[4*k+3])) & 0xFF;    \
    }

    HIGHT_DEC(33, 7, 6, 5, 4, 3, 2, 1, 0);
    HIGHT_DEC(32, 0, 7, 6, 5, 4, 3, 2, 1);
    HIGHT_DEC(31, 1, 0, 7, 6, 5, 4, 3, 2);
    HIGHT_DEC(30, 2, 1, 0, 7, 6, 5, 4, 3);
    HIGHT_DEC(29, 3, 2, 1, 0, 7, 6, 5, 4);
    HIGHT_DEC(28, 4, 3, 2, 1, 0, 7, 6, 5);
    HIGHT_DEC(27, 5, 4, 3, 2, 1, 0, 7, 6);
    HIGHT_DEC(26, 6, 5, 4, 3, 2, 1, 0, 7);
    HIGHT_DEC(25, 7, 6, 5, 4, 3, 2, 1, 0);
    HIGHT_DEC(24, 0, 7, 6, 5, 4, 3, 2, 1);
    HIGHT_DEC(23, 1, 0, 7, 6, 5, 4, 3, 2);
    HIGHT_DEC(22, 2, 1, 0, 7, 6, 5, 4, 3);
    HIGHT_DEC(21, 3, 2, 1, 0, 7, 6, 5, 4);
    HIGHT_DEC(20, 4, 3, 2, 1, 0, 7, 6, 5);
    HIGHT_DEC(19, 5, 4, 3, 2, 1, 0, 7, 6);
    HIGHT_DEC(18, 6, 5, 4, 3, 2, 1, 0, 7);
    HIGHT_DEC(17, 7, 6, 5, 4, 3, 2, 1, 0);
    HIGHT_DEC(16, 0, 7, 6, 5, 4, 3, 2, 1);
    HIGHT_DEC(15, 1, 0, 7, 6, 5, 4, 3, 2);
    HIGHT_DEC(14, 2, 1, 0, 7, 6, 5, 4, 3);
    HIGHT_DEC(13, 3, 2, 1, 0, 7, 6, 5, 4);
    HIGHT_DEC(12, 4, 3, 2, 1, 0, 7, 6, 5);
    HIGHT_DEC(11, 5, 4, 3, 2, 1, 0, 7, 6);
    HIGHT_DEC(10, 6, 5, 4, 3, 2, 1, 0, 7);
    HIGHT_DEC(9, 7, 6, 5, 4, 3, 2, 1, 0);
    HIGHT_DEC(8, 0, 7, 6, 5, 4, 3, 2, 1);
    HIGHT_DEC(7, 1, 0, 7, 6, 5, 4, 3, 2);
    HIGHT_DEC(6, 2, 1, 0, 7, 6, 5, 4, 3);
    HIGHT_DEC(5, 3, 2, 1, 0, 7, 6, 5, 4);
    HIGHT_DEC(4, 4, 3, 2, 1, 0, 7, 6, 5);
    HIGHT_DEC(3, 5, 4, 3, 2, 1, 0, 7, 6);
    HIGHT_DEC(2, 6, 5, 4, 3, 2, 1, 0, 7);

    dt[1] = (BYTE)(XX[1]);
    dt[3] = (BYTE)(XX[3]);
    dt[5] = (BYTE)(XX[5]);
    dt[7] = (BYTE)(XX[7]);

    dt[0] = (BYTE)(XX[0] - RoundKey[0]);
    dt[2] = (BYTE)(XX[2] ^ RoundKey[1]);
    dt[4] = (BYTE)(XX[4] - RoundKey[2]);
    dt[6] = (BYTE)(XX[6] ^ RoundKey[3]);
}

__global__ void HIGHT_Encryption(unsigned char* key, unsigned char* pt, unsigned char* ct) {
    __shared__ BYTE GPU_rk[136];
    memcpy(GPU_rk, key, 136 * sizeof(BYTE));
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    //HIGHT_Encrypt(GPU_rk, pt + tid, ct + tid);
    HIGHT_Encrypt(GPU_rk, pt + tid, ct + tid);

}

__global__ void HIGHT_Decryption(unsigned char* key, unsigned char* ct, unsigned char* dt)
{
    __shared__ BYTE GPU_rk[136];
    memcpy(GPU_rk, key, 136 * sizeof(BYTE));
    int tid = threadIdx.x * HIGHT_BLOCK_SIZE + blockIdx.x * blockDim.x * HIGHT_BLOCK_SIZE;
    HIGHT_Decrypt(GPU_rk, ct + tid, dt + tid);
}

void    HIGHT_KeySched(
    BYTE* UserKey,
    DWORD   UserKeyLen,
    BYTE* RoundKey)
{
    int     i, j;

    for (i = 0; i < 4; i++) {
        RoundKey[i] = UserKey[i + 12];
        RoundKey[i + 4] = UserKey[i];
    }

    for (i = 0; i < 8; i++) {
        for (j = 0; j < 8; j++)
            RoundKey[8 + 16 * i + j] = (BYTE)(UserKey[(j - i) & 7] + Delta[16 * i + j]);

        for (j = 0; j < 8; j++)
            RoundKey[8 + 16 * i + j + 8] = (BYTE)(UserKey[((j - i) & 7) + 8] + Delta[16 * i + j + 8]);
    }
}

void HIGHT_GPU_performance_Test(unsigned long long Blocksize, unsigned long long Threadsize) {

    int i;

    cudaEvent_t start, stop;
    float elapsed_time_ms = 0.0f;
    //CPU Memory
    unsigned char CPU_masterkey[16] = { 0x88, 0xE3, 0x4F, 0x8F, 0x08, 0x17, 0x79, 0xF1, 0xE9, 0xF3, 0x94, 0x37, 0x0A, 0xD4, 0x05, 0x89 };
    unsigned char CPU_roundkey[136] = { 0 };
    unsigned char* cpu_pt = NULL;
    cpu_pt = (unsigned char*)malloc(sizeof(unsigned char) * Blocksize * Threadsize * HIGHT_BLOCK_SIZE);
    if (cpu_pt == NULL)
        return;
    for (int i = 0; i < Blocksize * Threadsize; i++) {
        cpu_pt[HIGHT_BLOCK_SIZE * i + 0] = 0xD7;
        cpu_pt[HIGHT_BLOCK_SIZE * i + 1] = 0x6D;
        cpu_pt[HIGHT_BLOCK_SIZE * i + 2] = 0x0D;
        cpu_pt[HIGHT_BLOCK_SIZE * i + 3] = 0x18;
        cpu_pt[HIGHT_BLOCK_SIZE * i + 4] = 0x32;
        cpu_pt[HIGHT_BLOCK_SIZE * i + 5] = 0x7E;
        cpu_pt[HIGHT_BLOCK_SIZE * i + 6] = 0xC5;
        cpu_pt[HIGHT_BLOCK_SIZE * i + 7] = 0x62;
    }
    unsigned char* cpu_ct = NULL;
    cpu_ct = (unsigned char*)malloc(sizeof(unsigned char) * Blocksize * Threadsize * HIGHT_BLOCK_SIZE);
    unsigned char* cpu_dt = NULL;
    cpu_dt = (unsigned char*)malloc(sizeof(unsigned char) * Blocksize * Threadsize * HIGHT_BLOCK_SIZE);


    unsigned char* us_cpu_pt = NULL;
    us_cpu_pt = (unsigned char*)malloc(sizeof(unsigned char) * Blocksize * Threadsize * HIGHT_BLOCK_SIZE);

    //GPU Memory
    unsigned char* GPU_pt;
    unsigned char* GPU_ct;
    unsigned char* GPU_dt;
    unsigned char* GPU_roundkey;

    unsigned char* us_GPU_pt;

    //GPU memory allocation
    cudaMalloc((void**)&GPU_pt, Blocksize * Threadsize * HIGHT_BLOCK_SIZE * sizeof(unsigned char));
    cudaMalloc((void**)&GPU_ct, Blocksize * Threadsize * HIGHT_BLOCK_SIZE * sizeof(unsigned char));
    cudaMalloc((void**)&GPU_dt, Blocksize * Threadsize * HIGHT_BLOCK_SIZE * sizeof(unsigned char));
    cudaMalloc((void**)&GPU_roundkey, 136 * sizeof(unsigned char));

    cudaMalloc((void**)&us_GPU_pt, Blocksize * Threadsize * sizeof(unsigned char));

    //CPU-> GPU Memory copy
    HIGHT_KeySched(CPU_masterkey, 16, CPU_roundkey); //roundkey를 만들기

    cudaMemcpy(GPU_pt, cpu_pt, Blocksize * Threadsize * HIGHT_BLOCK_SIZE * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(GPU_roundkey, CPU_roundkey, 136 * sizeof(unsigned char), cudaMemcpyHostToDevice);

    printf("\n\nEncryption...\n");

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    HIGHT_Encryption << <Blocksize, Threadsize >> > (GPU_roundkey, GPU_pt, GPU_ct);

    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    elapsed_time_ms /= 100;
    elapsed_time_ms = (Blocksize * Threadsize * HIGHT_BLOCK_SIZE * sizeof(unsigned char)) / elapsed_time_ms;
    elapsed_time_ms *= 1000;
    elapsed_time_ms /= (1024 * 1024 * 1024);
    printf("File size = %d MB, Grid : %d, Block : %d, Performance : %4.2f GB/s\n", (Blocksize * Threadsize * HIGHT_BLOCK_SIZE) / (1024 * 1024), Blocksize, Threadsize, elapsed_time_ms);
    getchar();
    getchar();

    cudaMemcpy(cpu_ct, GPU_ct, Blocksize * Threadsize * HIGHT_BLOCK_SIZE * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaGetLastError();
    cudaDeviceSynchronize();

    printf("HIGHT Cipher Text : \n");
    for (i = 0; i < Blocksize * Threadsize * HIGHT_BLOCK_SIZE; i++)
    {
        printf("%02X ", cpu_ct[i]);
        if ((i + 1) % 8 == 0)
        {
            printf("\n");
        }
    }
    printf("\n");

    getchar();
    printf("\n\nDecryption...\n");

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    HIGHT_Decryption << <Blocksize, Threadsize >> > (GPU_roundkey, GPU_ct, GPU_dt);

    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    elapsed_time_ms /= 100;
    elapsed_time_ms = (Blocksize * Threadsize * HIGHT_BLOCK_SIZE * sizeof(unsigned char)) / elapsed_time_ms;
    elapsed_time_ms *= 1000;
    elapsed_time_ms /= (1024 * 1024 * 1024);
    printf("File size = %d MB, Grid : %d, Block : %d, Performance : %4.2f GB/s\n", (Blocksize * Threadsize * HIGHT_BLOCK_SIZE) / (1024 * 1024), Blocksize, Threadsize, elapsed_time_ms);
    getchar();
    getchar();

    cudaMemcpy(cpu_dt, GPU_dt, Blocksize * Threadsize * HIGHT_BLOCK_SIZE * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    printf("Plaintext  : \n");
    for (int i = 0; i < Blocksize * Threadsize * HIGHT_BLOCK_SIZE; i++) {
        printf("%02X ", cpu_dt[i]);
        if ((i + 1) % 8 == 0)
        {
            printf("\n");
        }
    }

    cudaFree(GPU_pt);
    cudaFree(GPU_ct);
    cudaFree(GPU_dt);
    cudaFree(GPU_roundkey);
    cudaFree(us_GPU_pt);
    free(cpu_ct);
    free(cpu_dt);
    free(cpu_pt);
    free(us_cpu_pt);
}

int main()
{
    unsigned long long BlockSize = 0, TreadSize = 0;
    printf("block의 크기: ");
    scanf("%d", &BlockSize);

    printf("tread의 크기: ");
    scanf("%d", &TreadSize);

    printf("\n");
    HIGHT_GPU_performance_Test(BlockSize, TreadSize);

    return 0;
}
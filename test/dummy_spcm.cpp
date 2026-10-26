//
// Created by Juntian Tu on 2026-04-14.
// Used to mimic a spcm libraries without a spcm card
//

#include  <stdlib.h>
#include "../spcm_header/dlltyp.h"
#include "../spcm_header/regs.h"
#include "../spcm_header/spcerr.h"
#include "../spcm_header/spcm_drv.h"
#include "../spcm_header/spcm_cuda_common.h"

using namespace std;
bool bInitCUDADevice (int lCUDADevIdx) {return 0;}
void* pvGetRDMABuffer (int lCUDADevIdx, size_t qwDMABufferSize_bytes) {return 0;}
SPCM_IMPORT drv_handle _stdcall spcm_hOpen (const char* szDeviceName) {return 0;}
SPCM_IMPORT void _stdcall spcm_vClose (drv_handle  hDevice) {}
SPCM_IMPORT uint32 _stdcall spcm_dwSetParam_i32 (
    drv_handle  hDevice,                // handle to an already opened device
    int32       lRegister,              // software register to be modified
    int32       lValue)
    {return 0;}
SPCM_IMPORT uint32 _stdcall spcm_dwSetParam_i64m (
    drv_handle  hDevice,                // handle to an already opened device
    int32       lRegister,              // software register to be modified
    int32       lValueHigh,             // upper 32 bit of the value. Containing the sign bit !
    uint32      dwValueLow)            // lower 32 bit of the value.
    {return 0;}
SPCM_IMPORT uint32 _stdcall spcm_dwSetParam_i64 (
    drv_handle  hDevice,                // handle to an already opened device
    int32       lRegister,              // software register to be modified
    int64       llValue)               // the value to be set
    {return 0;}
SPCM_IMPORT uint32 _stdcall spcm_dwGetParam_i32 (
    drv_handle  hDevice,                // handle to an already opened device
    int32       lRegister,              // software register to be read out
    int32*      plValue)               // pointer for the return value
    {return 0;}
SPCM_IMPORT uint32 _stdcall spcm_dwGetParam_i64m (
    drv_handle  hDevice,                // handle to an already opened device
    int32       lRegister,              // software register to be read out
    int32*      plValueHigh,            // pointer for the upper part of the return value
    uint32*     pdwValueLow)           // pointer for the lower part of the return value
    {return 0;}
SPCM_IMPORT uint32 _stdcall spcm_dwGetParam_i64 (
    drv_handle  hDevice,                // handle to an already opened device
    int32       lRegister,              // software register to be read out
    int64*      pllValue)              // pointer for the return value
    {return 0;}
SPCM_IMPORT uint32 _stdcall spcm_dwDefTransfer_i64m(
    drv_handle  hDevice,                // handle to an already opened device
    uint32      dwBufType,              // type of the buffer to define as listed above under SPCM_BUF_XXXX
    uint32      dwDirection,            // the transfer direction as defined above
    uint32      dwNotifySize,           // amount of bytes after which i want do receive an event (0=end of transfer)
    void*       pvDataBuffer,           // pointer to the data buffer
    uint32      dwBrdOffsH,             // high part of offset in board memory
    uint32      dwBrdOffsL,             // low part of offset in board memory
    uint32      dwTransferLenH,         // high part of transfer buffer length
    uint32      dwTransferLenL)        // low part of transfer buffer length
    {return 0;}
SPCM_IMPORT uint32 _stdcall spcm_dwDefTransfer_i64 (
    drv_handle  hDevice,                // handle to an already opened device
    uint32      dwBufType,              // type of the buffer to define as listed above under SPCM_BUF_XXXX
    uint32      dwDirection,            // the transfer direction as defined above
    uint32      dwNotifySize,           // amount of bytes after which i want do receive an event (0=end of transfer)
    void*       pvDataBuffer,           // pointer to the data buffer
    uint64      qwBrdOffs,              // offset for transfer in board memory
    uint64      qwTransferLen)         // buffer length
    {return 0;}
SPCM_IMPORT uint32 _stdcall spcm_dwInvalidateBuf (
    drv_handle  hDevice,                // handle to an already opened device
    uint32      dwBufType)             // type of the buffer to invalidate as listed above under SPCM_BUF_XXXX
    {return 0;}
SPCM_IMPORT uint32 _stdcall spcm_dwGetContBuf_i64 (
    drv_handle  hDevice,                // handle to an already opened device
    uint32      dwBufType,              // type of the buffer to read as listed above under SPCM_BUF_XXXX
    void**      ppvDataBuffer,          // address of available data buffer
    uint64*     pqwContBufLen)         // length of available continuous buffer
    {return 0;}
SPCM_IMPORT uint32 _stdcall spcm_dwGetContBuf_i64m (
    drv_handle  hDevice,                // handle to an already opened device
    uint32      dwBufType,              // type of the buffer to read as listed above under SPCM_BUF_XXXX
    void**      ppvDataBuffer,          // address of available data buffer
    uint32*     pdwContBufLenH,         // high part of length of available continuous buffer
    uint32*     pdwContBufLenL)        // low part of length of available continuous buffer
    {return 0;}
SPCM_IMPORT uint32 _stdcall spcm_dwGetErrorInfo_i32 (
    drv_handle  hDevice,                // handle to an already opened device
    uint32*     pdwErrorReg,            // adress of the error register (can zero if not of interest)
    int32*      plErrorValue,           // adress of the error value (can zero if not of interest)
    char        pszErrorTextBuffer[ERRORTEXTLEN]) // text buffer for text error
    {return 0;}
SPCM_IMPORT uint32 _stdcall spcm_dwStartEBox ()
    {return 0;}
SPCM_IMPORT uint32 _stdcall spcm_dwDiscovery (
    char** pszVisaString,           // user-allocated array of C-strings to return the VISA strings
    uint32 dwMaxNoOfDevices,        // the maximum number of devices to be returned
    uint32 dwMaxVisaLen,            // maximum length of one entry in pszVisaString
    uint32 dwTimeout)              // time in milli seconds that the function will wait until each device has answered.
    {return 0;}
SPCM_IMPORT uint32 _stdcall spcm_dwSendIDNRequest (
    char** szIdnString,                 //the IDN string looks like that: <manufacturer>,<model>,<serial number>,<firmware version>
    uint32 dwNoOfDevices,
    uint32 dwIdnStringLen)            //string which contains manufacturer, the model, the serial number and the firmware version
    {return 0;}

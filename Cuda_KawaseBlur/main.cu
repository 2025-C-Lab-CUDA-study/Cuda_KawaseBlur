#include<cuda_runtime.h>
#include<device_launch_parameters.h>

#include <iostream>
#include"BmpUtile.h"


#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CUDA_KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())


// Typedef ================

using uchar = unsigned char;


// Consts ================

constexpr int RATIO = 1;
constexpr int BLOCK = 16;
constexpr int BLURING_TIMES = 1;


__global__ void BilinearReduce(uchar* dstBuffer, size_t dstPitch, uchar* srcBuffer, size_t srcPitch, int dstWidth, int dstHeight, int srcWidth, int srcHeight)
{
    int src_pitch = srcPitch / sizeof(uchar);
    int dst_pitch = dstPitch / sizeof(uchar);

    for (
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        y < dstHeight;
        y += blockDim.y * gridDim.y
        )
    {
        for (
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            x < dstWidth;
            x += blockDim.x * gridDim.x
            )
        {
            int srcX = min(max(x * 2, 0), srcWidth - RATIO);
            int srcY = min(max(y * 2, 0), srcHeight - RATIO);

            uchar c1 = srcBuffer[srcX + src_pitch * srcY];
            uchar c2 = srcBuffer[srcX + RATIO + src_pitch * srcY];
            uchar c3 = srcBuffer[srcX + src_pitch * (srcY + RATIO)];
            uchar c4 = srcBuffer[srcX + RATIO + src_pitch * (srcY + RATIO)];

            dstBuffer[x + dst_pitch * y] = static_cast<uchar>((c1 + c2 + c3 + c4) * 0.25);
        }
    }
}

__global__ void BilinearIncrease(uchar* dstBuffer, size_t dstPitch, uchar* srcBuffer, size_t srcPitch, int dstWidth, int dstHeight, int srcWidth, int srcHeight)
{
    int src_pitch = srcPitch / sizeof(uchar);
    int dst_pitch = dstPitch / sizeof(uchar);

    for (
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        y < dstHeight;
        y += blockDim.y * gridDim.y
        )
    {
        for (
            int x = blockDim.x * blockIdx.x + threadIdx.x;
            x < dstWidth;
            x += blockDim.x * gridDim.x
            )
        {
            int srcX = min(max(x / 2, 0), dstWidth - RATIO);
            int srcY = min(max(y / 2, 0), dstHeight - RATIO);

            uchar c1 = srcBuffer[srcX         + src_pitch * srcY];
            uchar c2 = srcBuffer[srcX + RATIO + src_pitch * srcY];
            uchar c3 = srcBuffer[srcX         + src_pitch * (srcY + RATIO)];
            uchar c4 = srcBuffer[srcX + RATIO + src_pitch * (srcY + RATIO)];

            dstBuffer[x + dst_pitch * y] = static_cast<uchar>((c1 + c2 + c3 + c4) * 0.25);
        }
    }
}


__global__ void KawaseBlur(uchar* dstBuffer, size_t dstPitch, cudaTextureObject_t texObject, int offset, int width, int height)
{
    int dst_pitch = dstPitch / sizeof(uchar);

    for (int y = blockDim.y * blockIdx.y + threadIdx.y; y < height; y += blockDim.y * gridDim.y)
    {
        for (int x = blockDim.x * blockIdx.x + threadIdx.x; x < width; x += blockDim.x * gridDim.x)
        {
            int xmo = max(x - offset, 0);
            int ymo = max(y - offset, 0);
            int xpo = min(x + offset, width  - 1);
            int ypo = min(y + offset, height - 1);

            uchar c1 = tex2D<uchar>(texObject, xmo, ymo);
            uchar c2 = tex2D<uchar>(texObject, xpo, ymo);
            uchar c3 = tex2D<uchar>(texObject, xmo, ypo);
            uchar c4 = tex2D<uchar>(texObject, xpo, ypo);

            dstBuffer[x + dst_pitch * y] = static_cast<uchar>((c1 + c2 + c3 + c4) * 0.25);
        }
   }
}



int main(void)
{
    // Set Host data =========================================================================================

    uchar* h_rb = nullptr;
    uchar* h_gb = nullptr;
    uchar* h_bb = nullptr;
    int h_width, h_height;

    const char* path = "C:\\Users\\james\\Documents\\2025\\source_code\\lenna.bmp";
    if (!Bmp::BmpToRgbBuffers(path, &h_rb, &h_gb, &h_bb, h_width, h_height))
    {
        if (!h_rb) free(h_rb);
        if (!h_gb) free(h_gb);
        if (!h_bb) free(h_bb);
    }

    // Set Device data ========================================================================================

    int d_width = h_width;
    int d_height = h_height;
    size_t firstPitch, secondPitch;
    uchar* d_rb_first = nullptr, * d_rb_second = nullptr;
    uchar* d_gb_first = nullptr, * d_gb_second = nullptr;
    uchar* d_bb_first = nullptr, * d_bb_second = nullptr;

    CUDA_CHECK(cudaMallocPitch(&d_rb_first, &firstPitch, sizeof(uchar) * d_width, d_height)); // fisrt buffer
    CUDA_CHECK(cudaMallocPitch(&d_gb_first, &firstPitch, sizeof(uchar) * d_width, d_height));
    CUDA_CHECK(cudaMallocPitch(&d_bb_first, &firstPitch, sizeof(uchar) * d_width, d_height));

    CUDA_CHECK(cudaMallocPitch(&d_rb_second, &secondPitch, sizeof(uchar) * (d_width), d_height)); // second buffer
    CUDA_CHECK(cudaMallocPitch(&d_gb_second, &secondPitch, sizeof(uchar) * (d_width), d_height));
    CUDA_CHECK(cudaMallocPitch(&d_bb_second, &secondPitch, sizeof(uchar) * (d_width), d_height));

    CUDA_CHECK(cudaMemcpy2D(d_rb_first, firstPitch, h_rb, sizeof(uchar) * h_width, sizeof(uchar) * h_width, h_height, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(d_gb_first, firstPitch, h_gb, sizeof(uchar) * h_width, sizeof(uchar) * h_width, h_height, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy2D(d_bb_first, firstPitch, h_bb, sizeof(uchar) * h_width, sizeof(uchar) * h_width, h_height, cudaMemcpyHostToDevice));

	// Set Texture data =========================================================================================
    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar>();
    cudaArray_t d_rb_array = nullptr;
    cudaArray_t d_gb_array = nullptr;
    cudaArray_t d_bb_array = nullptr;

    cudaMallocArray(&d_rb_array, &channelDesc, d_width, d_height);
    cudaMallocArray(&d_gb_array, &channelDesc, d_width, d_height);
    cudaMallocArray(&d_bb_array, &channelDesc, d_width, d_height);

	cudaResourceDesc rbResDesc;
    memset(&rbResDesc, 0, sizeof(rbResDesc));
	rbResDesc.resType = cudaResourceTypeArray;
	rbResDesc.res.array.array = d_rb_array;

	cudaResourceDesc gbResDesc;
    memset(&gbResDesc, 0, sizeof(gbResDesc));
	gbResDesc.resType = cudaResourceTypeArray;
	gbResDesc.res.array.array = d_gb_array;

	cudaResourceDesc bbResDesc;
    memset(&bbResDesc, 0, sizeof(bbResDesc));
	bbResDesc.resType = cudaResourceTypeArray;
	bbResDesc.res.array.array = d_bb_array;

	cudaTextureDesc rbTexDesc;
    memset(&rbTexDesc, 0, sizeof(rbTexDesc));
	rbTexDesc.addressMode[0] = cudaAddressModeClamp; // x-axis
	rbTexDesc.addressMode[1] = cudaAddressModeClamp; // y-axis
	rbTexDesc.filterMode = cudaFilterModePoint; // Point filtering
	rbTexDesc.readMode = cudaReadModeElementType; // Read as element type
	rbTexDesc.normalizedCoords = false; // Use absolute coordinates
   
    cudaTextureDesc gbTexDesc;
    memset(&gbTexDesc, 0, sizeof(gbTexDesc));
    gbTexDesc.addressMode[0] = cudaAddressModeClamp; // x-axis
    gbTexDesc.addressMode[1] = cudaAddressModeClamp; // y-axis
    gbTexDesc.filterMode = cudaFilterModePoint; // Point filtering
    gbTexDesc.readMode = cudaReadModeElementType; // Read as element type
    gbTexDesc.normalizedCoords = false; // Use absolute coordinates

    cudaTextureDesc bbTexDesc;
    memset(&bbTexDesc, 0, sizeof(bbTexDesc));
    bbTexDesc.addressMode[0] = cudaAddressModeClamp; // x-axis
    bbTexDesc.addressMode[1] = cudaAddressModeClamp; // y-axis
    bbTexDesc.filterMode = cudaFilterModePoint; // Point filtering
    bbTexDesc.readMode = cudaReadModeElementType; // Read as element type
    bbTexDesc.normalizedCoords = false; // Use absolute coordinates

	cudaTextureObject_t texObjectRb, texObjectGb, texObjectBb;
    CUDA_CHECK(cudaCreateTextureObject(&texObjectRb, &rbResDesc, &rbTexDesc, nullptr));
    CUDA_CHECK(cudaCreateTextureObject(&texObjectGb, &gbResDesc, &gbTexDesc, nullptr));
    CUDA_CHECK(cudaCreateTextureObject(&texObjectBb, &bbResDesc, &bbTexDesc, nullptr));

    // Run Bilinear and kawaseBlur ===============================================================================

    int offset = 1;
    int transWidth  = d_width  / 4;
    int transHeight = d_height / 4;
    
    dim3 block(BLOCK, BLOCK);

    for (int reduceKawase = 0; reduceKawase < BLURING_TIMES; ++reduceKawase)    // Run reduce image & kawase blur ---------------------------------------------
    {
        dim3 reduceGrid((transWidth + BLOCK - 1) / BLOCK, (transHeight + BLOCK - 1) / BLOCK);
        BilinearReduce << <reduceGrid, block >> > (d_rb_second, secondPitch, d_rb_first, firstPitch, d_width / 2, d_height / 2, d_width, d_height);
        CUDA_KERNEL_CHECK();
        BilinearReduce << <reduceGrid, block >> > (d_gb_second, secondPitch, d_gb_first, firstPitch, d_width / 2, d_height / 2, d_width, d_height);
        CUDA_KERNEL_CHECK();
        BilinearReduce << <reduceGrid, block >> > (d_bb_second, secondPitch, d_bb_first, firstPitch, d_width / 2, d_height / 2, d_width, d_height);
        CUDA_KERNEL_CHECK();

        cudaDeviceSynchronize();
        cudaDestroyTextureObject(texObjectRb);
        cudaDestroyTextureObject(texObjectGb);
        cudaDestroyTextureObject(texObjectBb);
        
        transWidth  /= 2;
        transHeight /= 2;
        d_width     /= 2;
        d_height    /= 2;

        CUDA_CHECK(cudaMemcpy2DToArray(d_rb_array, 0, 0, d_rb_second, secondPitch, sizeof(uchar) * h_width, h_height, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy2DToArray(d_gb_array, 0, 0, d_gb_second, secondPitch, sizeof(uchar) * h_width, h_height, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy2DToArray(d_bb_array, 0, 0, d_bb_second, secondPitch, sizeof(uchar) * h_width, h_height, cudaMemcpyDeviceToDevice));
            
        rbResDesc.res.array.array = d_rb_array;
        gbResDesc.res.array.array = d_gb_array;
        bbResDesc.res.array.array = d_bb_array;

        CUDA_CHECK(cudaCreateTextureObject(&texObjectRb, &rbResDesc, &rbTexDesc, nullptr));
        CUDA_CHECK(cudaCreateTextureObject(&texObjectGb, &gbResDesc, &gbTexDesc, nullptr));
        CUDA_CHECK(cudaCreateTextureObject(&texObjectBb, &bbResDesc, &bbTexDesc, nullptr));

        dim3 kawaseGrid((transWidth + BLOCK - 1) / BLOCK, (transHeight + BLOCK - 1) / BLOCK);
        KawaseBlur << <kawaseGrid, block >> > (d_rb_first, firstPitch, texObjectRb, offset, d_width, d_height);
        CUDA_KERNEL_CHECK();
        KawaseBlur << <kawaseGrid, block >> > (d_gb_first, firstPitch, texObjectGb, offset, d_width, d_height);
        CUDA_KERNEL_CHECK();
        KawaseBlur << <kawaseGrid, block >> > (d_bb_first, firstPitch, texObjectBb, offset, d_width, d_height);
        CUDA_KERNEL_CHECK();

        ++offset;
    }


    // Check image status -----------------------------------------------------------------------------------------------------
    CUDA_CHECK(cudaMemcpy2D(h_rb, sizeof(uchar) * h_width, d_rb_first, firstPitch, d_width, d_height, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(h_gb, sizeof(uchar) * h_width, d_gb_first, firstPitch, d_width, d_height, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(h_bb, sizeof(uchar) * h_width, d_bb_first, firstPitch, d_width, d_height, cudaMemcpyDeviceToHost));
   
    const char* middlecheckPath = "C:\\Users\\james\\Documents\\2025\\source_code\\middleCheckLenna.bmp";
    if (!Bmp::RgbBuffersToBmp(middlecheckPath, h_rb, h_gb, h_bb, h_width, h_height))
    {
        std::cout << "Error : Writing bmp file failed";
    }

    --offset;
    for (int increaseKawase = 0; increaseKawase < BLURING_TIMES; ++increaseKawase)    // Run increasing image & kawase blur -------------------------------
    {
        dim3 grid((transWidth + BLOCK - 1) / BLOCK, (transHeight + BLOCK - 1) / BLOCK);

        BilinearIncrease <<<grid, block >>> (d_rb_second, secondPitch, d_rb_first, firstPitch, d_width * 2, d_height * 2, d_width, d_height);
        CUDA_KERNEL_CHECK();
        BilinearIncrease <<<grid, block >>> (d_gb_second, secondPitch, d_gb_first, firstPitch, d_width * 2, d_height * 2, d_width, d_height);
        CUDA_KERNEL_CHECK();
        BilinearIncrease <<<grid, block >>> (d_bb_second, secondPitch, d_bb_first, firstPitch, d_width * 2, d_height * 2, d_width, d_height);
        CUDA_KERNEL_CHECK();

        transWidth  *= 2;
        transHeight *= 2;
        d_width     *= 2;
        d_height    *= 2;

        CUDA_CHECK(cudaMemcpy2DToArray(d_rb_array, 0, 0, d_rb_second, secondPitch, sizeof(uchar) * h_width, h_height, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy2DToArray(d_gb_array, 0, 0, d_gb_second, secondPitch, sizeof(uchar) * h_width, h_height, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy2DToArray(d_bb_array, 0, 0, d_bb_second, secondPitch, sizeof(uchar) * h_width, h_height, cudaMemcpyDeviceToDevice));

        rbResDesc.res.array.array = d_rb_array;
        gbResDesc.res.array.array = d_gb_array;
        bbResDesc.res.array.array = d_bb_array;

        CUDA_CHECK(cudaCreateTextureObject(&texObjectRb, &rbResDesc, &rbTexDesc, nullptr));
        CUDA_CHECK(cudaCreateTextureObject(&texObjectGb, &gbResDesc, &gbTexDesc, nullptr));
        CUDA_CHECK(cudaCreateTextureObject(&texObjectBb, &bbResDesc, &bbTexDesc, nullptr));

        dim3 kawaseGrid((transWidth + BLOCK - 1) / BLOCK, (transHeight + BLOCK - 1) / BLOCK);
        KawaseBlur <<<kawaseGrid, block >>> (d_rb_first, firstPitch, texObjectRb, offset, d_width, d_height);
        CUDA_KERNEL_CHECK();
        KawaseBlur <<<kawaseGrid, block >>> (d_gb_first, firstPitch, texObjectGb, offset, d_width, d_height);
        CUDA_KERNEL_CHECK();
        KawaseBlur <<<kawaseGrid, block >>> (d_bb_first, firstPitch, texObjectBb, offset, d_width, d_height);
        CUDA_KERNEL_CHECK();

        --offset;
    }


    // Store resized image ======================================================================================

    CUDA_CHECK(cudaMemcpy2D(h_rb, sizeof(uchar) * h_width, d_rb_first, firstPitch, d_width, d_height, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(h_gb, sizeof(uchar) * h_width, d_gb_first, firstPitch, d_width, d_height, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy2D(h_bb, sizeof(uchar) * h_width, d_bb_first, firstPitch, d_width, d_height, cudaMemcpyDeviceToHost));

    const char* outPath2 = "C:\\Users\\james\\Documents\\2025\\source_code\\kawaseBluredLenna.bmp";
    if (!Bmp::RgbBuffersToBmp(outPath2, h_rb, h_gb, h_bb, h_width, h_height))
    {
        std::cout << "Error : Writing bmp file failed";
    }

    // free =====================================================================================================

    free(h_rb);
    free(h_gb);
    free(h_bb);
    cudaFree(d_rb_first);
    cudaFree(d_gb_first);
    cudaFree(d_bb_first);
    cudaFree(d_rb_second);
    cudaFree(d_gb_second);
    cudaFree(d_bb_second);
	cudaDestroyTextureObject(texObjectRb);
	cudaDestroyTextureObject(texObjectGb);
	cudaDestroyTextureObject(texObjectBb);
    cudaFreeArray(d_rb_array);
    cudaFreeArray(d_gb_array);
    cudaFreeArray(d_bb_array);

    return 0;
}
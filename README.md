# CUDA Kawase Blur 
이 프로젝트는 CUDA를 활용하여 이미지의 Bilinear 리사이징 ( 축소 및 확대 ) 및 Kawase Blur 필터를 구현한 예제입니다. 
BMP 이미지 파일을 입력으로 받아 다양한 이미지 처리 작업을 GPU 상에서 고속으로 수행합니다. 


---

##주요 기능
- Bilinear 이미지 축소
  - 원본 이미지를 Bilinear 보간법을 사용하여 1/2 크기로 축소합니다.  
- Bilinear 이미지 확대
  - 축소된 이미지를 Bilinear 보간법을 사용하여 2배 크기로 확대합니다. 
- Kawase Blur 필터
  - 텍스쳐 메모리를 사용하여 Kawase Blur 효과를 적용합니다.
  - 이 필터는 블러 강도를 조절할 수 있는 offset 매개변수를 가집니다.
- GPU 기반 이미지 처리
  - 모든 이미지 처리 작업은 CUDA 커널을 통해 NVIDIA GPU 상에서 병렬로 실행됩니다.


---

##프로젝트 구조
- main.cu : CUDA 커널 정의 및 호스트 코드 ( 메모리 할당, 데이터 전송, 커널 실행, 결과 저장 등 )
- BmpUtile.h : BMP 이미지 파일 로드 및 저장 유틸리티 ( 외부 라이브러리 또는 사용자 정의 헤더 파일로 가정 )


---

##코드 설명
###CUDA 커널
- BilinearReduce
  - 입력 이미지 픽셀을 2x2 블록으로 평균화하여 이미지의 한 픽셀을 계산합니다.
  - 원본 이미지의 크기를 절반으로 줄이는 데 사용됩니다.
- BilinearIncrease
  - 입력 이미지의 각 픽셀을 기준으로 2x2 블록을 Bilinear 보간하여 출력 이미지의 픽셀을 계산합니다.
  - 이미지 크기를 두배로 늘리는 데 사용됩니다.
- Kawase Blur
  - cudaTextureObject_t 을 사용하여 텍스처 메모리에서 주변 픽셀을 샘플링합니다.
  - offset 매개변수를 통해 블러의 강도를 조절합니다.
  - offset이 클수록 더 강한 블러 효과를 줍니다.

### main 함수 흐름
1. 호스트 데이터 설정
   - lenna.bmp 파일을 로드하여 R, G, B 채널별 데이터를 호스트 메모리에 할당합니다. 
2. 디바이스 데이터 설정
   - cudaMallocPitch 를 사용하여 R, G, B 채널별로 두 개의 디바이스 버퍼를 할당합니다.
   - pitch 메모리 할당은 2D 데이터 접근에 효율적입니다.
   - 원본 이미지 데이터를 첫 번째 디바이스 버퍼 ( d_rb_first, d_gb_first, d_bb_first )로 복사합니다.  
3. 텍스쳐 데이터 설정
   - cudaArray를 할당하고, cudaChannelFormatDesc 를 사용하여 채널 형식을 정의합니다.
   - cudaResourceDesc 및 cudaTextureDesc 를 설정하여 텍스처 객체를 생성합니다.
   - cudaCreateTextureObject 를 통해 텍스처 객체를 생성합니다. 
4. 이미지 처리 ( Reduce & Kawase Blur 루프 )
   - BLURING_TIMES 상수만큼 반복하면서 다음 작업을 수행합니다.
     - BilinearReduce 커널을 실행하여 이미지 축소합니다.
     - cudaMemcpy2DToArray를 사용하여 축소된 이미지를 cudaArray로 복사합니다.
     - 텍스처 객체를 업데이트하고 KawaseBlur 커널을 실행하여 블러를 적용합니다.
     - offset 값을 증가시켜 다음 블러 패스에서 더 넓은 영역을 샘플링하도록 합니다.  
5. 이미지 처리 ( Increase & Kawase Blur 루프 )
    -  BLURING_TIMES 상수만큼 반복하면서 다음 작업을 수행합니다.
      - BilinearIncrease 커널을 실행하여 이미지를 확대합니다.
      - cudaMemcpy2DToArray를 사용하여 확대된 이미지를 cudaArray로 복사합니다.
      - 텍스처 객체를 업데이트하고 KawaseBlur 커널을 실행하여 블러를 적용합니다.
      - offset 값을 감소시켜 다음 블러 패스에서 더 좁은 영역을 샘플링하도록 합니다.  
6. 결과 저장
    - 처리된 R, G, B 채널 데이터를 호스트 메모리로 다시 복사하고, RgbBuffersToBmp 함수를 사용하여 KawaseBluredLenna.bmp 파일로 저장합니다.
7. 메모리 해제
    - 할당된 모든 호스트 및 디바이스 메모리를 해제합니다.


  ---

  ##상수 설정
  - RATIO : Bilinear 보간에 사용되는 비율 ( 현재는 1 )
  - BLOCK : CUDA 스레드 블록의 크기 ( 16 X 16 )
  - BLURING_TIMES
    -  이미지 축소 및 확대 단계와 각 단계에서 KawaseBlur가 적용되는 횟수
    -  이 값이 클수록 블러 효과가 더 강해집니다.

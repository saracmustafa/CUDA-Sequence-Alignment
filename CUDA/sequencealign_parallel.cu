/*
* sequencealign_parallel.cu
*
* IMPORTANT: 
* 
* The final version of this code has been developed by Mustafa ACIKARAOĞLU, Mustafa SARAÇ,
* Mustafa Mert ÖGETÜRK as term project of Parallel Programming (COMP 429) course. 
* Koç University's code of ethics can be applied to this code and liability can not be 
* accepted for any negative situation. Therefore, be careful when you get content from here.
*
* This parallel version of Sequence Alignment code has been 
* implemented using the source specified in the link below. 
* 
* Reference:
* Implementation of Sequence Alignment in C++ 
* URL: <https://codereview.stackexchange.com/questions/97825/implementation-of-sequence-alignment-in-c>
* 
* NOTE:
*  
* THIS SOURCE CODE CONTAINS TWO VERSIONS OF
* PARALLELIZATION PROCESS.
* 
* FIRST VERSION: 
* ONLY TWO PART OF THE SERIAL IMPLEMENTATION IS PARALLELIZED
* IN A SUCCESSFUL WAY. THIS IMPLEMENTATION CONTAINS 
* 'alphabet_matching_penalty' AND 'array_filling_1' FUNCTIONS. 
* 
* SECOND VERSION:
* AS WE MENTIONED IN THE FINAL REPORT, WE TRIED TO IMPLEMENT
* THE PARALLELIZED VERSION OF THE PART OF THE 'align' FUNCTION
* FROM THE SERIAL IMPLEMENTATION, WHICH DIAGONALLY TRAVERSES 
* THROUGH THE END OF THE ARRAY BY FINDING THE MINIMUM OF THE 
* CURRENT INDEX'S LEFT, TOP AND LEFT-TOP INDEXES.
* HOWEVER, BECAUSE OF THE RACE CONDITION, WE TRIED TO FOLLOW
* MANY DIFFERENT WAYS TO SOLVE THIS ISSUE, BUT WE COULD NOT
* SUCCEED IT. THEREFORE, WE COMMENTED OUT ALL OF THE CODE THAT
* IS RELATED TO THE 'align_filling_2 kernel'.
*
*
* For more detailed questions you can review our project report.
*  
* You can also contact me at this email address: msarac13@ku.edu.tr
* 
*/ 

#include <iostream>
#include <string>
#include <vector>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string.h>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

const size_t alphabets = 26;
static const double kMicro = 1.0e-6;

/*
 * Returns the current time
 */
double get_time();

/*
 * Loading a file into a char array
 */
char* load_file(char const* path);

/*
 * alpha_d[i][j] = penalty for matching the ith alphabet with the
 *               jth alphabet.
 * Here: Penalty for matching an alphabet with anoter one is 1
 *       Penalty for matching an alphabet with itself is 0
 */
__global__ void alphabet_matching_penalty(int *alpha_d);

/*
 * Returns the minimum integer
 */
int min(int a, int b, int c);

/*
 * Filling the first row and the first
 * column of the array based on the gap
 * penalty, which is equal to 2.
 */
__global__ void align_filling_1(size_t n, size_t m, int *A, int alpha_gap);

/*
 * COMMENTED OUT:
 *
 * Align_filling_2 is diagonally traversing by 
 * finding the minimum value among the current
 * index's left, top and left-top indexes      
 * through the end of the array. 

__global__ void align_filling_2(size_t n, size_t m, char* input_1_d,
    char* input_2_d, int *alpha_d, int *A, int alpha_gap);
*/

int main()
{
    double time_0, time_1, time_2, time_3, time_4, time_5;

    int *alpha_h, *alpha_d, *array_h, *array_d;
    char *input_1, *input_2;
    string a_aligned, b_aligned;

    /* 
     * COMMENTED OUT:
     *
     * Device char arrays that will be used in the 
     * align_filling_2 kernel.

    char *input_1_d, *input_2_d;

    */

    time_0 = get_time();

    // Reading the input strings that need to be aligned
    input_1 = load_file("DNA_Sequence_1.txt");
    input_2 = load_file("DNA_Sequence_2.txt");

    size_t n = strlen(input_1);
    size_t m = strlen(input_2);

    // Penalty for any alphabet matched with a gap
    int gap_penalty = 2;

    // Allocation
    alpha_h = (int *) malloc(sizeof(int) * alphabets * alphabets);
    array_h = (int *) malloc(sizeof(int) * (n + 1) * (m + 1));

    if(cudaSuccess != cudaMalloc((void**) &array_d, sizeof(int) * (n + 1) * (m + 1))){
        cout << "Cuda Malloc error for array_d." << endl;
    }

    if(cudaSuccess != cudaMalloc((void**) &alpha_d, sizeof(int) * alphabets * alphabets)){
        cout << "Cuda Malloc error for alpha_d." << endl;
    }

    /*
     * COMMENTED OUT:
     *
     * Memory Allocations for the arrays that will be used in the 
     * align_filling_2 kernel.
     *

    if(cudaSuccess != cudaMalloc((void**) &input_1_d, sizeof(int) * n)){
        cout << "Cuda Malloc error for input_1_d." << endl;
    }

    if(cudaSuccess != cudaMalloc((void**) &input_2_d, sizeof(int) * m)){
        cout << "Cuda Malloc error for input_2_d." << endl;
    }

    */


    // MEMORY COPYING FROM HOST TO THE DEVICE
    if(cudaSuccess != cudaMemcpy(array_d, array_h, sizeof(int) * (n + 1) * (m + 1), cudaMemcpyHostToDevice)){
        cout << "Cuda Memory Copying error from array_h to array_d." << endl;
    }      

    if(cudaSuccess != cudaMemcpy(alpha_d, alpha_h, sizeof(int) * alphabets * alphabets, cudaMemcpyHostToDevice)){
        cout << "Cuda Memory Copying error from alpha_h to alpha_d." << endl;
    }

    /*
     * COMMENTED OUT:
     *
     * Memory Copying from Host to Device for the arrays that will be used 
     * in the align_filling_2 kernel.
     *

    if(cudaSuccess != cudaMemcpy(input_1_d, input_1, sizeof(int) * n, cudaMemcpyHostToDevice)){
        cout << "Cuda Memory Copying error from input_1 to input_1_d." << endl;
    }

    if(cudaSuccess != cudaMemcpy(input_2_d, input_2, sizeof(int) * m, cudaMemcpyHostToDevice)){
        cout << "Cuda Memory Copying error from input_2 to input_2_d." << endl;
    }

    */
    
    time_1 = get_time();


    dim3 threads(alphabets, alphabets, 1);
    dim3 grid(alphabets * alphabets/threads.x, alphabets * alphabets/threads.y);

    alphabet_matching_penalty<<<grid, threads>>>(alpha_d);

    cudaDeviceSynchronize();

    if(cudaGetLastError() != cudaSuccess){
        cout << "Kernel alphabet_matching_penalty was not launched." << endl;
    }

    time_2 = get_time();

    dim3 threads1(16, 16, 1);
    dim3 grid1(n / threads1.x, m / threads1.y);

    align_filling_1<<<grid1,threads1>>>(n, m, array_d, gap_penalty);

    cudaDeviceSynchronize();

    if(cudaGetLastError() != cudaSuccess){
        cout << threads1.x << endl;
        cout << threads1.y << endl;
        cout << "Kernel filling_1 was not launched." << endl;
    }

    time_3 = get_time();

    /*
     * COMMENTED OUT:
     *
     * Memory Allocations for the arrays that will be used in the 
     * align_filling_2 kernel.
     *
    align_filling_2<<<grid1,threads1>>>(n, m, input_1_d, input_2_d, alpha_d, array_d, gap_penalty);

    cudaDeviceSynchronize();

    if(cudaGetLastError() != cudaSuccess){
        cout << threads1.x << endl;
        cout << threads1.y << endl;
        cout << "Kernel filling_2 was not launched." << endl;
    }

    time_4 = get_time();
    */

    // MEMORY COPYING FROM DEVICE TO THE HOST
    if(cudaSuccess != cudaMemcpy(array_h, array_d, sizeof(int) * (n + 1) * (m + 1), cudaMemcpyDeviceToHost)){
        cout << "Cuda Memory Copying error from array_d to array_h." << endl;
    }

    if(cudaSuccess != cudaMemcpy(alpha_h, alpha_d, sizeof(int) * alphabets * alphabets, cudaMemcpyDeviceToHost)){
        cout << "Cuda Memory Copying error from alpha_d to alpha_h." << endl;
    }

    /*
     * COMMENTED OUT:
     *
     * Memory Copying from Device to Host for the arrays, which were 
     * used in the align_filling_2 kernel, 
     *

    if(cudaSuccess != cudaMemcpy(input_1_d, input_1, sizeof(int) * n, cudaMemcpyDeviceToHost)){
        cout << "Cuda Memory Copying error from input_1 to input_1_d." << endl;
    }

    if(cudaSuccess != cudaMemcpy(input_2_d, input_2, sizeof(int) * m, cudaMemcpyDeviceToHost)){
        cout << "Cuda Memory Copying error from input_2 to input_2_d." << endl;
    }

    */


    /*
     * After the first filling step is finished,
     * the function is diagonally traversing by 
     * finding the minimum value among the current
     * index's left, top and left-top indexes 
     * through the end of the array. 
     */
    for (size_t i = 1; i <= n; ++i)
    {
        for (size_t j = 1; j <= m; ++j)
        {
            char x_i = input_1[i-1];
            char y_j = input_2[j-1];
            array_h[i * n + j] = min(array_h[(i-1) * n + (j-1)] + alpha_h[(x_i - 'A') * alphabets + (y_j - 'A')],
                          array_h[(i-1) * n + j] + gap_penalty,
                          array_h[i * n + (j-1)] + gap_penalty);
        }
    }


    /*
     * After the second filling step is finished,
     * the function is diagonally tracebacking 
     * through the beginning of the array and it 
     * is generating the output strings, which are
     * the aligned DNA sequences.
     */
    long k;

    size_t i = n;
    size_t j = m;

    for (; i >= 1 && j >= 1; --i)
    {
      k= i * n + j;

        char x_i = input_1[i-1];
        char y_j = input_2[j-1];
        if (array_h[k] == array_h[(i-1)*n + (j-1)] + alpha_h[(x_i - 'A') * alphabets + (y_j - 'A')])
        {
            a_aligned = x_i + a_aligned;
            b_aligned = y_j + b_aligned;
            --j;
        }
        else if (array_h[k] == array_h[(i-1)*n + j] + gap_penalty)
        {
            a_aligned = x_i + a_aligned;
            b_aligned = '-' + b_aligned;
        }
        else
        {
            a_aligned = '-' + a_aligned;
            b_aligned = y_j + b_aligned;
            --j;
        }
    }

    while (i >= 1 && j < 1)
    {
        a_aligned = input_1[i-1] + a_aligned;
        b_aligned = '-' + b_aligned;
        --i;
    }
    while (j >= 1 && i < 1)
    {
        a_aligned = '-' + a_aligned;
        b_aligned = input_2[j-1] + b_aligned;
        --j;
    }

    time_4 = get_time();

    /*
     * Needleman Score that represents the similarity 
     * between the DNA sequences.
     */
    int needleman_score = array_h[n * m - 1];

    ofstream outputFile;
    outputFile.open("output_file_cuda_v1.txt");
    outputFile << a_aligned << endl << b_aligned << endl;
    outputFile.close();
    
    free(alpha_h);
    free(array_h);

    cudaFree(alpha_d);
    cudaFree(array_d);

    /*
     * COMMENTED OUT:
     *
     * Freeing the device arrays

    cudaFree(input_1_d);
    cudaFree(input_2_d);
    */

    time_5 = get_time();

    // print
    printf("Time for mallocs and memcopies: %9.6f s\n", (time_1 - time_0));
    printf("Time for alphabet_matching_penalty: %9.6f s\n", (time_2 - time_1));
    printf("Time for filling_1: %9.6f s\n", (time_3 - time_2));
    printf("Time for filling_2 and get_traceback: %9.6f s\n", (time_4 - time_3));
    printf("Needleman score : %d\n",needleman_score);
    printf("Total time: %9.6f s\n", (time_5 - time_0));

    return 0;
}

double get_time() {
    struct timeval TV;
    struct timezone TZ;
    const int RC = gettimeofday(&TV, &TZ);
    if(RC == -1) {
        printf("ERROR: Bad call to gettimeofday\n");
        return(-1);
    }
    return( ((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec) );
}

char* load_file(char const* path) {
    char* buffer = 0;
    long length;
    FILE * f = fopen (path, "rb"); //was "rb"

    if (f)
    {
      fseek (f, 0, SEEK_END);
      length = ftell (f);
      fseek (f, 0, SEEK_SET);
      buffer = (char*)malloc ((length+1)*sizeof(char));
      if (buffer)
      {
        fread (buffer, sizeof(char), length, f);
      }
      fclose (f);
    }
    buffer[length] = '\0';
    if(strlen(buffer) == 1){ printf("Failed to read the file"); }

    return buffer;
}

__global__ void alphabet_matching_penalty(int *alpha_d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    long k;

    if(i < alphabets && j < alphabets){
        k = i * alphabets + j;
        if (i == j) {
            alpha_d[k] = 0;
        } else {
            alpha_d[k] = 1;
        }
     } else {
        return;
     }
}

int min(int a, int b, int c)
{
    return std::min(std::min(a,b), c);
}

__global__ void align_filling_1(size_t n, size_t m, int *A, int alpha_gap)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(i <= n && j <= m){
        A[i*n] = alpha_gap * i;
        A[j] = alpha_gap * j;
    } else {
        return;
    }
}

/*
 * COMMENTED OUT:
 * 
 * Align_filling_2 is diagonally traversing by 
 * finding the minimum value among the current
 * index's left, top and left-top indexes      
 * through the end of the array. 

__global__ void align_filling_2(size_t n, size_t m, char* input_1_d,
    char* input_2_d, int *alpha_d, int *A, int alpha_gap)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if(((i >=1) && (i <= n)) && ((j >= 1) && (j <= m))){
            char x_i = input_1_d[i-1];
            char y_j = input_2_d[j-1];

            int first = A[(i-1) * n + (j-1)] + alpha_d[(x_i - 'A') * alphabets + (y_j - 'A')];
            int second = A[(i-1) * n + j] + alpha_gap;
            int third = A[i * n + (j-1)] + alpha_gap;
         
            if(first < second && first < third){
                A[i * n + j] = first;
            } else if(second < first && second < third){
                A[i * n + j] = second;
            } else if(third < first && third < second){
                A[i * n + j] = third;
            }

        }

}

*/

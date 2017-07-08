/*
* sequencealign_serial.cpp
* 
* IMPORTANT: 
* 
* The final version of this code has been developed by Mustafa ACIKARAOĞLU, Mustafa SARAÇ,
* Mustafa Mert ÖGETÜRK as term project of Parallel Programming (COMP 429) course. 
* Koç University's code of ethics can be applied to this code and liability can not be 
* accepted for any negative situation. Therefore, be careful when you get content from here.
*
* This serial version of Sequence Alignment code has been modified 
* using the source specified in the link below.
*
* In order to better observe the difference between serial and parallel
* implementation of this algorithm, we also put this file.
* 
* Reference:
* Implementation of Sequence Alignment in C++ 
* URL: <https://codereview.stackexchange.com/questions/97825/implementation-of-sequence-alignment-in-c>
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
#include <fstream>
#include <sstream>

using namespace std;

const size_t alphabets = 26;
static const double kMicro = 1.0e-6;

/*
 * Returns the current time
 */
double get_time();

/*
 * Returns the Needleman-Wunsch score for the best alignment of a and b
 * and stores the aligned sequences in a_aligned and b_aligned
 */
int align(const string &a, const string &b, int alpha_gap,
          int *alpha, string &a_aligned,
          string &b_aligned);

/*
 * Returns the minimum integer
 */
int min(int a, int b, int c);

int main()
{
    double time_0, time_1;

    time_0 = get_time();
    
    // The input strings that need to be aligned
    ifstream sequence_file_1("DNA_Sequence_1.txt");
    ifstream sequence_file_2("DNA_Sequence_2.txt");
    
    stringstream buffer_1;
    stringstream buffer_2;
    
    buffer_1 << sequence_file_1.rdbuf();
    buffer_2 << sequence_file_2.rdbuf();
    
    std::string input_1, input_2;
    input_1 = buffer_1.str();
    input_2 = buffer_2.str();
    
    // Penalty for any alphabet matched with a gap
    int gap_penalty = 2;
    
    int alpha[alphabets * alphabets];
    long k;

    for (size_t i = 0; i < alphabets; ++i)
    {
        for (size_t j = 0; j < alphabets; ++j)
        {
            k = i * alphabets + j;
            if (i == j) alpha[k] = 0;
            else alpha[k] = 1;
        }
    }
    
    // Aligned sequences
    string output_1, output_2;
    
    int penalty = align(input_1, input_2, gap_penalty, alpha, output_1, output_2);
    
    ofstream outputFile;
    outputFile.open("output_file_serial.txt");
    outputFile << output_1 << endl << output_2 << endl;
    outputFile.close();
    
    cout << "\nNeedleman-Wunsch Score: " << penalty << endl;
    
    time_1 = get_time();
    
    printf("Total time: %9.6f s\n", (time_1 - time_0));
    
    return 0;
}


int align(const string &a, const string &b, int alpha_gap,
          int *alpha, string &a_aligned,
          string &b_aligned)
{
    size_t n = a.size();
    size_t m = b.size();
    
    vector<vector<int> > A(n + 1, vector<int>(m + 1));
    
    /*
     * Filling the first row and the first
     * column of the array based on the gap
     * penalty, which is equal to 2.
     */
    for (size_t i = 0; i <= n; ++i)
        A[i][0] = alpha_gap * i;
    for (size_t i = 0; i <= m; ++i)
        A[0][i] = alpha_gap * i;


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
            char x_i = a[i-1];
            char y_j = b[j-1];
            A[i][j] = min(A[i-1][j-1] + alpha[(x_i - 'A') * alphabets + (y_j - 'A')],
                          A[i-1][j] + alpha_gap,
                          A[i][j-1] + alpha_gap);
        }
    }
    

    /*
     * After the second filling step is finished,
     * the function is diagonally tracebacking 
     * through the beginning of the array and it 
     * is generating the output strings, which are
     * the aligned DNA sequences.
     */
    a_aligned = "";
    b_aligned = "";
    size_t j = m;
    size_t i = n;
    for (; i >= 1 && j >= 1; --i)
    {
        char x_i = a[i-1];
        char y_j = b[j-1];
        if (A[i][j] == A[i-1][j-1] + alpha[(x_i - 'A') * alphabets + (y_j - 'A')])
        {
            a_aligned = x_i + a_aligned;
            b_aligned = y_j + b_aligned;
            --j;
        }
        else if (A[i][j] == A[i-1][j] + alpha_gap)
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
        a_aligned = a[i-1] + a_aligned;
        b_aligned = '-' + b_aligned;
        --i;
    }
    while (j >= 1 && i < 1)
    {
        a_aligned = '-' + a_aligned;
        b_aligned = b[j-1] + b_aligned;
        --j;
    }

    return A[n][m];
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

int min(int a, int b, int c)
{
    if (a <= b && a <= c)
        return a;
    else if (b <= a && b <= c)
        return b;
    else
        return c;
}

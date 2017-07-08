# CUDA-Sequence-Alignment

:computer: &nbsp;**Fall 2016 COMP 429 Parallel Programming Course Project, Koç University**

The final version of this code has been developed by **Mustafa ACIKARAOĞLU**, **Mustafa SARAÇ** and **Mustafa Mert ÖGETÜRK** as a project of Parallel Programming (COMP 429) course. **Koç University's code of ethics can be applied to this code and liability can not be accepted for any negative situation. Therefore, be careful when you get content from here.**

In order to implement the serial version of the sequence alignment algorithm, the following source was taken reference at some points.
<br>**Reference:** [Implementation of Sequence Alignment in C++ ](<https://codereview.stackexchange.com/questions/97825/implementation-of-sequence-alignment-in-c>)

**Description:** The serialized version of the sequence alignment algorithm was parallelized using various methods via CUDA.
### IMPORTANT: 

**This source code contains two versions of parallelization process.**

**First Version:** Only two part of the serial implementation is parallelized in a successful way. This implementation contains ***alphabet_matching_penalty*** and ***array_filling_1*** functions.
 
**Second Version:** We tried to implement the parallelized version of the part of the ***align*** function from the serial implementation, which diagonally traverses through the end of the array by finding the minimum of the current index's left, top and left-top indexes. However, even though we tried to follow many different ways to solve this issue, because of the race condition we could not succeed it. Therefore, we commented out all of the code that is related to the ***align_filling_2 kernel***.
 
You can also view our **project report**.

#### For more detailed questions, you can contact me at this email address: msarac13@ku.edu.tr &nbsp;&nbsp;:email:

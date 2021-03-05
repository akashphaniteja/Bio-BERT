## Installation

`pip install -e .`

[**Collab Example**](https://colab.research.google.com/drive/1iSWsKSRxGd3ObNU7KvsanKMjFoOa_HWP?usp=sharing)

## Points to be Noted
There are **3 formats** of the data
1. **Raw** Labelbox export data - json/csv, Indexing of annotation is **not** python based - [start:end] **end is inclusive**
  - See Image below, Left is json, right is csv
 
    ![image](https://user-images.githubusercontent.com/45713796/109597305-f8a95300-7b3d-11eb-9f0b-e7ea918244b4.png)
    
2. Dataframe format which the scripts use to process the data, **indexing is python based**, See example below

  ![image](https://user-images.githubusercontent.com/45713796/109598058-8127f380-7b3e-11eb-94ec-0107c99b1a47.png)

  
3. **Jsonl** format used as input to evaluation scripts, Indexing is python based

   ![image](https://user-images.githubusercontent.com/45713796/109599292-f268a600-7b40-11eb-8678-6bcb670846c3.png)

## Pipeline
- **Training**

![image](https://user-images.githubusercontent.com/45713796/109605715-9d7e5d00-7b4b-11eb-83d5-02174bb33386.png)

- **Evaluation**

![image](https://user-images.githubusercontent.com/45713796/109647221-0af7b100-7b7f-11eb-8ecd-3fe3e85fc343.png)


  


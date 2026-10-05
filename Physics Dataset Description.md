### **Dataset**

The data are drawn from ViPPS (Vietnamese Physics Problem Solving), a multimodal dataset for physics problem solving in Vietnamese. The version used for this challenge consists exclusively of the **Physics\_Dataset\_no\_image**, which contains Vietnamese problems focused on electricity and magnetism.

The dataset consists of text-only problems whose information is presented entirely in natural language, requiring models to rely on reading comprehension, physical reasoning, and mathematical calculation. It covers topics such as Coulomb forces, electric fields, capacitors, electric and magnetic field energy, magnetic fields of solenoids, alternating-current circuits and resonance, and measurement error. Each problem is accompanied by a reference solution written in four explicit steps: identifying the given information, selecting the relevant formulas, substituting values and computing, and stating the conclusion. The dataset contains a total of 1,355 problems.

The dataset will be divided into training, development, and test sets, and ground-truth answers for the test set will be withheld and used for official evaluation.

### **Data format**

The dataset is provided as a CSV file, containing the following headers: `id,question,cot,answer,unit`

An example of a row in the dataset might look like this: 

TD401,Tính năng lượng điện trường tích trữ trong tụ C khi C \= 100 μF dưới U \= 30 V,"1. Nhận diện dữ kiện:

\- Điện dung tụ C \= 100 μF \= 100 \* 10^-6 F, hiệu điện thế U \= 30 V.

\- Cần tính năng lượng điện trường tích trữ trong tụ.

2\. Công thức:

\- Công thức tính năng lượng điện trường tích trữ trong tụ điện là W \= (1/2) \* C \* U^2.

3\. Thay số và tính toán:

\- W \= (1/2) \* (100 \* 10^-6) \* (30^2) \= 0.5 \* 100 \* 10^-6 \* 900 \= 0.045 J.

4\. Kết luận:

\- Năng lượng điện trường tích trữ trong tụ là 0.045 J.",0.045,Jn

The principal fields are:

* `id`: A unique identifier for the physics problem.  
* `question`: The text of the physics problem written in Vietnamese.  
* `cot`: The reference solution (Chain of Thought). This is a detailed text string that walks through the reasoning process in four explicit steps: identifying the given information, selecting the relevant formulas, substituting values and computing, and stating the conclusion.  
* `answer`: The final numerical answer derived from the calculations or an expression/short text answer.  
* `unit`: The physical unit associated with the final answer (e.g., `V`, `A`, `J`, `Hz`).

**Link dataset: [ViPRA Materials](https://drive.google.com/drive/folders/1dyvQ3lZAGefMO4XAilnHs8MqEDqIm_hM?usp=sharing)**

**Submission format**

Each team submits an API endpoint together with a single ZIP file, \<team\_name\>.zip, containing:

* solution.pdf: a one-page description of the datasets used, the approach, and the model-size calculation;  
* source\_code.zip: the source code;  
* urls.txt: the prediction URL and all /v1/models URLs;  
* notation\_mapping.csv: the completed Notation Mapping CSV.

The input/output format of the API will be specified later in the [ViPRA Materials](https://drive.google.com/drive/folders/1dyvQ3lZAGefMO4XAilnHs8MqEDqIm_hM?usp=sharing)

**Evaluation metrics**

The final score is evaluated by three criteria, weights will be announced later:

* **P1 \- Correctness**: Automated matching of the final answer and unit against the ground truth. Details of the numerical tolerance will be announced later.  
* **P2 \- Explanation Quality**: The committee reviews clarity, coherence, and faithfulness of the natural-language explanations.  
* **P3 \- Reasoning Depth**: Extra credit for verifiable reasoning evidence such as FOL derivations, Chain-of-Thought steps, or cited premises. Details of how this criterion is evaluated will be announced later. 

**Zalo: https\://zalo.me/g/wg5vwforpbvakwstk5xw**


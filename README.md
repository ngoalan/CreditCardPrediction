**Credit Card Payment Default Model**

Have you ever wondered how banks/credit card companies plan their finances for their future? They may encounter a new set of people defaulting on their payments every month. They need to have a predictive model to help them with this information that will let them plan their finances strategically. We aim to predict if a person will default on his upcoming month’s credit card payment based on parameters such as Credit limit, sex, education level, marital status, age, and past 6 months payment history.

Dataset: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

**Questions / Goals**: We first plan on performing EDA and get a sense of the data and other interesting patterns/ information. Our primary goal is to use different supervised classification methods and compare their results. We also aim to perform clustering analysis which may reveal segments such as (High Credit Limit, Low Risk) Individuals, (High Credit Limit, High Risk) Individuals, (Low Credit Limit, Low Risk) Individuals, (Low Credit Limit, High Risk) individuals, / Responsible Payers, Delayed Payers, etc. Hypothesis: People who did not pay their bill the majority of the times in the past 6 months will default the next month irrespective of their age, sex, marital status, etc Hypothesis: People with a higher credit are less likely to default

**Algorithms**: We plan on using the following algorithms: Supervised: KNN Classification Decision Trees / Random Forest Logistic Regression Naive Bayes

**Unsupervised**: K Means Clustering

**Note**: Used algorithms are subject to change.

**Project Description**
This project aims to predict credit card payment defaults using factors like credit limit, demographics, and past payment history, utilizing a dataset from the UCI Machine Learning Repository. It will compare various supervised classification algorithms such as KNN, Decision Trees/Random Forest, Logistic Regression, Naive Bayes, and employ K Means Clustering for customer segmentation. The goal is to identify patterns that indicate the likelihood of default, assisting financial institutions in risk management and strategic planning.

<img width="871" alt="Screenshot 2025-01-24 at 2 04 15 AM" src="https://github.com/user-attachments/assets/a4234333-75aa-44ea-89c7-085d34db9f37" />

## Conclusion

The models had a higher accuracy on the original dataset than on the oversampled dataset. However, due to the imbalance in the dataset, the precision, recall, and F1 scores were not optimal. A model should not always be judged solely by its accuracy. Precision, recall, and F1 scores are critical metrics, and the choice of the best model depends on the context and specific business needs.

### Example: Understanding Contextual Model Selection

#### Department A: New Credit / Loan Approval
- **Prefers Precision**
  - **Why Precision Matters**:
    - **Goal**: Avoid mistakenly denying credit to individuals who are likely to pay back.
    - **Problem With Being Wrong**: Incorrectly predicting someone as a defaulter might unfairly deny them credit or loans.
    - **Solution**: Ensure high precision when predicting someone as a potential defaulter.
    - **Result**: Fewer mistakes → Fewer unfair credit denials → Happy Customers.

#### Department B: Credit Risk Management
- **Prefers Recall**
  - **Why Recall Matters**:
    - **Goal**: Identify every person likely to default to prevent financial losses.
    - **Problem With Missing Someone**: Missing a likely defaulter could lead to significant financial losses.
    - **Solution**: Focus on catching as many potential defaulters as possible.
    - **Result**: More defaulters identified → Reduced Financial Risk → Happy Stakeholders.

---

### Model Evaluation

1. **Random Forest**
   - Precision and recall significantly improved after oversampling.
   - Good overall accuracy, precision, and recall.
   - **Suitable for**: Both Department A and B.

2. **K-Nearest Neighbors (KNN)**
   - Poor recall performance on the original dataset due to class imbalance.
   - Better recall than precision after oversampling.
   - **Suitable for**: Department B.

3. **Logistic Regression**
   - Poor performance on the original dataset due to class imbalance.
   - Better recall than precision after oversampling.
   - **Suitable for**: Department B.

4. **Naive Bayes**
   - Best recall but sub-par accuracy and precision.
   - **If recall is the primary requirement**, this model is the best choice.
   - **Suitable for**: Department B.

---

### Suggestions for Improvement
The overall performance of these models can be enhanced by:
- Careful feature selection.
- Better sampling techniques.
- Hyperparameter tuning.
